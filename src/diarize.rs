//! 说话人识别（声纹）：pyannote 分割 + CAM++ 声纹向量聚类，区分"谁在说"；
//! 再和声纹库比对，认出登记过的人（如常上课的老师）。
//!
//! 声纹库：~/.local/share/livenotes/voiceprints/<名字>.json，每个文件是若干条声纹向量。

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
};

use crate::asr::SR;
use crate::llm::DeepSeek;
use crate::segment::Cue;

pub const SEGMENTATION_DIR: &str = "sherpa-onnx-pyannote-segmentation-3-0";
pub const EMBEDDING_MODEL: &str = "3dspeaker_campplus_zh_en_advanced.onnx";
/// 和声纹库比对的相似度阈值（余弦）
const MATCH_THRESHOLD: f32 = 0.75;
/// 每个说话人最多取多少秒的语音算声纹
const PROFILE_SECS: f32 = 60.0;

/// 一段连续的"某人在说"
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub speaker: usize,
}

/// 某个时间点是谁在说：落在哪个 Turn 里；落在空隙就取最近的
pub fn speaker_at(turns: &[Turn], t: f64) -> Option<usize> {
    if turns.is_empty() {
        return None;
    }
    let i = turns.partition_point(|x| x.end < t);
    let mut best: Option<(f64, usize)> = None;
    for x in turns[i.saturating_sub(1)..(i + 2).min(turns.len())].iter() {
        let d = if t < x.start {
            x.start - t
        } else if t > x.end {
            t - x.end
        } else {
            0.0
        };
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, x.speaker));
        }
    }
    best.map(|(_, s)| s)
}

/// 调参用：环境变量覆盖默认值
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub struct Diarizer {
    sd: OfflineSpeakerDiarization,
    extractor: SpeakerEmbeddingExtractor,
}

impl Diarizer {
    pub fn new(models: &Path, threads: i32, num_speakers: Option<usize>) -> Result<Self> {
        let seg = models.join(SEGMENTATION_DIR).join("model.onnx");
        let emb = models.join(EMBEDDING_MODEL);
        for p in [&seg, &emb] {
            if !p.exists() {
                bail!("缺少声纹模型 {}，先运行 install.sh", p.display());
            }
        }
        let s = |p: &Path| Some(p.to_string_lossy().into_owned());
        let embedding = SpeakerEmbeddingExtractorConfig {
            model: s(&emb),
            num_threads: threads,
            ..Default::default()
        };
        let cfg = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: s(&seg),
                    window_shift_ratio: env_f32("SUBS_SEG_SHIFT", 0.25),
                },
                num_threads: threads,
                ..Default::default()
            },
            embedding: embedding.clone(),
            clustering: FastClusteringConfig {
                num_clusters: num_speakers.map_or(-1, |n| n as i32),
                threshold: env_f32("SUBS_CLUSTER_THRESHOLD", 0.8),
                ..Default::default()
            },
            ..Default::default()
        };
        let sd = OfflineSpeakerDiarization::create(&cfg).context("加载说话人分离模型失败")?;
        let extractor =
            SpeakerEmbeddingExtractor::create(&embedding).context("加载声纹模型失败")?;
        Ok(Self { sd, extractor })
    }

    /// 整段音频 → 说话人片段（说话人按首次出现的顺序编号 0,1,2…）
    pub fn run(&self, samples: &[f32]) -> Result<Vec<Turn>> {
        let r = self.sd.process(samples).context("说话人分离失败")?;
        let mut order: Vec<i32> = Vec::new();
        let mut turns = Vec::new();
        for seg in r.sort_by_start_time() {
            let id = match order.iter().position(|&s| s == seg.speaker) {
                Some(i) => i,
                None => {
                    order.push(seg.speaker);
                    order.len() - 1
                }
            };
            turns.push(Turn {
                start: seg.start as f64,
                end: seg.end as f64,
                speaker: id,
            });
        }
        Ok(turns)
    }

    /// 一段音频的声纹向量
    pub fn embed(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(SR as i32, samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return None;
        }
        self.extractor.compute(&stream)
    }

    /// 每个说话人的声纹：取他最长的几段（共约 60 秒）拼起来算
    pub fn profiles(&self, samples: &[f32], turns: &[Turn]) -> Vec<Option<Vec<f32>>> {
        let n = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
        (0..n)
            .map(|spk| {
                let mut mine: Vec<&Turn> = turns.iter().filter(|t| t.speaker == spk).collect();
                mine.sort_by(|a, b| (b.end - b.start).total_cmp(&(a.end - a.start)));
                let mut audio = Vec::new();
                for t in mine {
                    if audio.len() as f32 / SR as f32 >= PROFILE_SECS {
                        break;
                    }
                    let (a, b) = (
                        (t.start * SR as f64) as usize,
                        ((t.end * SR as f64) as usize).min(samples.len()),
                    );
                    if a < b {
                        audio.extend_from_slice(&samples[a..b]);
                    }
                }
                (audio.len() >= SR).then(|| self.embed(&audio)).flatten()
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
struct Voiceprint {
    name: String,
    embeddings: Vec<Vec<f32>>,
}

/// 声纹库：登记过名字的人
pub struct VoiceBook {
    dir: PathBuf,
    entries: Vec<(String, Vec<Vec<f32>>)>,
}

impl VoiceBook {
    pub fn open(dir: PathBuf, dim: usize) -> Self {
        let mut entries = Vec::new();
        if let Ok(files) = std::fs::read_dir(&dir) {
            for e in files.flatten() {
                let Ok(text) = std::fs::read_to_string(e.path()) else {
                    continue;
                };
                let Ok(vp) = serde_json::from_str::<Voiceprint>(&text) else {
                    continue;
                };
                let good: Vec<Vec<f32>> = vp
                    .embeddings
                    .into_iter()
                    .filter(|v| v.len() == dim)
                    .collect();
                if !good.is_empty() {
                    entries.push((vp.name, good));
                }
            }
        }
        Self { dir, entries }
    }

    /// 和库里每个人的相似度（取这个人各条声纹里最像的一条），从高到低
    pub fn scores(&self, embedding: &[f32]) -> Vec<(String, f32)> {
        let mut v: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|(name, embs)| {
                let best = embs
                    .iter()
                    .map(|e| cosine(embedding, e))
                    .fold(f32::MIN, f32::max);
                (name.clone(), best)
            })
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }

    /// 声纹库里最像的人（超过阈值才算）
    pub fn identify(&self, embedding: &[f32]) -> Option<String> {
        self.scores(embedding)
            .into_iter()
            .next()
            .filter(|(_, s)| *s >= MATCH_THRESHOLD)
            .map(|(n, _)| n)
    }

    /// 把这个人的声纹存进库（同名追加，最多保留 10 条）
    pub fn enroll(&self, name: &str, embedding: &[f32]) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)?;
        let file: String = name
            .chars()
            .map(|c| if "/\\:*?\"<>|".contains(c) { '_' } else { c })
            .collect();
        let path = self.dir.join(format!("{file}.json"));
        let mut vp = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<Voiceprint>(&t).ok())
            .unwrap_or(Voiceprint {
                name: name.to_owned(),
                embeddings: Vec::new(),
            });
        vp.embeddings.push(embedding.to_vec());
        let extra = vp.embeddings.len().saturating_sub(10);
        vp.embeddings.drain(..extra);
        std::fs::write(&path, serde_json::to_string(&vp)?)?;
        Ok(path)
    }
}

// ===== 接入流水线：后台分离 + 标注 + 起名 =====

/// 占位名：说话人A、说话人B…（结束后按内容推断出真名再替换）
pub fn placeholder(i: usize) -> String {
    let letter = (b'A' + (i % 26) as u8) as char;
    format!("说话人{letter}")
}

/// 总时长太短（多为开场剪辑、插话、误分）的类，并进声纹最像的大类
const MIN_SPEAKER_SECS: f64 = 15.0;

pub struct SpeakerInfo {
    pub turns: Vec<Turn>,
    /// 每个说话人的显示名（声纹库认出的真名，或占位名）
    pub labels: Vec<String>,
    /// 声纹库认出来的（不需要 LLM 起名）
    pub known: Vec<bool>,
    pub profiles: Vec<Option<Vec<f32>>>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-9)
}

fn analyze(
    path: &Path,
    models: &Path,
    threads: i32,
    num_speakers: Option<usize>,
    voices: &Path,
) -> Result<SpeakerInfo> {
    let audio = crate::asr::decode_all(path)?;
    let d = Diarizer::new(models, threads, num_speakers)?;
    let mut turns = d.run(&audio)?;
    let mut profiles = d.profiles(&audio, &turns);
    drop(audio);

    // 合并零碎小类
    let n = profiles.len();
    let total = |turns: &[Turn], s: usize| -> f64 {
        turns
            .iter()
            .filter(|t| t.speaker == s)
            .map(|t| t.end - t.start)
            .sum()
    };
    let big: Vec<usize> = (0..n)
        .filter(|&s| total(&turns, s) >= MIN_SPEAKER_SECS)
        .collect();
    if !big.is_empty() {
        let mut remap: Vec<usize> = (0..n).collect();
        for s in (0..n).filter(|s| !big.contains(s)) {
            remap[s] = match &profiles[s] {
                Some(p) => *big
                    .iter()
                    .max_by(|&&a, &&b| {
                        let sa = profiles[a].as_ref().map_or(-1.0, |q| cosine(p, q));
                        let sb = profiles[b].as_ref().map_or(-1.0, |q| cosine(p, q));
                        sa.total_cmp(&sb)
                    })
                    .unwrap(),
                None => big[0],
            };
        }
        // 重新按首次出现编号
        let mut order: Vec<usize> = Vec::new();
        for t in &mut turns {
            let s = remap[t.speaker];
            t.speaker = match order.iter().position(|&x| x == s) {
                Some(i) => i,
                None => {
                    order.push(s);
                    order.len() - 1
                }
            };
        }
        profiles = order.iter().map(|&s| profiles[s].clone()).collect();
    }

    let dim = profiles.iter().flatten().next().map_or(0, Vec::len);
    let book = VoiceBook::open(voices.to_path_buf(), dim);
    if std::env::var("SUBS_DEBUG_SPK").is_ok() {
        for (i, p) in profiles.iter().enumerate() {
            let secs = total(&turns, i);
            let sc = p.as_ref().map(|p| book.scores(p)).unwrap_or_default();
            let sc: Vec<String> = sc.iter().map(|(n, s)| format!("{n}={s:.2}")).collect();
            eprintln!("  [spk] {} {:6.0}s  {}", placeholder(i), secs, sc.join(" "));
        }
        for i in 0..profiles.len() {
            for j in i + 1..profiles.len() {
                if let (Some(a), Some(b)) = (&profiles[i], &profiles[j]) {
                    eprintln!(
                        "  [sim] {}-{} {:.2}",
                        placeholder(i),
                        placeholder(j),
                        cosine(a, b)
                    );
                }
            }
        }
    }
    // 同一个人被拆成多类（长音频里常见）：并到最像的"锚点"
    anchor_merge(&mut turns, &mut profiles, env_f32("SUBS_MERGE_SIM", 0.7));

    // 声纹库认人
    // 声纹库里认成同一个人的类合并
    let ids: Vec<Option<String>> = profiles
        .iter()
        .map(|p| p.as_ref().and_then(|p| book.identify(p)))
        .collect();
    let mut first: HashMap<String, usize> = HashMap::new();
    let mut remap: Vec<usize> = (0..profiles.len()).collect();
    for (i, id) in ids.iter().enumerate() {
        if let Some(name) = id {
            remap[i] = *first.entry(name.clone()).or_insert(i);
        }
    }
    if remap.iter().enumerate().any(|(i, &r)| i != r) {
        let keep: Vec<usize> = (0..profiles.len()).filter(|&i| remap[i] == i).collect();
        for t in turns.iter_mut() {
            t.speaker = keep.iter().position(|&k| k == remap[t.speaker]).unwrap();
        }
        profiles = keep.iter().map(|&k| profiles[k].clone()).collect();
    }
    let mut labels = Vec::new();
    let mut known = Vec::new();
    for (i, p) in profiles.iter().enumerate() {
        match p.as_ref().and_then(|p| book.identify(p)) {
            Some(name) => {
                labels.push(name);
                known.push(true);
            }
            None => {
                labels.push(placeholder(i));
                known.push(false);
            }
        }
    }
    Ok(SpeakerInfo {
        turns,
        labels,
        known,
        profiles,
    })
}

/// "锚点"合并：按说话时长从长到短，和已有锚点都不像（< `min_sim`）的类自成一个新的人，
/// 否则并入最像的锚点。长音频里同一个人常被拆成好几类，而各类两两比较时不同人之间偶尔也很像，
/// 以时长最长的类为锚点比两两合并稳得多。
fn anchor_merge(turns: &mut [Turn], profiles: &mut Vec<Option<Vec<f32>>>, min_sim: f32) {
    let n = profiles.len();
    let secs: Vec<f64> = (0..n)
        .map(|s| {
            turns
                .iter()
                .filter(|t| t.speaker == s)
                .map(|t| t.end - t.start)
                .sum()
        })
        .collect();
    let mut by_len: Vec<usize> = (0..n).collect();
    by_len.sort_by(|&a, &b| secs[b].total_cmp(&secs[a]));
    let mut anchors: Vec<usize> = Vec::new();
    let mut target = vec![0usize; n];
    for &c in &by_len {
        let best = profiles[c].as_ref().and_then(|p| {
            anchors
                .iter()
                .filter_map(|&a| profiles[a].as_ref().map(|q| (a, cosine(p, q))))
                .max_by(|x, y| x.1.total_cmp(&y.1))
        });
        match best {
            Some((a, sim)) if sim >= min_sim => target[c] = a,
            None if !anchors.is_empty() && profiles[c].is_none() => target[c] = anchors[0],
            _ => {
                anchors.push(c);
                target[c] = c;
            }
        }
    }
    // 锚点按首次出现重新编号；声纹按成员时长加权平均
    let mut order: Vec<usize> = Vec::new();
    for t in turns.iter_mut() {
        let a = target[t.speaker];
        t.speaker = match order.iter().position(|&x| x == a) {
            Some(i) => i,
            None => {
                order.push(a);
                order.len() - 1
            }
        };
    }
    let merged: Vec<Option<Vec<f32>>> = order
        .iter()
        .map(|&a| {
            let members: Vec<usize> = (0..n)
                .filter(|&c| target[c] == a && profiles[c].is_some())
                .collect();
            let dim = profiles[a].as_ref()?.len();
            let w: f64 = members.iter().map(|&c| secs[c]).sum();
            let mut v = vec![0f32; dim];
            for &c in &members {
                for (x, y) in v.iter_mut().zip(profiles[c].as_ref().unwrap()) {
                    *x += y * (secs[c] / w.max(1e-6)) as f32;
                }
            }
            Some(v)
        })
        .collect();
    *profiles = merged;
}

/// 后台跑说话人分离；第一次要标注时才等它的结果（识别和翻译照常并行进行）
pub struct SpeakerJob {
    handle: Mutex<Option<JoinHandle<Result<SpeakerInfo>>>>,
    info: OnceLock<Option<SpeakerInfo>>,
    /// 标注过的每条字幕 (说话人, 英文)，结束后用来推断名字
    pub lines: Mutex<Vec<(String, String)>>,
}

impl SpeakerJob {
    pub fn start(
        path: PathBuf,
        models: PathBuf,
        threads: i32,
        num_speakers: Option<usize>,
        voices: PathBuf,
    ) -> Arc<Self> {
        let handle = std::thread::spawn(move || {
            let t = std::time::Instant::now();
            let r = analyze(&path, &models, threads, num_speakers, &voices);
            if let Ok(info) = &r {
                eprintln!(
                    "  🗣  区分出 {} 位说话人（用时 {:.0}s）",
                    info.labels.len(),
                    t.elapsed().as_secs_f64()
                );
            }
            r
        });
        Arc::new(Self {
            handle: Mutex::new(Some(handle)),
            info: OnceLock::new(),
            lines: Mutex::new(Vec::new()),
        })
    }

    pub fn info(&self) -> Option<&SpeakerInfo> {
        self.info
            .get_or_init(|| {
                let h = self.handle.lock().unwrap().take()?;
                match h.join() {
                    Ok(Ok(info)) => Some(info),
                    Ok(Err(e)) => {
                        eprintln!("  ! 说话人识别失败，不标注说话人: {e:#}");
                        None
                    }
                    Err(_) => None,
                }
            })
            .as_ref()
    }

    /// 给一条字幕标上说话人：按时长统计每个词落在谁的片段里，取最多的
    pub fn label(&self, c: &mut Cue) {
        let Some(info) = self.info() else { return };
        let mut votes: BTreeMap<usize, f64> = BTreeMap::new();
        for w in &c.words {
            if let Some(s) = speaker_at(&info.turns, (w.start + w.end) / 2.0) {
                *votes.entry(s).or_default() += (w.end - w.start).max(0.05);
            }
        }
        if let Some((&s, _)) = votes.iter().max_by(|a, b| a.1.total_cmp(b.1)) {
            c.speaker = info.labels.get(s).cloned();
            self.lines
                .lock()
                .unwrap()
                .push((c.speaker.clone().unwrap_or_default(), c.text()));
        }
    }
}

const NAMING_PROMPT: &str = "\
你根据对话内容判断每个说话人是谁。输入 JSON：title 是视频标题，hint 是主题提示，\
speakers 是需要起名的说话人占位名，known 是已经由声纹确定身份的人（对话里直接用真名标注），\
dialogue 是对话节选（由声纹自动区分，偶有标错）。
要求：
- 只根据对话里的称呼、自我介绍、问答关系和标题判断，不要凭空编造
- 能确定真名就写真名（如 \"Elon Musk\"）；确定不了真名就写角色
- 判断角色看问答关系：主要在提问、引导话题的是主持人/采访者（多位就写 \"主持人2\"、\"采访者\" 以区分），\
主要在回答的是嘉宾；课堂上讲授的是老师，提问的是学生
- 不同的说话人不要起同一个名字，也不能用 known 里已有的名字
- 输出 JSON 对象：{\"说话人A\": \"名字\", ...}，speakers 里每一个都要有";

/// 让 LLM 根据对话推断占位名对应的人。返回 占位名 → 名字。
pub fn name_speakers(
    llm: &DeepSeek,
    info: &SpeakerInfo,
    lines: &[(String, String)],
    title: &str,
    hint: Option<&str>,
) -> HashMap<String, String> {
    let unknown: Vec<&String> = info
        .labels
        .iter()
        .zip(&info.known)
        .filter(|(_, k)| !**k)
        .map(|(l, _)| l)
        .collect();
    if unknown.is_empty() || lines.is_empty() {
        return HashMap::new();
    }
    // 节选：开头 60 句（通常有介绍）+ 每个人各自的前 10 句
    let mut picked: Vec<usize> = (0..lines.len().min(60)).collect();
    for u in &unknown {
        picked.extend(
            lines
                .iter()
                .enumerate()
                .filter(|(_, (s, _))| s == *u)
                .map(|(i, _)| i)
                .take(10),
        );
    }
    picked.sort_unstable();
    picked.dedup();
    let dialogue: Vec<String> = picked
        .iter()
        .map(|&i| format!("{}: {}", lines[i].0, lines[i].1))
        .collect();
    let known: Vec<&String> = info
        .labels
        .iter()
        .zip(&info.known)
        .filter(|(_, k)| **k)
        .map(|(l, _)| l)
        .collect();
    let input = serde_json::json!({
        "title": title, "hint": hint.unwrap_or(""), "speakers": unknown, "known": known,
        "dialogue": dialogue.join("\n"),
    });
    match llm.chat_json(NAMING_PROMPT, &input.to_string(), 0.3) {
        Ok(v) => {
            let proposed: Vec<Option<String>> = unknown
                .iter()
                .map(|u| v[u.as_str()].as_str().map(|n| n.trim().to_owned()))
                .collect();
            dedupe_names(&unknown, proposed, &known)
        }
        Err(e) => {
            eprintln!("  ! 推断说话人名字失败，保留占位名: {e:#}");
            HashMap::new()
        }
    }
}

/// 兜底：LLM 给出的名字和已知的人或彼此重名时，第二个起改用 "名字（2）"，
/// 保证不同的人在文件里名字不同（之后还能用 --rename 改成真名）。
fn dedupe_names(
    unknown: &[&String],
    proposed: Vec<Option<String>>,
    known: &[&String],
) -> HashMap<String, String> {
    let mut used: Vec<String> = known.iter().map(|k| k.to_string()).collect();
    let mut out = HashMap::new();
    for (u, name) in unknown.iter().zip(proposed) {
        let Some(mut name) = name.filter(|n| !n.is_empty()) else {
            continue;
        };
        if used.contains(&name) {
            let base = name.clone();
            let mut k = 2;
            while used.contains(&name) {
                name = format!("{base}（{k}）");
                k += 1;
            }
        }
        used.push(name.clone());
        out.insert((*u).clone(), name);
    }
    out
}

/// 把目录里字幕、文字稿、笔记中的占位名换成名字（同时更新 speakers.json）
/// 同时替换多组名字（"说话人E→说话人D"和"说话人D→说话人A"同时生效，不会连锁替换）
fn replace_all_at_once(text: &str, pairs: &[(&String, &String)]) -> String {
    let mut out = text.to_owned();
    for (i, (from, _)) in pairs.iter().enumerate() {
        out = out.replace(from.as_str(), &format!("\u{E000}{i}\u{E001}"));
    }
    for (i, (_, to)) in pairs.iter().enumerate() {
        out = out.replace(&format!("\u{E000}{i}\u{E001}"), to);
    }
    out
}

pub fn rename_in_dir(dir: &Path, stem: &str, names: &HashMap<String, String>) -> Result<()> {
    if names.is_empty() {
        return Ok(());
    }
    if let Ok(mut entries) = read_sidecar(dir, stem) {
        for e in &mut entries {
            if let Some(to) = names.get(&e.name) {
                e.name = to.clone();
            }
        }
        write_sidecar(dir, stem, &entries)?;
    }
    // 长的先换，避免 "说话人A" 误伤（占位名都是等长的，这里只是保险）
    let mut pairs: Vec<(&String, &String)> = names.iter().collect();
    pairs.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    for e in std::fs::read_dir(dir)?.flatten() {
        let p = e.path();
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let ours = name.starts_with(stem)
            && (name.ends_with(".srt") || name.ends_with(".md") || name.ends_with(".html"));
        if !ours {
            continue;
        }
        let text = std::fs::read_to_string(&p)?;
        std::fs::write(&p, replace_all_at_once(&text, &pairs))?;
    }
    Ok(())
}

// ===== 会话里的说话人记录：<标题>.speakers.json =====
//
// 自动起的名字只是猜测，不直接进声纹库。用户核对（必要时 --rename-speaker 改名）之后，
// 再用 --enroll 把这里记下的声纹存进声纹库。

#[derive(Serialize, Deserialize, Clone)]
pub struct SpeakerEntry {
    pub name: String,
    /// 声纹库 / 按内容推断 / 未知
    pub source: String,
    pub secs: f64,
    pub embedding: Option<Vec<f32>>,
}

fn sidecar(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.speakers.json"))
}

pub fn write_sidecar(dir: &Path, stem: &str, entries: &[SpeakerEntry]) -> Result<()> {
    std::fs::write(sidecar(dir, stem), serde_json::to_string_pretty(entries)?)?;
    Ok(())
}

pub fn read_sidecar(dir: &Path, stem: &str) -> Result<Vec<SpeakerEntry>> {
    let p = sidecar(dir, stem);
    let text = std::fs::read_to_string(&p)
        .with_context(|| format!("没有找到 {}（这次处理没开说话人识别？）", p.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// 角色名、占位名、自动加了"（2）"的名字都不算确认过的真名，不存声纹
pub fn is_confirmed_name(name: &str) -> bool {
    let roles = ["主持人", "嘉宾", "采访者", "老师", "学生", "未知", "说话人"];
    !roles.iter().any(|r| name.contains(r)) && !name.ends_with('）')
}

/// 把会话里（已确认名字的）说话人声纹存进声纹库
pub fn enroll_session(dir: &Path, stem: &str, voices: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut saved = Vec::new();
    for e in read_sidecar(dir, stem)? {
        let Some(emb) = e.embedding.as_ref().filter(|_| is_confirmed_name(&e.name)) else {
            continue;
        };
        let book = VoiceBook::open(voices.to_path_buf(), emb.len());
        saved.push((e.name.clone(), book.enroll(&e.name, emb)?));
    }
    Ok(saved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renames_do_not_chain() {
        let (e, d, a) = (
            "说话人E".to_string(),
            "说话人D".to_string(),
            "说话人A".to_string(),
        );
        let pairs = [(&e, &d), (&d, &a)];
        assert_eq!(
            replace_all_at_once("说话人D：hi 说话人E：yo", &pairs),
            "说话人A：hi 说话人D：yo"
        );
    }

    #[test]
    fn speaker_lookup_handles_gaps() {
        let turns = [
            Turn {
                start: 0.0,
                end: 5.0,
                speaker: 0,
            },
            Turn {
                start: 6.0,
                end: 9.0,
                speaker: 1,
            },
            Turn {
                start: 20.0,
                end: 25.0,
                speaker: 0,
            },
        ];
        assert_eq!(speaker_at(&turns, 2.0), Some(0));
        assert_eq!(speaker_at(&turns, 7.0), Some(1));
        assert_eq!(speaker_at(&turns, 5.8), Some(1)); // 空隙里取最近的
        assert_eq!(speaker_at(&turns, 30.0), Some(0));
        assert_eq!(speaker_at(&[], 1.0), None);
    }

    #[test]
    fn anchor_merge_joins_fragments_to_longest_speaker() {
        // 0：甲（长）、1：乙（长）、2：甲的碎片、3：乙的碎片
        let mut turns = vec![
            Turn {
                start: 0.0,
                end: 100.0,
                speaker: 0,
            },
            Turn {
                start: 100.0,
                end: 160.0,
                speaker: 1,
            },
            Turn {
                start: 160.0,
                end: 170.0,
                speaker: 2,
            },
            Turn {
                start: 170.0,
                end: 180.0,
                speaker: 3,
            },
        ];
        let mut profiles = vec![
            Some(vec![1.0, 0.0, 0.0]),
            Some(vec![0.0, 1.0, 0.0]),
            Some(vec![0.9, 0.1, 0.3]),
            Some(vec![0.2, 0.95, 0.1]),
        ];
        anchor_merge(&mut turns, &mut profiles, 0.7);
        let who: Vec<usize> = turns.iter().map(|t| t.speaker).collect();
        assert_eq!(who, [0, 1, 0, 1]);
        assert_eq!(profiles.len(), 2);
    }

    #[test]
    fn duplicate_names_get_suffix() {
        let (a, b) = ("说话人B".to_string(), "说话人C".to_string());
        let known_name = "Dwarkesh Patel".to_string();
        let out = dedupe_names(
            &[&a, &b],
            vec![Some("Dwarkesh Patel".into()), Some("主持人".into())],
            &[&known_name],
        );
        assert_eq!(out["说话人B"], "Dwarkesh Patel（2）");
        assert_eq!(out["说话人C"], "主持人");
    }
}
