//! 实时模式的说话人标注：每个 VAD 段算一个声纹向量（CAM++），在线聚类成"说话人A/B…"。
//!
//! 离线模式能看到整段音频再聚类；实时模式只能边听边分，所以：
//! - 太短的段（声纹不可靠）沿用上一段的说话人；
//! - 和已有的人都不像、且足够长，才新开一个人；
//! - 两个人的平均声纹越来越像时合并（同一个人被拆开）；
//! - 结束时把太零碎的人并进最像的大类，按首次出现重新编号，给出"旧名 → 新名"的改名表，
//!   交给 `diarize::rename_in_dir` 改已经写出的文件。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

use crate::asr::SR;
use crate::diarize::{EMBEDDING_MODEL, VoiceBook, placeholder};

/// 在线聚类参数（余弦相似度 / 秒）
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// 和某人平均声纹的相似度 ≥ 这个值，直接算他
    pub assign: f32,
    /// 和所有人都 < 这个值（且段够长），新开一个人
    pub new_speaker: f32,
    /// 两个人的平均声纹 ≥ 这个值，合并
    pub merge: f32,
    /// 新开一个人至少要这么长的段
    pub min_new_secs: f64,
    /// 短于这个的段不看声纹，沿用上一段的人
    pub short_secs: f64,
    /// 至少这么长的段才更新平均声纹
    pub min_update_secs: f64,
    /// 结束时总时长不到这个的人并进最像的大类
    pub min_final_secs: f64,
    /// 换人的门槛：最像的人要比上一段的人高出这么多才换
    pub stickiness: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            assign: env_f32("SUBS_LIVE_SPK_ASSIGN", 0.55),
            new_speaker: env_f32("SUBS_LIVE_SPK_NEW", 0.35),
            merge: env_f32("SUBS_LIVE_SPK_MERGE", 0.7),
            min_new_secs: env_f32("SUBS_LIVE_SPK_MIN_NEW", 2.0) as f64,
            short_secs: env_f32("SUBS_LIVE_SPK_SHORT", 1.5) as f64,
            min_update_secs: 1.5,
            min_final_secs: env_f32("SUBS_LIVE_SPK_MIN_FINAL", 8.0) as f64,
            stickiness: env_f32("SUBS_LIVE_SPK_STICKY", 0.15),
        }
    }
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Cluster {
    /// 单位化声纹按时长加权求和
    sum: Vec<f32>,
    /// sum 单位化后的平均声纹
    centroid: Vec<f32>,
    /// 标给这个人的总时长
    secs: f64,
    /// 并进了哪一类
    merged_into: Option<usize>,
    /// 当前显示名（占位名，或声纹库认出的名字）
    name: String,
    /// 声纹库认出来的
    known: bool,
    /// 曾经用过的显示名（最后改名用）
    names_used: Vec<String>,
}

impl Cluster {
    fn add(&mut self, e: &[f32], w: f64) {
        for (s, x) in self.sum.iter_mut().zip(e) {
            *s += x * w as f32;
        }
        self.centroid = normalize(&self.sum);
    }
}

/// 纯聚类逻辑（不涉及模型），方便测试
pub struct Clusters {
    p: Params,
    list: Vec<Cluster>,
    prev: Option<usize>,
}

/// 结束时每个说话人的汇总
#[derive(Clone, Debug)]
pub struct SpeakerSummary {
    pub label: String,
    pub secs: f64,
    pub known: bool,
    pub profile: Vec<f32>,
}

impl Clusters {
    pub fn new(p: Params) -> Self {
        Self {
            p,
            list: Vec::new(),
            prev: None,
        }
    }

    fn root(&self, mut i: usize) -> usize {
        while let Some(j) = self.list[i].merged_into {
            i = j;
        }
        i
    }

    fn active(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.list.len()).filter(|&i| self.list[i].merged_into.is_none())
    }

    fn create(&mut self, e: &[f32], secs: f64) -> usize {
        let i = self.list.len();
        let name = placeholder(i);
        self.list.push(Cluster {
            sum: e.iter().map(|x| x * secs as f32).collect(),
            centroid: e.to_vec(),
            secs,
            merged_into: None,
            names_used: vec![name.clone()],
            name,
            known: false,
        });
        i
    }

    /// 一段语音（声纹向量可能没有：段太短算不出来）→ 说话人编号
    pub fn assign(&mut self, emb: Option<&[f32]>, secs: f64) -> Option<usize> {
        let prev = self.prev.map(|i| self.root(i));
        let chosen = match emb {
            None => prev,
            Some(_) if secs < self.p.short_secs && prev.is_some() => prev,
            Some(raw) => {
                let e = normalize(raw);
                let best = self
                    .active()
                    .map(|i| (i, dot(&e, &self.list[i].centroid)))
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                let prev_sim = prev.map(|i| dot(&e, &self.list[i].centroid));
                let pick = match best {
                    None if secs >= self.p.min_new_secs => Some(self.create(&e, 0.0)),
                    None => None,
                    // 和所有人都不像、又足够长：新开一个人
                    Some((_, s)) if s < self.p.new_speaker && secs >= self.p.min_new_secs => {
                        Some(self.create(&e, 0.0))
                    }
                    // 上一段的人只要差得不多就不换，避免来回跳
                    Some((k, s)) => match (prev, prev_sim) {
                        (Some(pi), Some(ps)) if ps + self.p.stickiness >= s => Some(pi),
                        _ => Some(k),
                    },
                };
                if let Some(k) = pick {
                    let s = dot(&e, &self.list[k].centroid);
                    // 只有足够长、足够像（或刚建）的段才更新平均声纹
                    if secs >= self.p.min_update_secs
                        && (s >= self.p.assign || self.list[k].secs == 0.0)
                    {
                        self.list[k].add(&e, secs);
                        self.try_merge(k);
                    }
                }
                pick.map(|k| self.root(k))
            }
        };
        if let Some(k) = chosen {
            self.list[k].secs += secs;
        }
        self.prev = chosen.or(self.prev);
        chosen
    }

    /// 和其他人的平均声纹太像就合并（小的并进大的）
    fn try_merge(&mut self, k: usize) {
        let others: Vec<usize> = self.active().filter(|&i| i != k).collect();
        for o in others {
            let (k, o) = (self.root(k), self.root(o));
            if k == o {
                continue;
            }
            if dot(&self.list[k].centroid, &self.list[o].centroid) >= self.p.merge {
                let (big, small) = if self.list[k].secs >= self.list[o].secs {
                    (k, o)
                } else {
                    (o, k)
                };
                self.merge(small, big);
            }
        }
    }

    fn merge(&mut self, small: usize, big: usize) {
        let (sum, secs, names) = {
            let s = &self.list[small];
            (s.sum.clone(), s.secs, s.names_used.clone())
        };
        let b = &mut self.list[big];
        for (x, y) in b.sum.iter_mut().zip(&sum) {
            *x += y;
        }
        b.centroid = normalize(&b.sum);
        b.secs += secs;
        b.names_used.extend(names);
        self.list[small].merged_into = Some(big);
    }

    pub fn name(&self, i: usize) -> &str {
        &self.list[self.root(i)].name
    }

    /// 声纹库认人：说够一定时长后拿平均声纹去比
    pub fn identify_with(&mut self, i: usize, min_secs: f64, f: impl Fn(&[f32]) -> Option<String>) {
        let i = self.root(i);
        let c = &self.list[i];
        if c.known || c.secs < min_secs {
            return;
        }
        if let Some(name) = f(&c.centroid) {
            // 库里同一个人已经有一类了：并过去
            let same = self
                .active()
                .find(|&j| j != i && self.list[j].known && self.list[j].name == name);
            if let Some(j) = same {
                let (big, small) = if self.list[i].secs >= self.list[j].secs {
                    (i, j)
                } else {
                    (j, i)
                };
                self.merge(small, big);
                let b = &mut self.list[big];
                b.name = name.clone();
                b.known = true;
                b.names_used.push(name);
                return;
            }
            let c = &mut self.list[i];
            c.name = name.clone();
            c.known = true;
            c.names_used.push(name);
        }
    }

    /// 结束：零碎的人并进最像的大类，按出现顺序重新编号。
    /// 返回每个最终说话人的汇总，和"用过的名字 → 最终名字"的改名表（只含有变化的）。
    pub fn finish(&mut self) -> (Vec<SpeakerSummary>, HashMap<String, String>) {
        let roots: Vec<usize> = self.active().collect();
        let big: Vec<usize> = roots
            .iter()
            .copied()
            .filter(|&i| self.list[i].secs >= self.p.min_final_secs || self.list[i].known)
            .collect();
        if !big.is_empty() {
            for &s in roots.iter().filter(|s| !big.contains(s)) {
                let target = *big
                    .iter()
                    .max_by(|&&a, &&b| {
                        let c = &self.list[s].centroid;
                        dot(c, &self.list[a].centroid).total_cmp(&dot(c, &self.list[b].centroid))
                    })
                    .unwrap();
                self.merge(s, target);
            }
        }
        // 按创建顺序（≈首次出现）重新编号占位名
        let finals: Vec<usize> = self.active().collect();
        let mut summary = Vec::new();
        let mut renames = HashMap::new();
        let mut n = 0;
        for &i in &finals {
            let c = &self.list[i];
            let label = if c.known {
                c.name.clone()
            } else {
                n += 1;
                placeholder(n - 1)
            };
            for old in &c.names_used {
                if *old != label {
                    renames.insert(old.clone(), label.clone());
                }
            }
            summary.push(SpeakerSummary {
                label,
                secs: c.secs,
                known: c.known,
                profile: c.centroid.clone(),
            });
        }
        (summary, renames)
    }
}

/// 实时说话人标注：声纹模型 + 在线聚类 + 声纹库
pub struct OnlineSpeakers {
    extractor: SpeakerEmbeddingExtractor,
    book: VoiceBook,
    pub clusters: Clusters,
}

/// 说够这么久才去声纹库比（平均声纹稳定下来）
const IDENTIFY_AFTER_SECS: f64 = 20.0;

impl OnlineSpeakers {
    pub fn new(models: &Path, voices: &Path, threads: i32) -> Result<Self> {
        let emb = models.join(EMBEDDING_MODEL);
        if !emb.exists() {
            bail!("缺少声纹模型 {}，先运行 install.sh", emb.display());
        }
        let extractor = SpeakerEmbeddingExtractor::create(&SpeakerEmbeddingExtractorConfig {
            model: Some(emb.to_string_lossy().into_owned()),
            num_threads: threads,
            ..Default::default()
        })
        .context("加载声纹模型失败")?;
        let book = VoiceBook::open(voices.to_path_buf(), extractor.dim() as usize);
        Ok(Self {
            extractor,
            book,
            clusters: Clusters::new(Params::default()),
        })
    }

    pub fn embed(&self, samples: &[f32]) -> Option<Vec<f32>> {
        // 太短算不出可靠的声纹
        if samples.len() < SR / 2 {
            return None;
        }
        let stream = self.extractor.create_stream()?;
        stream.accept_waveform(SR as i32, samples);
        stream.input_finished();
        if !self.extractor.is_ready(&stream) {
            return None;
        }
        self.extractor.compute(&stream)
    }

    /// 一个 VAD 段（16kHz 单声道）→ 说话人显示名
    pub fn assign(&mut self, samples: &[f32]) -> Option<String> {
        let secs = samples.len() as f64 / SR as f64;
        let emb = if secs >= self.clusters.p.short_secs {
            self.embed(samples)
        } else {
            None
        };
        let k = self.clusters.assign(emb.as_deref(), secs)?;
        let book = &self.book;
        self.clusters
            .identify_with(k, IDENTIFY_AFTER_SECS, |c| book.identify(c));
        Some(self.clusters.name(k).to_owned())
    }

    pub fn finish(&mut self) -> (Vec<SpeakerSummary>, HashMap<String, String>) {
        self.clusters.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 两个"人"：在各自方向附近抖动
    fn voice(base: usize, jitter: f32, seed: u32) -> Vec<f32> {
        let mut v = vec![0f32; 8];
        v[base] = 1.0;
        // 简单的确定性"噪声"
        for (i, x) in v.iter_mut().enumerate() {
            let r = ((seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 40503)) % 1000) as f32
                / 1000.0;
            *x += (r - 0.5) * jitter;
        }
        v
    }

    fn params() -> Params {
        Params {
            assign: 0.55,
            new_speaker: 0.35,
            merge: 0.7,
            min_new_secs: 2.0,
            short_secs: 1.5,
            min_update_secs: 1.5,
            min_final_secs: 8.0,
            stickiness: 0.08,
        }
    }

    #[test]
    fn separates_two_voices_and_keeps_labels_stable() {
        let mut c = Clusters::new(params());
        let a = c.assign(Some(&voice(0, 0.3, 1)), 4.0).unwrap();
        for s in 2..10 {
            assert_eq!(c.assign(Some(&voice(0, 0.3, s)), 4.0), Some(a));
        }
        let b = c.assign(Some(&voice(3, 0.3, 50)), 3.0).unwrap();
        assert_ne!(a, b);
        assert_eq!(c.assign(Some(&voice(3, 0.3, 51)), 3.0), Some(b));
        assert_eq!(c.assign(Some(&voice(0, 0.3, 52)), 4.0), Some(a));
        assert_eq!(c.name(a), "说话人A");
        assert_eq!(c.name(b), "说话人B");
    }

    #[test]
    fn short_segments_inherit_previous_speaker() {
        let mut c = Clusters::new(params());
        let a = c.assign(Some(&voice(0, 0.2, 1)), 4.0).unwrap();
        // 很短的段即使声纹像别人也沿用上一个
        assert_eq!(c.assign(Some(&voice(5, 0.2, 2)), 0.8), Some(a));
        assert_eq!(c.assign(None, 0.4), Some(a));
        // 第一段就很短：不知道是谁
        let mut c = Clusters::new(params());
        assert_eq!(c.assign(Some(&voice(0, 0.2, 1)), 1.0), None);
    }

    #[test]
    fn does_not_open_new_speaker_on_short_odd_segment() {
        let mut c = Clusters::new(params());
        let a = c.assign(Some(&voice(0, 0.2, 1)), 4.0).unwrap();
        // 1.8s：够算声纹，但不够新开一个人
        assert_eq!(c.assign(Some(&voice(6, 0.2, 2)), 1.8), Some(a));
    }

    #[test]
    fn merges_split_clusters_and_renames_at_finish() {
        let mut c = Clusters::new(params());
        let a = c.assign(Some(&voice(0, 0.1, 1)), 10.0).unwrap();
        // 一个奇怪的段被开成了新的人，但它其实和 A、和后来的 C 都不太一样
        let b = c.assign(Some(&voice(4, 0.1, 2)), 3.0).unwrap();
        assert_ne!(a, b);
        let cc = c.assign(Some(&voice(2, 0.1, 3)), 20.0).unwrap();
        assert_ne!(cc, b);
        let (summary, renames) = c.finish();
        // B 只说了 3s，并进最像的大类；C 重新编号成 B
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].label, "说话人A");
        assert_eq!(summary[1].label, "说话人B");
        assert_eq!(renames.get("说话人C").map(String::as_str), Some("说话人B"));
        // 旧 B 的段改到某个最终说话人名下
        let old_b = renames.get("说话人B").map_or("说话人B", String::as_str);
        assert!(summary.iter().any(|s| s.label == old_b));
    }

    #[test]
    fn identifies_from_voicebook_after_enough_speech() {
        let mut c = Clusters::new(params());
        let a = c.assign(Some(&voice(0, 0.1, 1)), 5.0).unwrap();
        c.identify_with(a, 20.0, |_| Some("张老师".into()));
        assert_eq!(c.name(a), "说话人A");
        for s in 0..4 {
            c.assign(Some(&voice(0, 0.1, 10 + s)), 5.0);
        }
        c.identify_with(a, 20.0, |_| Some("张老师".into()));
        assert_eq!(c.name(a), "张老师");
        let (summary, renames) = c.finish();
        assert_eq!(summary[0].label, "张老师");
        assert_eq!(renames.get("说话人A").map(String::as_str), Some("张老师"));
    }
}
