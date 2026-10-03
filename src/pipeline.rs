//! 流式流水线：识别线程 → 按块并行补标点/断句/翻译 → 按顺序增量输出。
//!
//! 识别线程每攒够 `block_secs` 秒语音就打成一个块发给 worker；worker 之间并行，
//! 输出线程按块序号重排后写 SRT（每条写完即 flush），并可在 stdout 输出 JSONL，
//! 方便下游（如笔记生成）边识别边消费。

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;

use crate::asr::{Asr, Input};
use crate::llm::DeepSeek;
use crate::segment::{Cue, Limits, Word, apply_punctuation, build_cues, retime};
use crate::srt::{self, Track};

#[derive(Clone)]
pub struct Options {
    pub lim: Limits,
    pub punct: bool,
    pub translate: bool,
    pub hint: Option<String>,
    pub workers: usize,
    pub block_secs: f64,
    pub jsonl: bool,
    /// 实时输入：每个语音段立即处理，空闲时尽快输出
    pub live: bool,
    /// 说话人标注：每条字幕写出前调用（可能会等后台的声纹识别完成）
    pub labeler: Option<Labeler>,
}

pub type Labeler = std::sync::Arc<dyn Fn(&mut Cue) + Send + Sync>;

struct Block {
    idx: usize,
    segments: Vec<Vec<Word>>,
    /// 每个 VAD 段的说话人（实时声纹）
    speakers: Vec<Option<String>>,
    /// 上一块结尾几段原文，给翻译当上文
    before: Vec<String>,
}

pub struct Summary {
    pub cues: usize,
    pub audio_secs: f64,
}

/// 每输出一条字幕回调一次（写文字稿、交给笔记、终端显示等）
pub type OnCue<'a> = &'a mut (dyn FnMut(&Cue) -> Result<()> + Send);

/// 输出回调：每条正式字幕 / （实时模式）正在说的文字
pub struct Hooks<'a> {
    pub on_cue: OnCue<'a>,
    pub on_partial: Option<&'a mut dyn FnMut(&str)>,
    /// 给每个 VAD 段的音频标说话人（实时声纹）
    pub speaker_of: Option<&'a mut crate::asr::SpeakerOf<'a>>,
}

pub fn run(
    input: Input,
    out_dir: &Path,
    stem: &str,
    asr: &Asr,
    llm: Option<&DeepSeek>,
    opt: &Options,
    hooks: Hooks,
) -> Result<Summary> {
    let Hooks {
        on_cue,
        on_partial,
        speaker_of,
    } = hooks;
    let translate = opt.translate && llm.is_some();
    let sink = Sink::create(out_dir, stem, translate, opt.jsonl)?;
    let (block_tx, block_rx) = mpsc::sync_channel::<Block>(opt.workers * 2);
    let block_rx = Mutex::new(block_rx);
    let (done_tx, done_rx) = mpsc::channel::<(usize, Vec<Cue>)>();

    thread::scope(|sc| {
        for _ in 0..opt.workers.max(1) {
            let (rx, tx) = (&block_rx, done_tx.clone());
            sc.spawn(move || {
                loop {
                    let Ok(block) = rx.lock().unwrap().recv() else {
                        break;
                    };
                    let idx = block.idx;
                    if tx
                        .send((idx, process_block(block, llm, opt, translate)))
                        .is_err()
                    {
                        break;
                    }
                }
            });
        }
        drop(done_tx);

        let live = opt.live;
        let mut em = Emitter {
            on_cue,
            labeler: opt.labeler.as_ref(),
            sink,
            lim: &opt.lim,
            held: None,
            not_before: 0.0,
        };
        let emitter = sc.spawn(move || -> Result<Sink> {
            let mut ready: BTreeMap<usize, Vec<Cue>> = BTreeMap::new();
            let mut next = 0;
            loop {
                match done_rx.recv_timeout(IDLE_FLUSH) {
                    Ok((i, cues)) => {
                        ready.insert(i, cues);
                        while let Some(cues) = ready.remove(&next) {
                            next += 1;
                            em.push(cues)?;
                        }
                    }
                    // 实时模式：一段时间没有新内容（对方停下来了），先把压着的最后一条放出去
                    Err(RecvTimeoutError::Timeout) if live && ready.is_empty() => {
                        em.flush_held()?
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            for (_, cues) in std::mem::take(&mut ready) {
                em.push(cues)?;
            }
            em.flush_held()?;
            Ok(em.sink)
        });

        // 识别在当前线程进行
        let mut pending: Vec<(Vec<Word>, Option<String>)> = Vec::new();
        let mut idx = 0;
        let mut before: Vec<String> = Vec::new();
        let send = |pending: &mut Vec<(Vec<Word>, Option<String>)>,
                    idx: &mut usize,
                    before: &mut Vec<String>| {
            if pending.is_empty() {
                return;
            }
            let (segments, speakers): (Vec<_>, Vec<_>) =
                std::mem::take(pending).into_iter().unzip();
            let tail: Vec<String> = segments
                .iter()
                .rev()
                .take(3)
                .rev()
                .map(|ws| join(ws))
                .collect();
            let block = Block {
                idx: *idx,
                segments,
                speakers,
                before: std::mem::replace(before, tail),
            };
            *idx += 1;
            let _ = block_tx.send(block);
        };
        let jsonl = opt.jsonl;
        let asr_result = asr.stream(
            input,
            |words, speaker, pos| {
                pending.push((words, speaker));
                // 第一块小一点，让下游尽快拿到内容
                let target = if idx == 0 {
                    opt.block_secs.min(30.0)
                } else {
                    opt.block_secs
                };
                let speech: f64 = pending
                    .iter()
                    .map(|(ws, _)| ws[ws.len() - 1].end - ws[0].start)
                    .sum();
                if speech >= target {
                    send(&mut pending, &mut idx, &mut before);
                }
                if jsonl {
                    println!("{}", json!({"type": "progress", "audio_secs": pos}));
                } else if !live {
                    eprint!("\r  已识别 {}", clock(pos));
                    let _ = std::io::stderr().flush();
                }
            },
            on_partial,
            speaker_of,
        );
        send(&mut pending, &mut idx, &mut before);
        drop(block_tx);

        let sink = emitter.join().expect("emitter panicked")?;
        let audio_secs = asr_result?;
        if !jsonl && !live {
            eprintln!();
        }
        Ok(Summary {
            cues: sink.count,
            audio_secs,
        })
    })
}

fn process_block(block: Block, llm: Option<&DeepSeek>, opt: &Options, translate: bool) -> Vec<Cue> {
    let mut segments = block.segments;
    if let Some(llm) = llm.filter(|_| opt.punct) {
        let raw: Vec<String> = segments.iter().map(|ws| join(ws)).collect();
        for (ws, text) in segments.iter_mut().zip(llm.punctuate(&raw, 1)) {
            if let Some(text) = text {
                *ws = apply_punctuation(ws, &text);
            }
        }
    }
    let mut cues = build_cues(&segments, &opt.lim);
    for c in &mut cues {
        if let Some(s) = block.speakers.get(c.seg).cloned().flatten() {
            c.speaker = Some(s);
        }
    }
    if let Some(llm) = llm.filter(|_| translate && !cues.is_empty()) {
        let lines: Vec<String> = cues.iter().map(Cue::text).collect();
        let zh = llm.translate(&lines, &block.before, opt.hint.as_deref(), 2);
        for (c, zh) in cues.iter_mut().zip(zh) {
            c.zh = zh;
        }
        retime(&mut cues, &opt.lim, 0.0);
    }
    cues
}

/// 实时模式下多久没有新字幕就把压着的最后一条输出
const IDLE_FLUSH: Duration = Duration::from_millis(1200);

/// 按顺序输出。每块最后一条要等下一块到了才能确定结束时间（不能和下一条重叠），先压着。
struct Emitter<'a> {
    on_cue: OnCue<'a>,
    labeler: Option<&'a Labeler>,
    sink: Sink,
    lim: &'a Limits,
    held: Option<Cue>,
    not_before: f64,
}

impl Emitter<'_> {
    fn push(&mut self, cues: Vec<Cue>) -> Result<()> {
        let mut all: Vec<Cue> = self.held.take().into_iter().chain(cues).collect();
        if all.is_empty() {
            return Ok(());
        }
        retime(&mut all, self.lim, self.not_before);
        self.held = all.pop();
        for c in &mut all {
            self.write(c)?;
        }
        Ok(())
    }

    fn flush_held(&mut self) -> Result<()> {
        if let Some(mut last) = self.held.take() {
            retime(std::slice::from_mut(&mut last), self.lim, self.not_before);
            self.write(&mut last)?;
        }
        Ok(())
    }

    fn write(&mut self, c: &mut Cue) -> Result<()> {
        if let Some(l) = self.labeler {
            l(c);
        }
        self.sink.write(c)?;
        self.not_before = c.end + self.lim.gap;
        (self.on_cue)(c)
    }
}

/// 增量写出：每条字幕写完就 flush，处理中途也能打开看。
struct Sink {
    files: Vec<(File, Track)>,
    jsonl: bool,
    count: usize,
    last_speaker: Option<String>,
}

impl Sink {
    fn create(dir: &Path, stem: &str, translate: bool, jsonl: bool) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut tracks = vec![("en", Track::En)];
        if translate {
            tracks.extend([("zh", Track::Zh), ("zh-en", Track::Bilingual)]);
        }
        let mut files = Vec::new();
        for (suffix, track) in tracks {
            let path = dir.join(format!("{stem}.{suffix}.srt"));
            eprintln!("  → {}", path.display());
            files.push((File::create(path)?, track));
        }
        Ok(Self {
            files,
            jsonl,
            count: 0,
            last_speaker: None,
        })
    }

    fn write(&mut self, c: &Cue) -> Result<()> {
        self.count += 1;
        // 换人时标出说话人
        let label = (c.speaker.is_some() && c.speaker != self.last_speaker)
            .then(|| c.speaker.clone())
            .flatten();
        self.last_speaker = c.speaker.clone();
        for (f, track) in &mut self.files {
            f.write_all(srt::entry(self.count, c, *track, label.as_deref()).as_bytes())?;
            f.flush()?;
        }
        if self.jsonl {
            let line = json!({
                "type": "cue", "i": self.count, "start": c.start, "end": c.end,
                "en": c.text(), "zh": c.zh, "speaker": c.speaker,
            });
            let mut out = std::io::stdout().lock();
            writeln!(out, "{line}")?;
            out.flush()?;
        }
        Ok(())
    }
}

fn join(ws: &[Word]) -> String {
    ws.iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn clock(secs: f64) -> String {
    let s = secs as u64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}
