//! ffmpeg 解码 → Silero VAD 切段 → 逐段识别（Parakeet / SenseVoice），输出带时间戳的单词。
//!
//! 实时模式下还会在一段话说到一半时，每隔一小段重新识别当前这段，回调"正在说"的文字。

use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result, bail};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
    OfflineTransducerModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};

use crate::preproc::{Preproc, PreprocConfig};
use crate::segment::{Word, estimate_ends};

pub const SR: usize = 16000;

/// 给一段语音（16kHz 单声道）标说话人
pub type SpeakerOf<'a> = dyn FnMut(&[f32]) -> Option<String> + 'a;
const VAD_WINDOW: usize = 512;
/// 实时模式：正在说的这段每隔多久重新识别一次
const PARTIAL_EVERY: usize = SR * 6 / 10;
/// 实时模式：VAD 触发前保留的一小段音频，避免吃掉开头的字
const PRE_ROLL: usize = SR * 3 / 10;
/// 预处理时保留多少原始音频（要覆盖 VAD 缓冲里最长的一段）
const KEEP_ORIG: usize = SR * 90;

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Engine {
    /// Qwen3-ASR 1.7B：最准，在 Apple GPU 上跑（MLX）；「正在说」的预览仍用 Parakeet
    Qwen,
    /// NVIDIA Parakeet TDT 0.6B v2：CPU 上快，自带标点
    Parakeet,
    /// 阿里 SenseVoice-Small：多语种，英文标点弱（会用 LLM 补）
    Sensevoice,
}

pub const PARAKEET_DIR: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";
pub const SENSEVOICE_DIR: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";

/// 启动 ffmpeg，把任意音视频（或 URL / 设备）解码成 16kHz 单声道 f32 流。
fn spawn_ffmpeg(input: &Path) -> Result<Child> {
    let ffmpeg = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
        .into_iter()
        .find(|p| Path::new(p).exists())
        .unwrap_or("ffmpeg");
    Command::new(ffmpeg)
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("无法运行 ffmpeg（brew install ffmpeg）")
}

/// 整个文件解码进内存（说话人分离需要完整音频）
pub fn decode_all(input: &Path) -> Result<Vec<f32>> {
    let mut c = spawn_ffmpeg(input)?;
    let mut bytes = Vec::new();
    c.stdout.take().expect("piped").read_to_end(&mut bytes)?;
    if !c.wait()?.success() {
        let mut err = String::new();
        if let Some(mut e) = c.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        bail!("ffmpeg 解码失败: {}", err.trim());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// 音频来源
pub enum Input {
    /// 任意音视频文件，用 ffmpeg 解码
    File(PathBuf),
    /// 现成的 16kHz 单声道 f32le 字节流（stdin、系统声音采集）
    Raw(Box<dyn Read>),
}

pub struct Asr {
    vad: VadModelConfig,
    /// CPU 识别（Parakeet / SenseVoice）：实时预览，以及 Qwen 不可用时的正式结果
    rec: OfflineRecognizer,
    /// GPU 识别：正式字幕
    qwen: Option<std::sync::Mutex<crate::qwen::Qwen>>,
    /// 给 Qwen 的术语提示（课程主题）
    context: String,
    pre: PreprocConfig,
}

impl Asr {
    pub fn new(models: &Path, engine: Engine, threads: i32, max_segment: f32) -> Result<Self> {
        let s = |p: &Path| Some(p.to_string_lossy().into_owned());
        let vad_model = models.join("silero_vad.onnx");
        let mut cfg = OfflineRecognizerConfig::default();
        let needed = match engine {
            Engine::Parakeet | Engine::Qwen => {
                let d = models.join(PARAKEET_DIR);
                cfg.model_config.transducer = OfflineTransducerModelConfig {
                    encoder: s(&d.join("encoder.int8.onnx")),
                    decoder: s(&d.join("decoder.int8.onnx")),
                    joiner: s(&d.join("joiner.int8.onnx")),
                };
                cfg.model_config.model_type = Some("nemo_transducer".into());
                cfg.model_config.tokens = s(&d.join("tokens.txt"));
                vec![d.join("encoder.int8.onnx"), d.join("tokens.txt")]
            }
            Engine::Sensevoice => {
                let d = models.join(SENSEVOICE_DIR);
                cfg.model_config.sense_voice = OfflineSenseVoiceModelConfig {
                    model: s(&d.join("model.onnx")),
                    language: Some("en".into()),
                    use_itn: true,
                };
                cfg.model_config.tokens = s(&d.join("tokens.txt"));
                vec![d.join("model.onnx"), d.join("tokens.txt")]
            }
        };
        for p in needed.iter().chain([&vad_model]) {
            if !p.exists() {
                bail!("缺少模型文件 {}，先运行 install.sh 下载模型", p.display());
            }
        }
        cfg.model_config.num_threads = threads;
        let rec = OfflineRecognizer::create(&cfg).context("加载识别模型失败")?;

        let vad = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: s(&vad_model),
                threshold: 0.5,
                // 超过 0.5s 的停顿才算断开；段边界是字幕的硬边界
                min_silence_duration: 0.5,
                min_speech_duration: 0.25,
                window_size: VAD_WINDOW as i32,
                // 一次处理 30s 以内效果最好；实时模式用更短的段降低延迟
                max_speech_duration: max_segment,
            },
            sample_rate: SR as i32,
            num_threads: 1,
            ..Default::default()
        };
        let qwen = match engine {
            Engine::Qwen => {
                let data = models.parent().unwrap_or(models);
                Some(std::sync::Mutex::new(crate::qwen::Qwen::start(data)?))
            }
            _ => None,
        };
        Ok(Self {
            vad,
            rec,
            qwen,
            context: String::new(),
            pre: PreprocConfig::default(),
        })
    }

    /// 术语提示：交给 Qwen，让课程里的专有名词认得更准
    pub fn with_context(mut self, context: Option<&str>) -> Self {
        self.context = context.unwrap_or_default().trim().to_owned();
        self
    }

    /// 正式识别一段：优先 GPU 上的 Qwen，失败时退回 CPU 识别
    fn decode_final(&self, samples: &[f32], start: f64, end: f64) -> Vec<Word> {
        if let Some(q) = &self.qwen {
            match q.lock().unwrap().transcribe(samples, start, &self.context) {
                Ok(words) => return words,
                Err(e) => eprintln!("  ! {e:#}，这段改用 Parakeet"),
            }
        }
        self.decode(samples, start, end)
    }

    /// VAD 前的预处理（AGC / 降噪）。只影响切段，识别仍用原始音频。
    pub fn with_preproc(mut self, pre: PreprocConfig) -> Self {
        self.pre = pre;
        self
    }

    /// 边解码边识别：每识别完一个 VAD 段就回调 `on_segment`（单词为全局时间、这段的说话人、
    /// 已解码的音频时长）。给了 `on_partial` 时，一段话说到一半也会回调当前识别出的文字；
    /// 给了 `speaker_of` 时用它给每个 VAD 段标说话人（实时声纹）。
    /// 返回音频总时长（秒）。
    pub fn stream(
        &self,
        input: Input,
        mut on_segment: impl FnMut(Vec<Word>, Option<String>, f64),
        mut on_partial: Option<&mut dyn FnMut(&str)>,
        mut speaker_of: Option<&mut SpeakerOf<'_>>,
    ) -> Result<f64> {
        let vad = VoiceActivityDetector::create(&self.vad, 60.0).context("加载 VAD 失败")?;
        let mut child = None;
        let mut stdout: Box<dyn Read> = match input {
            Input::File(p) => {
                let mut c = spawn_ffmpeg(&p)?;
                let out = c.stdout.take().expect("piped");
                child = Some(c);
                Box::new(out)
            }
            Input::Raw(r) => r,
        };
        let mut buf = vec![0u8; VAD_WINDOW * 4 * 64];
        let mut bytes: Vec<u8> = Vec::new();
        let mut samples: Vec<f32> = Vec::new();
        let mut total = 0usize;
        // 实时模式：最近一小段音频 + 正在说的这段
        let mut history: VecDeque<f32> = VecDeque::with_capacity(PRE_ROLL + VAD_WINDOW);
        let mut active: Option<Vec<f32>> = None;
        let mut since_partial = 0usize;

        let mut pre = if self.pre.enabled() {
            Some(Preproc::new(&self.pre)?)
        } else {
            None
        };
        // 有预处理时 VAD 吃的是处理过的音频，识别要从这里按位置取原始音频（orig[0] 的绝对位置是 base）
        let mut orig: VecDeque<f32> = VecDeque::new();
        let mut base = 0usize;

        let mut drain = |vad: &VoiceActivityDetector,
                         total: usize,
                         orig: &VecDeque<f32>,
                         base: usize|
         -> bool {
            let mut popped = false;
            while let Some(seg) = vad.front() {
                let start = seg.start() as f64 / SR as f64;
                let end = start + seg.n() as f64 / SR as f64;
                let samples = if orig.is_empty() {
                    seg.samples().to_vec()
                } else {
                    let a = (seg.start().max(0) as usize).saturating_sub(base);
                    let b = (a + seg.n().max(0) as usize).min(orig.len());
                    orig.range(a.min(b)..b).copied().collect()
                };
                let words = self.decode_final(&samples, start, end);
                vad.pop();
                popped = true;
                if !words.is_empty() {
                    // 声纹也用原始音频
                    let speaker = speaker_of.as_mut().and_then(|f| f(&samples));
                    on_segment(words, speaker, total as f64 / SR as f64);
                }
            }
            popped
        };
        loop {
            let n = stdout.read(&mut buf).context("读取音频流失败")?;
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..n]);
            let whole = bytes.len() / 4 * 4;
            samples.extend(
                bytes[..whole]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b)),
            );
            bytes.drain(..whole);
            // VAD 按固定窗口喂
            let full = samples.len() / VAD_WINDOW * VAD_WINDOW;
            for w in samples[..full].as_chunks::<VAD_WINDOW>().0 {
                match pre.as_mut() {
                    Some(p) => {
                        orig.extend(w.iter().copied());
                        let extra = orig.len().saturating_sub(KEEP_ORIG);
                        orig.drain(..extra);
                        base += extra;
                        vad.accept_waveform(&p.process(w));
                    }
                    None => vad.accept_waveform(w),
                }
                total += VAD_WINDOW;
                if drain(&vad, total, &orig, base) {
                    // 这段已经出了正式结果；如果还在说（超长被切开），从头开始新的一段
                    active = vad.detected().then(Vec::new);
                    since_partial = 0;
                }
                let Some(cb) = on_partial.as_mut() else {
                    continue;
                };
                match active.as_mut() {
                    Some(a) => a.extend_from_slice(w),
                    None if vad.detected() => {
                        let mut a: Vec<f32> = history.iter().copied().collect();
                        a.extend_from_slice(w);
                        active = Some(a);
                        since_partial = 0;
                    }
                    None => {}
                }
                history.extend(w.iter().copied());
                let extra = history.len().saturating_sub(PRE_ROLL);
                history.drain(..extra);
                since_partial += VAD_WINDOW;
                if let Some(a) = &active
                    && since_partial >= PARTIAL_EVERY
                    && a.len() >= SR / 2
                {
                    since_partial = 0;
                    let words = self.decode(a, 0.0, a.len() as f64 / SR as f64);
                    let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
                    cb(&text.join(" "));
                }
            }
            samples.drain(..full);
        }
        match pre.as_mut() {
            Some(p) => {
                orig.extend(samples.iter().copied());
                let mut y = p.process(&samples);
                y.extend(p.flush());
                if !y.is_empty() {
                    vad.accept_waveform(&y);
                }
            }
            None if !samples.is_empty() => vad.accept_waveform(&samples),
            None => {}
        }
        total += samples.len();
        vad.flush();
        drain(&vad, total, &orig, base);

        if let Some(mut child) = child {
            let status = child.wait()?;
            if !status.success() {
                let mut err = String::new();
                if let Some(mut e) = child.stderr.take() {
                    let _ = e.read_to_string(&mut err);
                }
                bail!("ffmpeg 解码失败: {}", err.trim());
            }
        }
        Ok(total as f64 / SR as f64)
    }

    fn decode(&self, samples: &[f32], start: f64, end: f64) -> Vec<Word> {
        let stream = self.rec.create_stream();
        stream.accept_waveform(SR as i32, samples);
        self.rec.decode(&stream);
        let Some(r) = stream.get_result() else {
            return Vec::new();
        };
        let ts = r.timestamps.unwrap_or_default();
        let durs = r.durations.unwrap_or_default();
        let mut words = tokens_to_words(&r.tokens, &ts, &durs, start);
        // SenseVoice 只给 token 起点，需要估计终点；Parakeet 有每个 token 的时长
        if !words.is_empty() && durs.len() != ts.len() {
            estimate_ends(&mut words, end);
        }
        words
    }
}

/// BPE token（前导空格表示新词）→ 单词。`durs` 为空时终点暂取最后一个 token 的起点。
pub fn tokens_to_words(tokens: &[String], ts: &[f32], durs: &[f32], offset: f64) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut new_word = true;
    for (i, (tok, &t)) in tokens.iter().zip(ts).enumerate() {
        if tok.starts_with("<|") && tok.ends_with("|>") {
            continue;
        }
        let piece = tok.trim_start();
        if piece.len() != tok.len() {
            new_word = true;
        }
        if piece.is_empty() {
            continue;
        }
        let t = offset + t as f64;
        let end = t + durs.get(i).copied().unwrap_or(0.0) as f64;
        match words.last_mut() {
            Some(w) if !new_word => {
                w.text.push_str(piece);
                w.end = end;
            }
            _ => words.push(Word::new(piece, t, end)),
        }
        new_word = false;
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn merges_bpe_pieces() {
        let toks = toks(&[
            "We", "l", "com", "e", " back", " we", "'", "re", " ", "ok", ",",
        ]);
        let ts: Vec<f32> = (0..toks.len()).map(|i| i as f32 * 0.1).collect();
        let ws = tokens_to_words(&toks, &ts, &[], 10.0);
        let texts: Vec<_> = ws.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["Welcome", "back", "we're", "ok,"]);
        assert!((ws[1].start - 10.4).abs() < 1e-6);
    }

    #[test]
    fn uses_token_durations_for_word_end() {
        let toks = toks(&[" So", " p", "le", "ase", "."]);
        let ts = [0.4, 0.64, 0.72, 0.8, 0.88];
        let durs = [0.08, 0.08, 0.08, 0.08, 0.32];
        let ws = tokens_to_words(&toks, &ts, &durs, 0.0);
        assert_eq!(ws.len(), 2);
        assert_eq!(ws[1].text, "please.");
        assert!((ws[1].end - 1.2).abs() < 1e-6);
    }
}
