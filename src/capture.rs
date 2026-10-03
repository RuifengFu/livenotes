//! 录电脑内部声音（ScreenCaptureKit）和/或麦克风（CoreAudio，经 cpal），
//! 混成 16kHz 单声道 f32le 字节流读取。
//!
//! 麦克风不走 ScreenCaptureKit：实测它录到的麦克风声音只有直接录的 1/3 左右且更糊，
//! 远处讲台上的老师基本识别不出来。
//!
//! 输出按墙钟时间驱动（独立的采集线程）：没有声音时补静音，保证样本数 = 真实经过的时间，
//! 字幕时间轴才能和会议对得上。`stop` 置位后返回 EOF，下游把剩余内容处理完。

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufWriter, Read};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use screencapturekit::cm::CMTime;
use screencapturekit::prelude::*;

const SR: u32 = 16000;
/// 每次产出 50ms
const TICK: Duration = Duration::from_millis(50);
/// 采集缓冲超过这么多就丢最旧的（防止时钟漂移累积延迟）
const MAX_BACKLOG: usize = SR as usize / 2;

type Buf = Arc<Mutex<VecDeque<f32>>>;

struct Tap(Buf);

impl SCStreamOutputTrait for Tap {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, _t: SCStreamOutputType) {
        let Ok(list) = sample.audio_buffer_list() else {
            return;
        };
        // 单声道：只取第一个 buffer
        if let Some(b) = (&list).into_iter().next() {
            let mut q = self.0.lock().unwrap();
            q.extend(
                b.data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c)),
            );
            let extra = q.len().saturating_sub(MAX_BACKLOG);
            q.drain(..extra);
        }
    }
}

pub struct Capture {
    stream: Option<SCStream>,
    _mic_stream: Option<cpal::Stream>,
    /// 采集线程按墙钟产出的音频；读得慢只会积压，不会丢
    rx: Option<mpsc::Receiver<Vec<u8>>>,
    producer: Option<std::thread::JoinHandle<()>>,
    pending: Vec<u8>,
}

impl Capture {
    /// `sys`：录电脑内部声音；`mic`：录默认麦克风。
    pub fn start(sys: bool, mic: bool, wav: Option<&Path>, stop: Arc<AtomicBool>) -> Result<Self> {
        let (stream, sys_buf) = if sys {
            let (s, b) = start_system()?;
            (Some(s), Some(b))
        } else {
            (None, None)
        };
        let (mic_stream, mic_buf) = if mic {
            let (s, b) = start_mic()?;
            (Some(s), Some(b))
        } else {
            (None, None)
        };
        let wav = match wav {
            Some(p) => Some(hound::WavWriter::create(
                p,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: SR,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?),
            None => None,
        };
        // 采集和识别解耦：识别线程解码一段要几百毫秒，期间如果没人取数，
        // 采集缓冲会溢出丢音频（之前的录音里每隔 0.6s 就有一小段被替换成静音）。
        let (rx, producer) =
            spawn_producer(sys_buf.into_iter().chain(mic_buf).collect(), wav, stop);
        Ok(Self {
            stream,
            _mic_stream: mic_stream,
            rx: Some(rx),
            producer: Some(producer),
            pending: Vec::new(),
        })
    }

    fn finish(&mut self) {
        if let Some(s) = self.stream.take() {
            let _ = s.stop_capture();
        }
        self._mic_stream = None;
        // 先断开接收端，采集线程下次发送失败就会退出
        self.rx = None;
        if let Some(p) = self.producer.take() {
            let _ = p.join();
        }
    }
}

type Wav = hound::WavWriter<BufWriter<File>>;

fn spawn_producer(
    bufs: Vec<Buf>,
    wav: Option<Wav>,
    stop: Arc<AtomicBool>,
) -> (mpsc::Receiver<Vec<u8>>, std::thread::JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let mut p = Producer {
        bufs,
        start: Instant::now(),
        written: 0,
        wav,
    };
    let handle = std::thread::spawn(move || {
        while !stop.load(Ordering::SeqCst) {
            std::thread::sleep(TICK);
            if tx.send(p.produce()).is_err() {
                break;
            }
        }
        if let Some(w) = p.wav.take() {
            let _ = w.finalize();
        }
    });
    (rx, handle)
}

struct Producer {
    bufs: Vec<Buf>,
    start: Instant,
    written: u64,
    wav: Option<Wav>,
}

impl Producer {
    /// 按墙钟补齐到当前时刻应有的样本数（设备没给够的部分补静音）。
    fn produce(&mut self) -> Vec<u8> {
        let due = (self.start.elapsed().as_secs_f64() * SR as f64) as u64;
        let n = (due - self.written) as usize;
        let mut frame = vec![0.0f32; n];
        for buf in &self.bufs {
            let mut q = buf.lock().unwrap();
            let k = n.min(q.len());
            for (o, s) in frame.iter_mut().zip(q.drain(..k)) {
                *o += s;
            }
        }
        let mut bytes = Vec::with_capacity(n * 4);
        for s in &mut frame {
            *s = s.clamp(-1.0, 1.0);
            bytes.extend_from_slice(&s.to_le_bytes());
            if let Some(w) = self.wav.as_mut()
                && w.write_sample((*s * i16::MAX as f32) as i16).is_err()
            {
                self.wav = None;
            }
        }
        self.written += n as u64;
        bytes
    }
}

fn start_system() -> Result<(SCStream, Buf)> {
    let content = SCShareableContent::get().map_err(|e| {
        anyhow!(
            "无法录制电脑声音（{e:?}）。请在 系统设置 → 隐私与安全性 → 屏幕与系统录音 \
             里允许你用的终端，然后重启终端"
        )
    })?;
    let display = content
        .displays()
        .into_iter()
        .next()
        .context("没有找到显示器")?;
    let filter = SCContentFilter::create()
        .with_display(&display)
        .with_excluding_windows(&[])
        .build()?;
    let config = SCStreamConfiguration::new()
        // 只要音频：视频开到最小
        .with_width(2)
        .with_height(2)
        .with_minimum_frame_interval(&CMTime::new(1, 1))
        .with_captures_audio(true)
        .with_sample_rate(SR as i32)
        .with_channel_count(1)
        .with_excludes_current_process_audio(true);
    let buf: Buf = Arc::default();
    let mut stream = SCStream::new(&filter, &config)?;
    stream.add_output_handler(Tap(buf.clone()), SCStreamOutputType::Audio)?;
    stream
        .start_capture()
        .context("启动录制失败（检查 屏幕与系统录音 权限）")?;
    Ok((stream, buf))
}

fn start_mic() -> Result<(cpal::Stream, Buf)> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let dev = cpal::default_host()
        .default_input_device()
        .context("没有找到麦克风")?;
    let cfg = dev.default_input_config().context("读取麦克风格式失败")?;
    if cfg.sample_format() != cpal::SampleFormat::F32 {
        bail!("暂不支持的麦克风采样格式 {:?}", cfg.sample_format());
    }
    let channels = cfg.channels() as usize;
    let mut rs = Resampler::new(cfg.sample_rate());
    let buf: Buf = Arc::default();
    let out = buf.clone();
    let stream = dev
        .build_input_stream(
            cfg.config(),
            move |data: &[f32], _| {
                let mono: Vec<f32> = data
                    .chunks(channels)
                    .map(|f| f.iter().sum::<f32>() / channels as f32)
                    .collect();
                let mut q = out.lock().unwrap();
                q.extend(rs.push(&mono));
                let extra = q.len().saturating_sub(MAX_BACKLOG);
                q.drain(..extra);
            },
            |e| eprintln!("  ! 麦克风错误: {e}"),
            None,
        )
        .context("打开麦克风失败（检查 麦克风 权限）")?;
    stream.play().context("启动麦克风失败")?;
    Ok((stream, buf))
}

/// 降采样到 16kHz：整数倍时先做块平均（简单低通）再抽取，否则线性插值。
struct Resampler {
    rate: u32,
    acc: Vec<f32>,
    pos: f64,
    prev: f32,
}

impl Resampler {
    fn new(rate: u32) -> Self {
        Self {
            rate,
            acc: Vec::new(),
            pos: 0.0,
            prev: 0.0,
        }
    }

    fn push(&mut self, x: &[f32]) -> Vec<f32> {
        if self.rate == SR {
            return x.to_vec();
        }
        if self.rate.is_multiple_of(SR) {
            let k = (self.rate / SR) as usize;
            self.acc.extend_from_slice(x);
            let n = self.acc.len() / k * k;
            let out = self.acc[..n]
                .chunks(k)
                .map(|c| c.iter().sum::<f32>() / k as f32)
                .collect();
            self.acc.drain(..n);
            return out;
        }
        let step = self.rate as f64 / SR as f64;
        let mut out = Vec::new();
        // pos 是相对于当前块开头的位置（-1 表示上一块最后一个样本）
        while self.pos < x.len() as f64 - 1.0 {
            let i = self.pos.floor();
            let f = (self.pos - i) as f32;
            let a = if i < 0.0 { self.prev } else { x[i as usize] };
            let b = x[(i + 1.0) as usize];
            out.push(a + (b - a) * f);
            self.pos += step;
        }
        self.pos -= x.len() as f64;
        self.prev = *x.last().unwrap_or(&self.prev);
        out
    }
}

impl Read for Capture {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.pending.is_empty() {
            let Some(rx) = &self.rx else { return Ok(0) };
            match rx.recv() {
                Ok(b) => self.pending = b,
                // 已停止：采集线程退出、积压的都读完了
                Err(_) => {
                    self.finish();
                    return Ok(0);
                }
            }
        }
        let n = out.len().min(self.pending.len());
        out[..n].copy_from_slice(&self.pending[..n]);
        self.pending.drain(..n);
        Ok(n)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 读的一方很慢（模拟识别线程在解码）时，采集到的声音一个样本都不能丢
    #[test]
    fn slow_reader_loses_nothing() {
        let buf: Buf = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        let feeder = {
            let (buf, stop) = (buf.clone(), stop.clone());
            // 按墙钟送数，模拟麦克风
            std::thread::spawn(move || {
                let t0 = Instant::now();
                let mut sent = 0usize;
                while !stop.load(Ordering::SeqCst) {
                    let due = (t0.elapsed().as_secs_f64() * SR as f64) as usize;
                    let mut q = buf.lock().unwrap();
                    q.extend(std::iter::repeat_n(0.5f32, due - sent));
                    sent = due;
                    let extra = q.len().saturating_sub(MAX_BACKLOG);
                    q.drain(..extra);
                    drop(q);
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
        };
        let (rx, producer) = spawn_producer(vec![buf], None, stop.clone());
        let mut got = Vec::new();
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(800));
            got.extend(rx.try_iter().flatten());
        }
        stop.store(true, Ordering::SeqCst);
        producer.join().unwrap();
        feeder.join().unwrap();
        let samples: Vec<f32> = got
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        // 开头可能有几个 tick 的静音（设备还没开始送数），之后应当没有空洞
        let first = samples.iter().position(|&v| v > 0.0).unwrap();
        let zeros = samples[first..].iter().filter(|&&v| v == 0.0).count();
        assert!(zeros < SR as usize / 10, "{zeros} 个样本被补成了静音");
        assert!(samples.len() > SR as usize * 2);
    }

    #[test]
    fn resampler_keeps_duration() {
        for rate in [48000, 44100, 16000] {
            let mut r = Resampler::new(rate);
            let mut n = 0;
            for _ in 0..10 {
                n += r.push(&vec![0.5; rate as usize / 10]).len();
            }
            assert!((n as i64 - SR as i64).abs() <= 2, "{rate}: {n}");
        }
    }
}
