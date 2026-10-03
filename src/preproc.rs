//! 只给 VAD 用的预处理：慢速 AGC（把远处、小声的说话人拉到正常电平）+ 可选 GTCRN 降噪。
//!
//! 识别仍然用原始音频：实测降噪后的音频直接给 Parakeet 反而更差（术语、小声处错得更多），
//! 但 VAD 在预处理后的音频上能更稳地切出小声的话。

use std::path::PathBuf;

use anyhow::{Context, Result};
use sherpa_onnx::{
    OfflineSpeechDenoiserModelConfig, OnlineSpeechDenoiser, OnlineSpeechDenoiserConfig,
};

#[derive(Clone, Debug, Default)]
pub struct PreprocConfig {
    pub agc: bool,
    /// gtcrn_simple.onnx 的路径；None 不降噪
    pub denoise: Option<PathBuf>,
}

impl PreprocConfig {
    pub fn enabled(&self) -> bool {
        self.agc || self.denoise.is_some()
    }
}

/// 每个音频流一份（带状态）
pub struct Preproc {
    agc: Option<Agc>,
    den: Option<OnlineSpeechDenoiser>,
}

impl Preproc {
    pub fn new(cfg: &PreprocConfig) -> Result<Self> {
        let den = match &cfg.denoise {
            Some(p) => {
                let mut m = OfflineSpeechDenoiserModelConfig::default();
                m.gtcrn.model = Some(p.to_string_lossy().into_owned());
                m.num_threads = 1;
                Some(
                    OnlineSpeechDenoiser::create(&OnlineSpeechDenoiserConfig { model: m })
                        .with_context(|| format!("加载降噪模型失败 {}", p.display()))?,
                )
            }
            None => None,
        };
        Ok(Self {
            agc: cfg.agc.then(Agc::default),
            den,
        })
    }

    /// 输入一段 16kHz 原始音频，返回给 VAD 的音频（降噪可能有缓冲，长度不一定相同，
    /// 但样本位置是对齐的：累计输出 = 累计输入减去模型内部缓冲）。
    pub fn process(&mut self, x: &[f32]) -> Vec<f32> {
        let mut y = match &self.den {
            Some(d) => d.run(x, crate::asr::SR as i32).samples,
            None => x.to_vec(),
        };
        if let Some(a) = &mut self.agc {
            a.process(&mut y);
        }
        y
    }

    /// 流结束：取出降噪模型里剩下的样本
    pub fn flush(&mut self) -> Vec<f32> {
        let mut y = match &self.den {
            Some(d) => d.flush().samples,
            None => Vec::new(),
        };
        if let Some(a) = &mut self.agc {
            a.process(&mut y);
        }
        y
    }
}

/// 慢速 AGC：只在明显高于底噪的帧上估计说话电平，增益慢升快降，最后软限幅。
pub struct Agc {
    target: f32,
    max_gain: f32,
    floor: f32,
    level: f32,
    gain: f32,
}

impl Default for Agc {
    /// 目标 -23 dBFS，最多放大 24 dB
    fn default() -> Self {
        Self::new(-23.0, 24.0)
    }
}

impl Agc {
    pub fn new(target_db: f32, max_gain_db: f32) -> Self {
        Self {
            target: db_to_amp(target_db),
            max_gain: db_to_amp(max_gain_db),
            floor: db_to_amp(-60.0),
            level: db_to_amp(target_db),
            gain: 1.0,
        }
    }

    pub fn process(&mut self, x: &mut [f32]) {
        const FRAME: usize = 160; // 10ms
        for f in x.chunks_mut(FRAME) {
            let rms = (f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32).sqrt() + 1e-9;
            // 底噪：快降慢升（约 +0.4 dB/s）
            if rms < self.floor {
                self.floor = rms;
            } else {
                self.floor *= 1.0005;
            }
            // 比底噪高 9 dB 的帧才算在说话，用来更新电平
            if rms > self.floor * 2.8 {
                let a = if rms > self.level { 0.05 } else { 0.01 };
                self.level += a * (rms - self.level);
            }
            let want = (self.target / self.level).clamp(1.0, self.max_gain);
            // 增益升慢（~1s）降快（~50ms），帧内线性过渡避免咔哒声
            let a = if want > self.gain { 0.01 } else { 0.2 };
            let g0 = self.gain;
            self.gain += a * (want - self.gain);
            let n = f.len() as f32;
            for (i, v) in f.iter_mut().enumerate() {
                let g = g0 + (self.gain - g0) * (i as f32 / n);
                *v = soft_limit(*v * g);
            }
        }
    }
}

fn soft_limit(v: f32) -> f32 {
    const K: f32 = 0.8;
    if v.abs() <= K {
        v
    } else {
        v.signum() * (K + (1.0 - K) * ((v.abs() - K) / (1.0 - K)).tanh())
    }
}

fn db_to_amp(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_db(x: &[f32]) -> f32 {
        10.0 * (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).log10()
    }

    /// 小声的"说话"（-45 dBFS 的断续正弦）会被拉到目标电平附近，且不超过满幅
    #[test]
    fn agc_lifts_quiet_speech() {
        let sr = 16000;
        let mut x: Vec<f32> = (0..sr * 6)
            .map(|i| {
                let on = (i / (sr / 2)) % 2 == 0; // 0.5s 说 / 0.5s 停
                let noise = ((i * 7919) % 101) as f32 / 101.0 - 0.5;
                let s = if on {
                    (i as f32 * 0.07).sin() * db_to_amp(-42.0)
                } else {
                    0.0
                };
                s + noise * db_to_amp(-65.0)
            })
            .collect();
        let mut agc = Agc::default();
        agc.process(&mut x);
        // 最后一个"说话"段（5.0–5.5s）
        let tail = &x[sr * 5..sr * 5 + sr / 2];
        let db = rms_db(tail);
        assert!(db > -32.0 && db < -18.0, "{db}");
        assert!(x.iter().all(|v| v.abs() <= 1.0));
    }
}
