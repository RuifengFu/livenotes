//! Qwen3-ASR 识别（Apple GPU / MLX）：常驻的 Python 子进程（qwen/qwen_server.py）。
//!
//! 每个 VAD 段发一次：JSON 头一行 + 原始 f32 样本；收到一行 JSON（带标点的词和时间）。
//! 子进程挂了、超时或报错时返回 Err，调用方退回 Parakeet，不会卡住实时字幕。

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

use crate::segment::Word;

/// 第一次运行要下载模型，给足时间
const STARTUP_TIMEOUT: Duration = Duration::from_secs(900);

pub struct Qwen {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    next_id: u64,
}

/// venv 和服务脚本的位置（install.sh 装在这里）
pub fn paths(data: &Path) -> (PathBuf, PathBuf) {
    (
        data.join("qwen-venv/bin/python"),
        data.join("qwen_server.py"),
    )
}

pub fn available(data: &Path) -> bool {
    let (py, script) = paths(data);
    py.exists() && script.exists()
}

impl Qwen {
    pub fn start(data: &Path) -> Result<Self> {
        let (py, script) = paths(data);
        if !py.exists() || !script.exists() {
            bail!("没有安装 Qwen 识别服务（运行 install.sh），或用 --engine parakeet");
        }
        let mut child = Command::new(&py)
            .arg(&script)
            .env("PYTHONUNBUFFERED", "1")
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("无法启动 {}", py.display()))?;
        let stdin = child.stdin.take().expect("piped");
        let stdout = child.stdout.take().expect("piped");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(&line)
                    && tx.send(v).is_err()
                {
                    break;
                }
            }
        });
        let ready = rx
            .recv_timeout(STARTUP_TIMEOUT)
            .map_err(|_| anyhow!("Qwen 识别服务没有启动成功"))?;
        if ready["ready"] != true {
            bail!("Qwen 识别服务启动失败: {ready}");
        }
        Ok(Self {
            child,
            stdin,
            rx,
            next_id: 0,
        })
    }

    /// 识别一段（16kHz 单声道），单词时间加上 `offset`
    pub fn transcribe(&mut self, samples: &[f32], offset: f64, context: &str) -> Result<Vec<Word>> {
        self.next_id += 1;
        let id = self.next_id;
        let secs = samples.len() as f64 / 16000.0;
        // 正常语速每秒 3-4 个 token；限制上限，防止模型偶尔复读停不下来
        let max_tokens = (secs * 8.0) as usize + 32;
        let head = serde_json::json!({
            "id": id, "n": samples.len(), "context": context, "max_tokens": max_tokens,
        });
        let mut buf = head.to_string().into_bytes();
        buf.push(b'\n');
        for s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        self.stdin
            .write_all(&buf)
            .and_then(|_| self.stdin.flush())
            .context("Qwen 识别服务已退出")?;
        // 正常远快于实时；给到段长 + 20s 还没回就放弃这段
        let deadline = Duration::from_secs_f64(secs + 20.0);
        loop {
            let v = self
                .rx
                .recv_timeout(deadline)
                .map_err(|_| anyhow!("Qwen 识别超时"))?;
            // 之前超时那段的迟到回复，丢掉
            if v["id"].as_u64() != Some(id) {
                continue;
            }
            if let Some(e) = v["error"].as_str() {
                bail!("Qwen 识别出错: {e}");
            }
            return Ok(parse_words(&v, offset));
        }
    }
}

fn parse_words(v: &Value, offset: f64) -> Vec<Word> {
    v["words"]
        .as_array()
        .map(|ws| {
            ws.iter()
                .filter_map(|w| {
                    let text = w[0].as_str()?.trim();
                    (!text.is_empty()).then(|| {
                        Word::new(
                            text,
                            offset + w[1].as_f64().unwrap_or(0.0),
                            offset + w[2].as_f64().unwrap_or(0.0),
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

impl Drop for Qwen {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_words_with_offset() {
        let v: Value = serde_json::from_str(
            r#"{"id":1,"text":"Hi, Tom.","words":[["Hi,",0.1,0.3],["Tom.",0.4,0.8],["",1,1]]}"#,
        )
        .unwrap();
        let ws = parse_words(&v, 10.0);
        assert_eq!(ws.len(), 2);
        assert_eq!(ws[1].text, "Tom.");
        assert!((ws[1].start - 10.4).abs() < 1e-9 && (ws[1].end - 10.8).abs() < 1e-9);
    }
}
