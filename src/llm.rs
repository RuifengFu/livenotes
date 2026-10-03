//! DeepSeek（OpenAI 兼容接口，关闭思考模式）：补标点、翻译字幕、流式生成笔记。

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

#[derive(Clone)]
pub struct DeepSeek {
    key: String,
    model: String,
    url: String,
    agent: ureq::Agent,
}

impl DeepSeek {
    pub fn new(key: String, model: String, base_url: &str) -> Self {
        // 流式输出可能持续较久，只限制连接和每次读取的等待时间
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(120))
            .build();
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        Self {
            key,
            model,
            url,
            agent,
        }
    }

    /// 要求模型返回 JSON 对象；网络错误、限流、5xx 自动重试。
    pub fn chat_json(&self, system: &str, user: &str, temperature: f32) -> Result<Value> {
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "stream": false,
        });
        let mut last_err = anyhow!("unreachable");
        for attempt in 0..5 {
            if attempt > 0 {
                std::thread::sleep(Duration::from_secs(2u64.pow(attempt)));
            }
            let resp = self
                .agent
                .post(&self.url)
                .set("Authorization", &format!("Bearer {}", self.key))
                .send_json(&body);
            match resp {
                Ok(r) => {
                    let v: Value = r.into_json().context("DeepSeek 返回的不是 JSON")?;
                    let content = v["choices"][0]["message"]["content"]
                        .as_str()
                        .ok_or_else(|| anyhow!("DeepSeek 响应缺少 content: {v}"))?;
                    return parse_json_prefix(content)
                        .with_context(|| format!("模型输出不是合法 JSON: {content}"));
                }
                Err(ureq::Error::Status(code, r)) if code == 429 || code >= 500 => {
                    last_err = anyhow!("HTTP {code}: {}", r.into_string().unwrap_or_default());
                }
                Err(ureq::Error::Status(code, r)) => {
                    bail!(
                        "DeepSeek HTTP {code}: {}",
                        r.into_string().unwrap_or_default()
                    );
                }
                Err(e) => last_err = e.into(),
            }
        }
        Err(last_err)
    }

    /// 流式对话：每收到一段输出就回调一次，返回完整文本。
    /// 失败时不重试（调用方已经把部分输出写出去了，由它决定如何回滚）。
    pub fn stream_chat(
        &self,
        prompt: &str,
        mut on_delta: impl FnMut(&str) -> Result<()>,
    ) -> Result<String> {
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"},
            "stream": true,
        });
        let resp = self
            .agent
            .post(&self.url)
            .set("Authorization", &format!("Bearer {}", self.key))
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(code, r) => {
                    anyhow!("HTTP {code}: {}", r.into_string().unwrap_or_default())
                }
                e => e.into(),
            })?;
        let mut text = String::new();
        for line in BufReader::new(resp.into_reader()).lines() {
            let line = line.context("读取流式响应失败")?;
            let Some(data) = line.strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if data == "[DONE]" {
                break;
            }
            let v: Value =
                serde_json::from_str(data).with_context(|| format!("流式数据不是 JSON: {data}"))?;
            if let Some(d) = v["choices"][0]["delta"]["content"].as_str() {
                on_delta(d)?;
                text.push_str(d);
            }
        }
        Ok(text)
    }

    /// 为每个 VAD 段的原始文本补标点和大小写。失败的段返回 None（调用方保留原文）。
    pub fn punctuate(&self, segments: &[String], workers: usize) -> Vec<Option<String>> {
        // 按单词数打包，一次请求 ~400 词
        let mut batches: Vec<Vec<usize>> = vec![vec![]];
        let mut n = 0;
        for (i, s) in segments.iter().enumerate() {
            let w = s.split_whitespace().count();
            if n + w > 400 && !batches.last().unwrap().is_empty() {
                batches.push(vec![]);
                n = 0;
            }
            batches.last_mut().unwrap().push(i);
            n += w;
        }
        let results = par_map(&batches, workers, |ids| {
            let input: BTreeMap<String, &str> = ids
                .iter()
                .map(|&i| (i.to_string(), segments[i].as_str()))
                .collect();
            let v = self.chat_json(PUNCT_PROMPT, &serde_json::to_string(&input)?, 0.3)?;
            Ok(ids
                .iter()
                .map(|i| v[i.to_string()].as_str().map(str::to_owned))
                .collect::<Vec<_>>())
        });
        let mut out = vec![None; segments.len()];
        for (ids, r) in batches.iter().zip(results) {
            match r {
                Ok(texts) => ids.iter().zip(texts).for_each(|(&i, t)| out[i] = t),
                Err(e) => eprintln!("  ! 补标点失败（保留原文）: {e:#}"),
            }
        }
        out
    }

    /// 逐条翻译成中文，带前后文。返回与输入等长的译文，失败的条目为空字符串。
    pub fn translate(
        &self,
        lines: &[String],
        before: &[String],
        hint: Option<&str>,
        workers: usize,
    ) -> Vec<String> {
        const BATCH: usize = 30;
        let ranges: Vec<(usize, usize)> = (0..lines.len())
            .step_by(BATCH)
            .map(|s| (s, (s + BATCH).min(lines.len())))
            .collect();
        let results = par_map(&ranges, workers, |&(s, e)| {
            Ok(self.translate_range(lines, before, s, e, hint))
        });
        results
            .into_iter()
            .flat_map(|r| r.unwrap_or_default())
            .collect()
    }

    /// 翻译 [s, e)；结果条数不对就重试，仍失败则二分缩小批次。
    fn translate_range(
        &self,
        lines: &[String],
        before: &[String],
        s: usize,
        e: usize,
        hint: Option<&str>,
    ) -> Vec<String> {
        // 上文：外部给的上一块结尾 + 本块前面的行，取最后 3 条
        let prior: Vec<&String> = before.iter().chain(&lines[..s]).collect();
        let input = json!({
            "hint": hint.unwrap_or(""),
            "context_before": prior[prior.len().saturating_sub(3)..],
            "lines": (s..e).map(|i| ((i + 1).to_string(), lines[i].as_str())).collect::<BTreeMap<_, _>>(),
            "context_after": lines[e..(e + 3).min(lines.len())],
        });
        for _ in 0..3 {
            match self.chat_json(TRANSLATE_PROMPT, &input.to_string(), 1.0) {
                Ok(v) => {
                    let got: Option<Vec<String>> = (s..e)
                        .map(|i| v[(i + 1).to_string()].as_str().map(clean_zh))
                        .collect();
                    if let Some(got) = got {
                        return got;
                    }
                }
                Err(err) => eprintln!("  ! 翻译请求失败，重试: {err:#}"),
            }
        }
        if e - s == 1 {
            eprintln!("  ! 翻译失败，留空: {}", lines[s]);
            return vec![String::new()];
        }
        let m = s + (e - s) / 2;
        let mut out = self.translate_range(lines, before, s, m, hint);
        out.extend(self.translate_range(lines, before, m, e, hint));
        out
    }
}

/// 解析开头的第一个 JSON 值，容忍模型在后面多吐的垃圾（偶尔会混入工具调用标记）。
fn parse_json_prefix(s: &str) -> Result<Value> {
    let mut it = serde_json::Deserializer::from_str(s.trim()).into_iter::<Value>();
    Ok(it.next().ok_or_else(|| anyhow!("空输出"))??)
}

/// 字幕习惯：句末不加句号。
fn clean_zh(t: &str) -> String {
    t.trim().trim_end_matches(['。', '.']).trim_end().to_owned()
}

/// 固定线程数并行处理，结果保持输入顺序。
fn par_map<T: Sync, R: Send>(
    items: &[T],
    workers: usize,
    f: impl Fn(&T) -> Result<R> + Sync,
) -> Vec<Result<R>> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Result<R>>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..workers.clamp(1, items.len().max(1)) {
            sc.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(i) else { break };
                    let r = f(item);
                    slots.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("every slot filled"))
        .collect()
}

const PUNCT_PROMPT: &str = "\
You restore punctuation and capitalization in English speech-recognition transcripts that will \
become video subtitles. The user sends a JSON object mapping ids to consecutive transcript \
segments from the same video. Return a JSON object with exactly the same ids, each value being \
that segment with proper punctuation (. , ? ! ; : and quotes) and sentence casing.
Rules:
- Keep every word, in the original order. Do not add, remove, paraphrase or merge words, and \
never move words between segments.
- You may fix a word only when it is an obvious recognition error given the context \
(e.g. \"lost landscape\" -> \"loss landscape\" in a talk about neural networks).
- Mark sentence boundaries carefully; long run-on speech should be split into sentences.
- End a segment with a period only if the sentence really ends there.";

const TRANSLATE_PROMPT: &str = "\
你是专业的影视字幕译者，把英文字幕逐条翻译成简体中文。
输入是 JSON：lines 是要翻译的字幕（键是编号）；context_before / context_after 是前后几条字幕，\
只供理解上下文，不要翻译；hint 是视频主题或术语提示（可能为空）。
要求：
1. lines 中每个编号输出一条译文，编号一一对应，不得合并、拆分或遗漏。
2. 一句英文常被拆成多条字幕。按中文语序通顺地分配到各条，每条译文尽量只对应该条英文的内容，\
让观众在对应时间看到对应的意思。
3. 口语化、简洁，符合中文字幕习惯：句末不加句号，句中停顿用逗号或空格；问号、感叹号保留。
4. 专有名词、术语用通行译法，没有通行译法的可保留英文。
英文来自语音识别自动转写，可能有同音词、术语拼错（如 \"stop domain\" 实为 \"Scott domain\"），\
按上下文理解真实意思后再翻译。
5. 只输出 JSON 对象，形如 {\"1\": \"译文\", \"2\": \"译文\"}。";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerates_trailing_garbage() {
        let v = parse_json_prefix("{\"31\": \"开始\"}</｜DSML｜ parameter>\n</invoke>").unwrap();
        assert_eq!(v["31"], "开始");
        assert!(parse_json_prefix("{\"31\": \"截断").is_err());
    }

    #[test]
    fn strips_final_period_only() {
        assert_eq!(clean_zh(" 这就是问题所在。"), "这就是问题所在");
        assert_eq!(clean_zh("真的吗？"), "真的吗？");
        assert_eq!(clean_zh("首先，我们"), "首先，我们");
    }
}
