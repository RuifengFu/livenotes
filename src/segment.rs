//! 把带时间戳的单词切成字幕条目。
//!
//! 输入按 VAD 段分组；VAD 段之间是真实停顿，永远是硬边界。段内先按句末标点
//! 切句，再把超过时长/字数上限的句子递归切开：切分点优先选逗号、停顿、连词前，
//! 避免切在冠词/介词后。最后按阅读速度决定每条的显示时长。

use similar::{Algorithm, DiffTag, capture_diff_slices, diff_ratio};

#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    /// 秒
    pub start: f64,
    /// 秒；CTC 时间戳只标出单词起点，终点是估计值
    pub end: f64,
}

impl Word {
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cue {
    pub words: Vec<Word>,
    pub start: f64,
    pub end: f64,
    pub zh: String,
    /// 所属 VAD 段序号
    pub seg: usize,
    /// 说话人（开了声纹识别时才有）
    pub speaker: Option<String>,
}

impl Cue {
    pub fn text(&self) -> String {
        join(&self.words)
    }
    fn speech_start(&self) -> f64 {
        self.words[0].start
    }
    fn speech_end(&self) -> f64 {
        self.words[self.words.len() - 1].end
    }
}

#[derive(Clone, Debug)]
pub struct Limits {
    /// 每条字幕最长显示时间
    pub max_dur: f64,
    /// 英文每条最多字符（Netflix 规范 42×2）
    pub max_chars: usize,
    /// 英文阅读速度，字符/秒
    pub en_cps: f64,
    /// 中文阅读速度，字/秒
    pub zh_cps: f64,
    /// 最短显示时间（后面有空间时）
    pub min_dur: f64,
    /// 比开口提前多少出现
    pub lead: f64,
    /// 说完后多停留多久
    pub tail: f64,
    /// 相邻字幕之间最小间隔
    pub gap: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_dur: 5.0,
            max_chars: 84,
            en_cps: 17.0,
            zh_cps: 9.0,
            min_dur: 1.0,
            lead: 0.05,
            tail: 0.3,
            gap: 0.04,
        }
    }
}

/// 在这些词前面切比较自然
const BREAK_BEFORE: &[&str] = &[
    "and", "but", "or", "so", "because", "which", "who", "whom", "whose", "that", "when", "while",
    "if", "where", "although", "though", "then", "unless", "until", "since", "whereas", "after",
    "before", "as", "like", "just",
];
/// 在这些词后面切很别扭
const NO_BREAK_AFTER: &[&str] = &[
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "by", "from", "my", "your",
    "his", "her", "their", "our", "its", "this", "these", "those", "i", "we", "you", "they", "he",
    "she", "it", "is", "are", "was", "were", "very", "really", "not", "and", "or", "but", "into",
    "about", "some", "any",
];
/// 单独成条时直接丢掉的语气词
const FILLERS: &[&str] = &["u", "uh", "um", "umm", "er", "ah", "eh", "hmm", "mm", "mhm"];
const ABBREV: &[&str] = &[
    "mr", "mrs", "ms", "dr", "st", "vs", "eg", "ie", "etc", "us", "uk",
];

fn join(ws: &[Word]) -> String {
    ws.iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// 小写并去掉标点，用于比较
pub fn bare(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '’')
        .flat_map(char::to_lowercase)
        .collect()
}

fn strip_closers(t: &str) -> &str {
    t.trim_end_matches(['"', '\'', ')', ']', '”', '’'])
}

fn ends_sentence(t: &str) -> bool {
    let s = strip_closers(t);
    if s.ends_with(['?', '!', '…']) {
        return true;
    }
    s.ends_with('.') && !ABBREV.contains(&bare(s).as_str())
}

fn ends_clause(t: &str) -> bool {
    let s = strip_closers(t);
    s.ends_with([',', ';', ':', '—', '–']) || s.ends_with("--")
}

/// 按单词长度估计终点（不越过下一个词的起点），段内最后一个词延伸到 VAD 段尾。
pub fn estimate_ends(words: &mut [Word], seg_end: f64) {
    let n = words.len();
    for i in 0..n {
        let s = words[i].start;
        let guess = s + (0.075 * words[i].text.chars().count() as f64).max(0.2);
        words[i].end = if i + 1 < n {
            guess.min(words[i + 1].start).max(s + 0.05)
        } else {
            guess.max(seg_end.min(s + 1.5))
        };
    }
}

fn dur(ws: &[Word]) -> f64 {
    ws[ws.len() - 1].end - ws[0].start
}

fn chars(ws: &[Word]) -> usize {
    ws.iter().map(|w| w.text.chars().count()).sum::<usize>() + ws.len() - 1
}

fn fits(ws: &[Word], lim: &Limits) -> bool {
    dur(ws) <= lim.max_dur - lim.lead && chars(ws) <= lim.max_chars
}

fn split_score(ws: &[Word], k: usize, lim: &Limits) -> f64 {
    let (left, right) = ws.split_at(k);
    let (prev, next) = (&ws[k - 1], &ws[k]);
    let mut score = 0.0;
    if ends_sentence(&prev.text) {
        score += 10.0;
    } else if ends_clause(&prev.text) {
        score += 6.0;
    }
    score += (next.start - prev.end).clamp(0.0, 1.0) * 8.0;
    if BREAK_BEFORE.contains(&bare(&next.text).as_str()) {
        score += 3.0;
    }
    if NO_BREAK_AFTER.contains(&bare(&prev.text).as_str()) && !ends_clause(&prev.text) {
        score -= 4.0;
    }
    let total = dur(ws).max(1e-3);
    score -= 4.0 * (dur(left) - dur(right)).abs() / total;
    if fits(left, lim) && fits(right, lim) {
        score += 2.0;
    }
    for side in [left, right] {
        if side.len() < 2 {
            score -= 5.0;
        }
        if dur(side) < 0.8 {
            score -= 3.0;
        }
    }
    score
}

/// 递归切分直到每块都满足时长和字数上限。
fn fit(ws: &[Word], lim: &Limits, out: &mut Vec<Vec<Word>>) {
    if ws.len() <= 1 || fits(ws, lim) {
        out.push(ws.to_vec());
        return;
    }
    let k = (1..ws.len())
        .max_by(|&a, &b| split_score(ws, a, lim).total_cmp(&split_score(ws, b, lim)))
        .expect("len >= 2");
    fit(&ws[..k], lim, out);
    fit(&ws[k..], lim, out);
}

fn split_sentences(ws: &[Word]) -> Vec<&[Word]> {
    let mut out = Vec::new();
    let mut begin = 0;
    for (i, w) in ws.iter().enumerate() {
        if ends_sentence(&w.text) {
            out.push(&ws[begin..=i]);
            begin = i + 1;
        }
    }
    if begin < ws.len() {
        out.push(&ws[begin..]);
    }
    out
}

/// 把同一 VAD 段里过短的相邻条目（如 "Yeah." "Okay."）合并。
fn merge_short(cues: Vec<Cue>, lim: &Limits) -> Vec<Cue> {
    let mut out: Vec<Cue> = Vec::with_capacity(cues.len());
    for c in cues {
        if let Some(p) = out.last_mut()
            && p.seg == c.seg
            && (dur(&p.words) < lim.min_dur || dur(&c.words) < lim.min_dur)
        {
            let merged: Vec<Word> = p.words.iter().chain(&c.words).cloned().collect();
            if fits(&merged, lim) {
                p.words = merged;
                continue;
            }
        }
        out.push(c);
    }
    out
}

pub fn build_cues(segments: &[Vec<Word>], lim: &Limits) -> Vec<Cue> {
    let mut cues = Vec::new();
    for (seg, ws) in segments.iter().enumerate() {
        for sent in split_sentences(ws) {
            let mut chunks = Vec::new();
            fit(sent, lim, &mut chunks);
            cues.extend(chunks.into_iter().map(|words| Cue {
                words,
                start: 0.0,
                end: 0.0,
                zh: String::new(),
                seg,
                speaker: None,
            }));
        }
    }
    let mut cues = merge_short(cues, lim);
    // 丢掉只剩语气词或标点（如识别出一个孤零零的 "."）的条目
    cues.retain(|c| {
        !c.words.iter().all(|w| {
            let b = bare(&w.text);
            b.is_empty() || FILLERS.contains(&b.as_str())
        })
    });
    retime(&mut cues, lim, 0.0);
    cues
}

/// 按阅读速度决定显示时长：不早于开口、不和下一条重叠、不超过上限。
/// 翻译完成后再调用一次，让中文较长的条目也有足够的阅读时间。
/// `not_before` 是第一条允许的最早开始时间（流式输出时为上一条已输出字幕的结束 + 间隔）。
pub fn retime(cues: &mut [Cue], lim: &Limits, not_before: f64) {
    for i in 0..cues.len() {
        let prev_end = if i > 0 {
            cues[i - 1].end + lim.gap
        } else {
            not_before
        };
        let next_start = cues
            .get(i + 1)
            .map(|n| n.speech_start() - lim.lead - lim.gap);
        let c = &mut cues[i];
        c.start = (c.speech_start() - lim.lead).max(prev_end).max(0.0);
        let read = (c.text().chars().count() as f64 / lim.en_cps)
            .max(c.zh.chars().count() as f64 / lim.zh_cps)
            .max(lim.min_dur);
        let want = (c.speech_end() + lim.tail).max(c.start + read);
        let mut cap = c.start + lim.max_dur;
        if let Some(n) = next_start {
            cap = cap.min(n);
        }
        c.end = want.min(cap).max(c.start + 0.3);
    }
}

/// 把 LLM 补过标点（可能顺手改了个别词）的文本对齐回原单词的时间戳。
///
/// 改动太大（对齐率 < 0.6）时返回原单词，保证不会因为 LLM 胡写而丢内容。
pub fn apply_punctuation(words: &[Word], text: &str) -> Vec<Word> {
    let toks: Vec<&str> = text.split_whitespace().collect();
    if toks.is_empty() || words.is_empty() {
        return words.to_vec();
    }
    let a: Vec<String> = words.iter().map(|w| bare(&w.text)).collect();
    let b: Vec<String> = toks.iter().map(|t| bare(t)).collect();
    let ops = capture_diff_slices(Algorithm::Myers, &a, &b);
    if diff_ratio(&ops, a.len(), b.len()) < 0.6 {
        return words.to_vec();
    }

    // Myers 只给 Delete/Insert，把相邻的一对合并成 Replace，方便分配时间
    let mut spans: Vec<(DiffTag, std::ops::Range<usize>, std::ops::Range<usize>)> = Vec::new();
    for op in &ops {
        let (tag, o, n) = op.as_tag_tuple();
        if let Some(last) = spans.last_mut() {
            let pair = matches!(
                (last.0, tag),
                (DiffTag::Delete | DiffTag::Replace, DiffTag::Insert)
                    | (DiffTag::Insert | DiffTag::Replace, DiffTag::Delete)
            );
            if pair {
                last.0 = DiffTag::Replace;
                last.1 = last.1.start.min(o.start)..last.1.end.max(o.end);
                last.2 = last.2.start.min(n.start)..last.2.end.max(n.end);
                continue;
            }
        }
        spans.push((tag, o, n));
    }

    let mut out: Vec<Word> = Vec::with_capacity(toks.len());
    let mut pending: Vec<&str> = Vec::new();
    fn attach<'a>(out: &mut [Word], pending: &mut Vec<&'a str>, t: &'a str) {
        match out.last_mut() {
            Some(w) => {
                if !bare(t).is_empty() {
                    w.text.push(' ');
                }
                w.text.push_str(t);
            }
            None => pending.push(t),
        }
    }
    for (tag, o, n) in spans {
        match tag {
            DiffTag::Equal => {
                for (i, j) in o.zip(n) {
                    out.push(Word::new(toks[j], words[i].start, words[i].end));
                }
            }
            DiffTag::Replace if !o.is_empty() => {
                let (t0, t1) = (words[o.start].start, words[o.end - 1].end);
                let step = (t1 - t0) / n.len() as f64;
                for (k, j) in n.enumerate() {
                    if bare(toks[j]).is_empty() && !out.is_empty() {
                        attach(&mut out, &mut pending, toks[j]);
                    } else {
                        let s = t0 + k as f64 * step;
                        out.push(Word::new(toks[j], s, s + step));
                    }
                }
            }
            DiffTag::Insert | DiffTag::Replace => {
                for j in n {
                    attach(&mut out, &mut pending, toks[j]);
                }
            }
            // 丢掉 LLM 认为多余的词（通常是口吃重复）
            DiffTag::Delete => {}
        }
    }
    if out.is_empty() {
        return words.to_vec();
    }
    if !pending.is_empty() {
        out[0].text = format!("{} {}", pending.join(" "), out[0].text);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每个词 0.3s，连续不停顿
    fn words(text: &str, start: f64) -> Vec<Word> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, t)| {
                let s = start + i as f64 * 0.3;
                Word::new(t, s, s + 0.3)
            })
            .collect()
    }

    #[test]
    fn short_sentence_stays_whole() {
        let cues = build_cues(
            &[words("Hello there, how are you?", 0.0)],
            &Limits::default(),
        );
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].text(), "Hello there, how are you?");
    }

    #[test]
    fn splits_at_sentence_end() {
        let ws = words(
            "This is the first sentence here. And this is the second one here.",
            0.0,
        );
        let cues = build_cues(&[ws], &Limits::default());
        let texts: Vec<_> = cues.iter().map(Cue::text).collect();
        assert_eq!(
            texts,
            [
                "This is the first sentence here.",
                "And this is the second one here."
            ]
        );
    }

    #[test]
    fn long_sentence_respects_max_duration_and_prefers_comma() {
        let lim = Limits::default();
        let ws = words(
            "when you train a neural network on a very large dataset of images, \
             the model slowly learns features that generalize to new examples",
            0.0,
        );
        let cues = build_cues(&[ws], &lim);
        assert!(cues.len() >= 2);
        assert!(cues[0].text().ends_with("images,"), "{:?}", cues[0].text());
        for c in &cues {
            assert!(c.end - c.start <= lim.max_dur + 1e-9, "{c:?}");
            assert!(c.text().len() <= lim.max_chars);
        }
    }

    #[test]
    fn never_breaks_after_article() {
        let lim = Limits {
            max_dur: 2.0,
            ..Limits::default()
        };
        let ws = words(
            "we looked at the results of the experiment and they were great",
            0.0,
        );
        for c in build_cues(&[ws], &lim) {
            let last = c.words.last().unwrap();
            assert!(
                !["the", "a", "of"].contains(&last.text.as_str()),
                "{}",
                c.text()
            );
        }
    }

    #[test]
    fn prefers_pause() {
        let mut ws = words(
            "one two three four five six seven eight nine ten eleven twelve",
            0.0,
        );
        // "seven" 前有 0.8s 停顿
        for w in ws.iter_mut().skip(6) {
            w.start += 0.8;
            w.end += 0.8;
        }
        let cues = build_cues(
            &[ws],
            &Limits {
                max_dur: 3.0,
                ..Limits::default()
            },
        );
        assert_eq!(cues[0].text(), "one two three four five six");
    }

    #[test]
    fn drops_filler_only_cues() {
        let lim = Limits::default();
        let cues = build_cues(
            &[
                words("So we start here.", 0.0),
                words("Uh.", 3.0),
                words("Next part.", 6.0),
            ],
            &lim,
        );
        let texts: Vec<_> = cues.iter().map(Cue::text).collect();
        assert_eq!(texts, ["So we start here.", "Next part."]);
    }

    #[test]
    fn vad_segments_are_hard_boundaries_and_short_bits_merge() {
        let lim = Limits::default();
        let cues = build_cues(
            &[
                words("Yeah. Okay. So let's start.", 0.0),
                words("Next part.", 5.0),
            ],
            &lim,
        );
        let texts: Vec<_> = cues.iter().map(Cue::text).collect();
        assert_eq!(texts, ["Yeah. Okay. So let's start.", "Next part."]);
    }

    #[test]
    fn timing_no_overlap_and_reading_speed() {
        let lim = Limits::default();
        let mut cues = build_cues(
            &[
                words(
                    "A fairly long line of dialogue that takes a while to read.",
                    0.0,
                ),
                words("Next.", 4.0),
            ],
            &lim,
        );
        cues[0].zh = "很长的中文译文".repeat(6); // 42 字，需要 4.7s 阅读
        retime(&mut cues, &lim, 0.0);
        for pair in cues.windows(2) {
            assert!(pair[0].end < pair[1].start);
        }
        for c in &cues {
            assert!(c.end - c.start <= lim.max_dur + 1e-9);
            assert!(c.end - c.start >= 0.3);
        }
        // 想显示 4.7s，但下一条 4s 开口，只能延长到它之前
        assert!((cues[0].end - (4.0 - lim.lead - lim.gap)).abs() < 1e-9);
    }

    #[test]
    fn estimate_ends_extends_last_word_to_segment_end() {
        let mut ws = vec![Word::new("hello", 1.0, 1.06), Word::new("world", 1.4, 1.46)];
        estimate_ends(&mut ws, 2.0);
        assert!((ws[0].end - 1.375).abs() < 1e-9);
        assert!((ws[1].end - 2.0).abs() < 1e-9);
    }

    #[test]
    fn punctuation_alignment_keeps_timestamps() {
        let ws = words(
            "welcome back to the channel today we talk about the lost landscape",
            0.0,
        );
        let out = apply_punctuation(
            &ws,
            "Welcome back to the channel. Today, we talk about the loss landscape.",
        );
        let texts: Vec<_> = out.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            texts,
            [
                "Welcome",
                "back",
                "to",
                "the",
                "channel.",
                "Today,",
                "we",
                "talk",
                "about",
                "the",
                "loss",
                "landscape."
            ]
        );
        for (a, b) in ws.iter().zip(&out) {
            assert_eq!(a.start, b.start);
        }
    }

    #[test]
    fn punctuation_alignment_attaches_detached_marks() {
        let ws = words("so what now", 0.0);
        let out = apply_punctuation(&ws, "So — what now ?");
        let texts: Vec<_> = out.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["So—", "what", "now?"]);
    }

    #[test]
    fn punctuation_alignment_rejects_rewrites() {
        let ws = words("the cat sat on the mat", 0.0);
        let out = apply_punctuation(&ws, "A completely different sentence appears here instead.");
        assert_eq!(out, ws);
    }
}
