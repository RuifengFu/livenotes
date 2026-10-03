//! 会话产物：中英对照文字稿 + 流式 AI 笔记（+ 会议纪要）。
//!
//! 字幕一条条进来：写进文字稿（按停顿/时长分段）；攒够一段讲稿就交给后台线程，
//! 由 DeepSeek 流式生成这一部分笔记，边生成边写文件，处理中途就能打开看。

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};

use crate::llm::DeepSeek;
use crate::segment::Cue;

/// 秒 → 显示用时间（视频用相对时间，会议用墙钟时间）
pub type Clock = Arc<dyn Fn(f64) -> String + Send + Sync>;

pub fn relative_clock() -> Clock {
    Arc::new(|t| {
        let s = t as u64;
        format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    })
}

pub fn wall_clock(start: chrono::DateTime<chrono::Local>) -> Clock {
    Arc::new(move |t| {
        (start + chrono::Duration::milliseconds((t * 1000.0) as i64))
            .format("%H:%M:%S")
            .to_string()
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Style {
    /// 课堂笔记
    Lecture,
    /// 会议记录 + 结束后的会议纪要
    Meeting,
}

/// 所有笔记 Prompt 共用：提醒模型输入是自动转写，有识别错误
const ASR_NOTE: &str = "\
注意：原文是**语音识别自动转写**的英文（不是讲稿），常有同音词、术语拼错、漏词、断句错误，\
例如 \"stop domain\" / \"scope domain\" 实为 \"Scott domain\"。请结合上下文和专业知识推断真实意思，\
笔记里写正确的术语；实在无法确定的，保留原文并标「（?）」，不要编造原文没有的内容。
听不清、乱码的零碎句子直接略过，**不要在笔记里评论转写质量**（不要写「转写不完整」「识别混乱」之类）。
";

const LECTURE_PROMPT: &str = "\
你是擅长把技术讲座讲清楚的助教。下面是一场英文讲座实时转写的第 {current} 部分（共 {total} 部分），\
请整理成**易懂、好记的中文课堂笔记**：
- 层级清晰（标题、小标题、要点），核心概念和术语加粗，术语首次出现时附英文原文
- 公式和重要算法用标准 LaTeX（行内 \\( \\)，独立公式 $$ $$）
- 通俗易懂，不照抄晦涩原文
- **不要漏掉课堂上的「事务性」信息**：老师推荐的书、课程、讲义、网站、论文、工具，作业、考试、\
截止时间、课程安排、通知等，哪怕只是顺口一提。放在这一部分末尾的「### 📌 资料与通知」下，\
写清名称和老师的原话要点（如「推荐了某暑期学校的 XX 课程，讲义网上能找到」）。\
只记真正的资料和通知，**没有就整节省略**（不要写「无」或「本部分未出现」）
- 有学生提问 / 课堂互动时，用「**提问：**…… **回答：**……」记下来。原文如果用声纹标了说话人\
（说话人A、说话人B…），说得最多的通常是老师，其他人多半是学生；笔记里写「老师」「学生」，不要写说话人A
- 禁止任何开场白/结束语，禁止写「第 X 部分」之类的元信息，直接从内容标题开始
";

const LECTURE_FINAL_PROMPT: &str = "\
下面是一节英文课的分段中文笔记。请写一份课后总结（Markdown），第一行是 `# 课后总结`，不要开场白，\
然后依次是下面四个二级标题（标题照抄，不要带括号里的说明）：
## 本节课一句话
## 核心要点 —— 5-8 条，每条一两句
## 📌 提到的资料与通知 —— 书、课程、讲义、网站、作业、考试、课程安排等，把各部分「资料与通知」汇总去重；没有就写「无」
## 课后复习问题 —— 3-5 个
只根据笔记内容写，不要编造。

分段笔记如下：
";

const MEETING_PART_PROMPT: &str = "\
你是专业的会议记录员。下面是一场英文线上会议的第 {current} 段讲话原文。
请整理成中文会议记录，要求：
- 按议题分层（### 议题），每个议题下列出讨论要点，保留关键数据、名称、时间
- 明确出现的**决定**和**待办事项**（谁、做什么、何时）要单独加粗标出
- 提到的文档、链接、工具、资料也要记下来
- 专有名词、项目名、人名保留英文原文
- 如果原文没有标出说话人，不要臆测是谁说的，除非原文明确提到名字
- 不要寒暄，不要写「本段」「第 X 部分」之类的元信息，直接从内容开始
";

const MEETING_FINAL_PROMPT: &str = "\
下面是一场英文线上会议的分段中文记录。请据此写一份简洁的中文会议纪要（Markdown）：
## 一句话概要
## 主要议题（每个议题 2-4 条要点）
## 结论与决定
## 待办事项（表格：事项 | 负责人 | 截止时间；没提到的写「未提及」）
## 未决问题 / 需要跟进
只根据记录内容写，不要编造。

分段记录如下：
";

#[derive(Clone)]
struct Line {
    start: f64,
    end: f64,
    en: String,
    zh: String,
    speaker: Option<String>,
}

fn ends_sentence(en: &str) -> bool {
    en.trim_end_matches(['"', '\'', ')', ']', ' '])
        .ends_with(['.', '?', '!', '…'])
}

/// 把逐条中文字幕连成段落：英文句末处补句号/问号，其余补逗号。
fn join_zh(lines: &[Line]) -> String {
    let mut out = String::new();
    for l in lines {
        let z = l.zh.trim();
        if z.is_empty() {
            continue;
        }
        out.push_str(z);
        if !z.ends_with(['？', '！', '?', '!', '。', '，', '…']) {
            let en = l.en.trim_end_matches(['"', '\'', ')', ']', ' ']);
            out.push(if en.ends_with('?') {
                '？'
            } else if ends_sentence(en) {
                '。'
            } else {
                '，'
            });
        }
    }
    if out.ends_with('，') {
        out.pop();
        out.push('。');
    }
    out
}

/// 给 LLM 的讲稿原文：有说话人时按人分行（"说话人A: ..."），否则连成一段
fn speech_text(lines: &[Line]) -> String {
    if lines.iter().all(|l| l.speaker.is_none()) {
        let en: Vec<&str> = lines.iter().map(|l| l.en.as_str()).collect();
        return en.join(" ");
    }
    let mut out = String::from(
        "（下面的讲稿由声纹自动区分了说话人，可能有少量标错；整理时请保留谁说了什么）\n",
    );
    let mut last: Option<&str> = None;
    for l in lines {
        let who = l.speaker.as_deref().unwrap_or("未知");
        if last != Some(who) {
            out.push_str(&format!("\n{who}: "));
            last = Some(who);
        } else {
            out.push(' ');
        }
        out.push_str(&l.en);
    }
    out
}

/// 中英对照文字稿：停顿超过 2s，或段落超过 60s 且在句末时分段。
struct Transcript {
    f: File,
    clock: Clock,
    para: Vec<Line>,
}

impl Transcript {
    fn create(path: &Path, title: &str, clock: Clock) -> Result<Self> {
        let mut f = File::create(path)?;
        writeln!(f, "# {title} 文字稿\n")?;
        Ok(Self {
            f,
            clock,
            para: Vec::new(),
        })
    }

    fn add(&mut self, line: Line) -> Result<()> {
        if let Some(last) = self.para.last() {
            let long = line.start - self.para[0].start > 60.0 && ends_sentence(&last.en);
            if line.start - last.end > 2.0 || long || line.speaker != last.speaker {
                self.flush()?;
            }
        }
        self.para.push(line);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.para.is_empty() {
            return Ok(());
        }
        let en: Vec<&str> = self.para.iter().map(|l| l.en.as_str()).collect();
        let who = self.para[0]
            .speaker
            .as_ref()
            .map(|s| format!(" **{s}**："))
            .unwrap_or_default();
        writeln!(
            self.f,
            "**[{}]**{who} {}\n",
            (self.clock)(self.para[0].start),
            en.join(" ")
        )?;
        let zh = join_zh(&self.para);
        if !zh.is_empty() {
            writeln!(self.f, "{zh}\n")?;
        }
        self.f.flush()?;
        self.para.clear();
        Ok(())
    }
}

/// 流式写入，失败就把这次写的内容截掉重来。
fn stream_into(f: &mut File, llm: &DeepSeek, prompt: &str) -> Result<String> {
    let start = f.stream_position()?;
    let mut last = anyhow!("unreachable");
    for attempt in 1..=3 {
        match llm.stream_chat(prompt, |d| {
            Ok(f.write_all(d.as_bytes()).and_then(|_| f.flush())?)
        }) {
            Ok(text) => return Ok(text),
            Err(e) => {
                eprintln!("  ! 生成失败（第 {attempt}/3 次）: {e:#}");
                f.set_len(start)?;
                f.seek(SeekFrom::Start(start))?;
                std::thread::sleep(Duration::from_secs(5 * attempt));
                last = e;
            }
        }
    }
    Err(last)
}

struct Part {
    idx: usize,
    lines: Vec<Line>,
}

pub struct SessionConfig {
    pub dir: PathBuf,
    pub title: String,
    pub style: Style,
    /// 自定义笔记 Prompt（可含 {current} / {total}），None 用内置
    pub prompt: Option<String>,
    pub chunk_words: usize,
    pub clock: Clock,
    /// 实时在终端显示字幕
    pub echo: bool,
    /// 课程主题 / 术语提示
    pub hint: Option<String>,
    /// 同一门课前几节讲过的内容（标题），帮助理解术语、衔接
    pub context: Option<String>,
    /// 网页上显示的笔记进度
    pub status: Option<Arc<Mutex<NotesStatus>>>,
}

/// 笔记进度（给实时网页显示）
#[derive(Default, Clone, serde::Serialize)]
pub struct NotesStatus {
    /// 还没交给笔记的英文词数 / 每部分词数
    pub pending_words: usize,
    pub chunk_words: usize,
    /// 正在写第几部分（None：空闲）
    pub writing: Option<usize>,
    pub done_parts: usize,
    /// 正在写课后总结 / 会议纪要
    pub summarizing: bool,
}

/// 一次会话：接收字幕，写文字稿，分段交给笔记线程。
pub struct Session {
    transcript: Transcript,
    tx: Option<Sender<Part>>,
    worker: Option<JoinHandle<Result<Vec<String>>>>,
    buf: Vec<Line>,
    words: usize,
    parts: usize,
    cfg: SessionConfig,
    llm: DeepSeek,
    pub notes_path: PathBuf,
    pub transcript_path: PathBuf,
}

impl Session {
    pub fn start(cfg: SessionConfig, llm: DeepSeek) -> Result<Self> {
        std::fs::create_dir_all(&cfg.dir)?;
        let transcript_path = cfg.dir.join(format!("{} 文字稿.md", cfg.title));
        let notes_path = cfg.dir.join(format!("{} AI摘要.md", cfg.title));
        let transcript = Transcript::create(&transcript_path, &cfg.title, cfg.clock.clone())?;

        let mut f = File::create(&notes_path)?;
        let source = if cfg.style == Style::Meeting {
            "会议"
        } else {
            "讲座"
        };
        writeln!(f, "# {}\n\n> 由 AI 根据{source}实时转写生成\n", cfg.title)?;
        let template = match (&cfg.prompt, cfg.style) {
            (Some(p), _) => p.clone(),
            (None, Style::Lecture) => LECTURE_PROMPT.to_owned(),
            (None, Style::Meeting) => MEETING_PART_PROMPT.to_owned(),
        };
        let (tx, rx) = mpsc::channel::<Part>();
        let (clock, worker_llm) = (cfg.clock.clone(), llm.clone());
        let status = cfg.status.clone();
        let set_status = move |f: &dyn Fn(&mut NotesStatus)| {
            if let Some(s) = &status {
                f(&mut s.lock().unwrap());
            }
        };
        let mut background = String::new();
        if let Some(h) = cfg.hint.as_deref().filter(|h| !h.trim().is_empty()) {
            background += &format!("\n课程主题 / 术语提示：{h}\n");
        }
        if let Some(c) = &cfg.context {
            background += &format!(
                "\n（同一门课前几节已讲过下面这些内容，仅供理解术语和衔接，不要复述：）\n{c}\n"
            );
        }
        let worker = std::thread::spawn(move || -> Result<Vec<String>> {
            let mut headings: Vec<String> = Vec::new();
            let mut parts = Vec::new();
            for part in rx {
                let (t0, t1) = (
                    clock(part.lines[0].start),
                    clock(part.lines[part.lines.len() - 1].end),
                );
                eprintln!("  📝 笔记第 {} 部分（{t0} – {t1}）...", part.idx);
                set_status(&|s| s.writing = Some(part.idx));
                let mut prompt = template
                    .replace("{current}", &part.idx.to_string())
                    .replace("{total}", "N（仍在实时转写，总数未知）");
                prompt += ASR_NOTE;
                prompt += &background;
                if !headings.is_empty() {
                    let recent = &headings[headings.len().saturating_sub(30)..];
                    prompt += &format!(
                        "\n（前面部分已整理过以下标题，衔接上文即可，不要重复已讲过的内容：）\n{}\n",
                        recent.join("\n")
                    );
                }
                prompt += &format!("\n\n{}", speech_text(&part.lines));
                writeln!(f, "\n---\n\n> ⏱ {t0} – {t1}\n")?;
                let text = stream_into(&mut f, &worker_llm, &prompt)?;
                writeln!(f)?;
                f.flush()?;
                headings.extend(
                    text.lines()
                        .filter(|l| l.starts_with('#'))
                        .map(|l| l.trim().to_owned()),
                );
                parts.push(format!("[{t0} – {t1}]\n{text}"));
                set_status(&|s| {
                    s.writing = None;
                    s.done_parts += 1;
                });
            }
            Ok(parts)
        });
        Ok(Self {
            transcript,
            tx: Some(tx),
            worker: Some(worker),
            buf: Vec::new(),
            words: 0,
            parts: 0,
            cfg,
            llm,
            notes_path,
            transcript_path,
        })
    }

    pub fn on_cue(&mut self, c: &Cue) -> Result<()> {
        let line = Line {
            start: c.start,
            end: c.end,
            en: c.text(),
            zh: c.zh.clone(),
            speaker: c.speaker.clone(),
        };
        if self.cfg.echo {
            let mut out = std::io::stdout().lock();
            writeln!(
                out,
                "\x1b[2m{}\x1b[0m  {}\n          \x1b[2m{}\x1b[0m",
                (self.cfg.clock)(c.start),
                line.zh,
                line.en
            )?;
            out.flush()?;
        }
        self.words += line.en.split_whitespace().count();
        let boundary = ends_sentence(&line.en);
        self.transcript.add(line.clone())?;
        self.buf.push(line);
        // 攒够一段且在句末，就交给笔记线程
        if self.words >= self.cfg.chunk_words && boundary {
            self.submit();
        }
        self.set_status(|s| {
            s.pending_words = self.words;
            s.chunk_words = self.cfg.chunk_words;
        });
        Ok(())
    }

    fn set_status(&self, f: impl FnOnce(&mut NotesStatus)) {
        if let Some(s) = &self.cfg.status {
            f(&mut s.lock().unwrap());
        }
    }

    fn submit(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        self.parts += 1;
        let part = Part {
            idx: self.parts,
            lines: std::mem::take(&mut self.buf),
        };
        self.words = 0;
        if let Some(tx) = &self.tx {
            let _ = tx.send(part);
        }
    }

    /// 送出最后一段，等笔记写完；讲座在笔记末尾加课后总结，会议另写会议纪要。
    /// 返回纪要路径（如果有）。
    pub fn finish(mut self) -> Result<Option<PathBuf>> {
        self.submit();
        self.set_status(|s| s.pending_words = 0);
        self.transcript.flush()?;
        drop(self.tx.take());
        let parts = match self.worker.take().map(|w| w.join()) {
            Some(Ok(r)) => r?,
            Some(Err(_)) => bail!("笔记线程崩溃"),
            None => Vec::new(),
        };
        if parts.is_empty() {
            return Ok(None);
        }
        self.set_status(|s| s.summarizing = true);
        if self.cfg.style == Style::Lecture {
            eprintln!("  📋 生成课后总结...");
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&self.notes_path)?;
            writeln!(f, "\n---\n")?;
            stream_into(
                &mut f,
                &self.llm,
                &format!("{LECTURE_FINAL_PROMPT}{}", parts.join("\n\n")),
            )?;
            writeln!(f)?;
            self.set_status(|s| s.summarizing = false);
            return Ok(None);
        }
        eprintln!("  📋 生成会议纪要...");
        let path = self.cfg.dir.join(format!("{} 会议纪要.md", self.cfg.title));
        let mut f = File::create(&path)?;
        writeln!(f, "# {} 会议纪要\n", self.cfg.title)?;
        stream_into(
            &mut f,
            &self.llm,
            &format!("{MEETING_FINAL_PROMPT}{}", parts.join("\n\n")),
        )?;
        writeln!(f)?;
        self.set_status(|s| s.summarizing = false);
        Ok(Some(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(en: &str, zh: &str) -> Line {
        Line {
            start: 0.0,
            end: 1.0,
            en: en.into(),
            zh: zh.into(),
            speaker: None,
        }
    }

    #[test]
    fn joins_chinese_with_sentence_marks() {
        let ls = [
            line("What is it?", "到底是什么"),
            line("it is just", "其实就是"),
            line("two files.", "两个文件"),
        ];
        assert_eq!(join_zh(&ls), "到底是什么？其实就是，两个文件。");
        assert_eq!(join_zh(&[line("and then", "然后")]), "然后。");
    }

    #[test]
    fn clocks_format() {
        assert_eq!(relative_clock()(3725.9), "01:02:05");
        let start = chrono::Local::now();
        assert_eq!(wall_clock(start)(0.0), start.format("%H:%M:%S").to_string());
    }
}
