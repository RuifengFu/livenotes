//! subs：英文语音 → 自动断句中英字幕（SenseVoice + DeepSeek，流式）→ 可选 AI 笔记。
//!
//! 用法：
//! - `subs video.mp4`：只出字幕
//! - `subs <B站链接> --notes`：下载音轨，字幕 + 文字稿 + 课堂笔记放进会话目录
//! - `subs --live [--mic | --mic-only]`：实时录电脑声音 / 麦克风（会议、线下课），终端滚动显示字幕，
//!   Ctrl+C 结束后补完笔记并生成会议纪要

mod asr;
#[cfg(target_os = "macos")]
mod capture;
mod course;
mod diarize;
mod live;
mod llm;
mod notes;
mod online_spk;
mod pipeline;
mod preproc;
mod qwen;
mod segment;
mod srt;
mod web;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::asr::Input;
use crate::llm::DeepSeek;
use crate::notes::{Clock, NotesStatus, Session, SessionConfig, Style};
use crate::pipeline::Options;
use crate::segment::{Cue, Limits};

const EXAMPLES: &str = "\
\x1b[1m常用场景（照抄即可）\x1b[0m

  线下上课（录麦克风）：        subs --live --mic-only --style lecture --title 课堂
  线上会议（录 Teams/Zoom 声音）：subs --live --title 组会
  会议里连自己的声音一起录：    subs --live --mic --title 组会        （请戴耳机）
  同一门课每次都归到一起：    subs --live --mic-only --style lecture --course Haskell
  B 站 / YouTube 课程出笔记：   subs \"https://www.bilibili.com/video/BV...\" --notes
  本地视频只要字幕：            subs lecture.mp4

实时模式按 Ctrl+C 结束：先停录音，把剩下的内容处理完、写完笔记再退出（再按一次直接退出）。

\x1b[1m输出在哪\x1b[0m

  实时模式和 --notes：~/Documents/LiveNotes/<日期_标题>/（用 --course 时：LiveNotes/<课程>/<课程_日期>/）
    <标题>.zh-en.srt 等字幕 · 文字稿.md · AI摘要.md · 笔记.html · 会议纪要.md（会议）· 录音.wav（实时）
  写笔记时浏览器打开 http://127.0.0.1:8765 可以实时看笔记
  只出字幕：和输入文件放在一起

\x1b[1m第一次使用\x1b[0m

  DeepSeek key：写进 ~/.config/livenotes/env，一行 DEEPSEEK_API_KEY=sk-...
  录电脑声音：系统设置 → 隐私与安全性 → 屏幕与系统录音，允许你的终端
  录麦克风：  系统设置 → 隐私与安全性 → 麦克风，允许你的终端

更多选项：subs --help";

#[derive(Parser)]
#[command(
    name = "subs",
    version,
    about = "英文语音 → 实时中英字幕 + AI 笔记（Parakeet / SenseVoice 识别，DeepSeek 翻译）",
    after_help = EXAMPLES,
    after_long_help = EXAMPLES,
)]
struct Args {
    /// 要处理的视频/音频文件，或 B 站 / YouTube 链接（实时模式不用填）
    #[arg(value_name = "文件或链接")]
    inputs: Vec<String>,

    /// 实时模式：边录边出字幕（默认录电脑里播放的声音，如 Teams/Zoom）
    #[arg(long, help_heading = "实时录音")]
    live: bool,
    /// 只录麦克风（线下课、面对面交流）
    #[arg(long, conflicts_with = "mic", help_heading = "实时录音")]
    mic_only: bool,
    /// 电脑声音 + 麦克风一起录（线上会议里自己也发言时；请戴耳机，否则有回声）
    #[arg(long, help_heading = "实时录音")]
    mic: bool,
    /// 在刘海下方显示悬浮字幕条
    #[arg(long, help_heading = "实时录音")]
    overlay: bool,

    /// 同时生成文字稿和 AI 笔记（实时模式默认就会生成）
    #[arg(long, help_heading = "笔记")]
    notes: bool,
    /// 笔记类型：lecture 课堂笔记 / meeting 会议记录+纪要（默认：实时 meeting，视频 lecture）
    #[arg(long, value_enum, value_name = "类型", help_heading = "笔记")]
    style: Option<Style>,
    /// 标题，用于文件夹和文件名，如 "编译原理第3讲"
    #[arg(long, value_name = "标题", help_heading = "笔记")]
    title: Option<String>,
    /// 归入课程：每节课放在 <输出目录>/<课程>/ 下，自动沿用课程的主题提示，笔记参考前几节的内容
    #[arg(long, value_name = "课程", help_heading = "笔记")]
    course: Option<String>,
    /// 不开实时笔记网页（默认写笔记时会开 http://127.0.0.1:8765 ）
    #[arg(long, help_heading = "笔记")]
    no_web: bool,
    /// 课程主题 / 术语提示，让翻译更准，如 "类型论、Haskell"
    #[arg(long, value_name = "提示", help_heading = "笔记")]
    hint: Option<String>,
    /// 区分说话人（声纹）：字幕、文字稿、笔记里标出谁在说，结束后自动推断名字
    #[arg(long, help_heading = "笔记")]
    speakers: bool,
    /// 已知说话人数（不填自动判断）
    #[arg(long, value_name = "人数", help_heading = "笔记")]
    num_speakers: Option<usize>,
    /// 把这次认出名字的人的声纹存进声纹库，以后自动认出（~/.local/share/livenotes/voiceprints）
    #[arg(long, help_heading = "笔记")]
    save_voices: bool,
    /// 实时模式只要字幕，不写笔记
    #[arg(long, help_heading = "笔记")]
    no_notes: bool,

    /// 输出根目录（默认 ~/Documents/LiveNotes）
    #[arg(
        short,
        long,
        value_name = "目录",
        help_heading = "高级",
        hide_short_help = true
    )]
    out_dir: Option<PathBuf>,
    /// 识别引擎：qwen 最准（Apple GPU，默认）/ parakeet（CPU）/ sensevoice
    #[arg(long, value_enum, help_heading = "高级", hide_short_help = true)]
    engine: Option<asr::Engine>,
    /// 下载需要登录的视频时用的 cookies 文件
    #[arg(
        long,
        value_name = "文件",
        help_heading = "高级",
        hide_short_help = true
    )]
    cookies: Option<PathBuf>,
    /// 自定义笔记 Prompt 文件（可含 {current} / {total} 占位符）
    #[arg(
        long,
        value_name = "文件",
        help_heading = "高级",
        hide_short_help = true
    )]
    prompt_file: Option<PathBuf>,
    /// 每攒够多少英文词生成一部分笔记（默认：会议 600 / 讲座 800）
    #[arg(long, help_heading = "高级", hide_short_help = true)]
    chunk_words: Option<usize>,
    /// 每条字幕最长秒数
    #[arg(
        long,
        default_value_t = 5.0,
        help_heading = "高级",
        hide_short_help = true
    )]
    max_dur: f64,
    /// 每条英文字幕最多字符数
    #[arg(
        long,
        default_value_t = 84,
        help_heading = "高级",
        hide_short_help = true
    )]
    max_chars: usize,
    /// 只要英文字幕，不翻译
    #[arg(long, help_heading = "高级", hide_short_help = true)]
    no_translate: bool,
    /// 不用 LLM 补标点（只影响 sensevoice 引擎）
    #[arg(long, help_heading = "高级", hide_short_help = true)]
    no_punct: bool,
    /// DeepSeek 模型（字幕和笔记共用）
    #[arg(
        long,
        env = "SUBS_MODEL",
        default_value = "deepseek-flash",
        help_heading = "高级",
        hide_short_help = true
    )]
    model: String,
    /// 并行处理的块数（每块最多 2 个并发请求）
    #[arg(
        long,
        default_value_t = 3,
        help_heading = "高级",
        hide_short_help = true
    )]
    workers: usize,
    /// 每块语音秒数：识别出这么多语音就送去断句翻译（实时模式每段立即处理）
    #[arg(
        long,
        default_value_t = 90.0,
        help_heading = "高级",
        hide_short_help = true
    )]
    block_secs: f64,
    /// 在 stdout 逐行输出 JSON（给 App / 其他程序用）
    #[arg(long, help_heading = "高级", hide_short_help = true)]
    jsonl: bool,
    /// 识别用的 CPU 线程数
    #[arg(
        long,
        default_value_t = 6,
        help_heading = "高级",
        hide_short_help = true
    )]
    threads: i32,
    /// 改说话人名字（改会话目录里所有字幕/文字稿/笔记），如 --rename-speaker "说话人B=张老师" <会话目录>
    #[arg(long, value_name = "旧名=新名", help_heading = "笔记")]
    rename_speaker: Option<String>,
    /// 把会话里已确认名字的说话人声纹存进声纹库：--enroll <会话目录>
    #[arg(long, help_heading = "笔记")]
    enroll: bool,
    /// 把会话目录里的 AI 笔记生成网页版（<标题> 笔记.html，公式会渲染）：--html <会话目录>
    #[arg(long, help_heading = "笔记")]
    html: bool,
    /// 调试：只跑说话人分离并打印结果
    #[arg(long, hide = true)]
    debug_diarize: bool,
    /// 模型目录
    #[arg(
        long,
        env = "SUBS_MODELS",
        help_heading = "高级",
        hide_short_help = true
    )]
    models: Option<PathBuf>,
}

fn voices_dir() -> PathBuf {
    home().join(".local/share/livenotes/voiceprints")
}

/// 实时模式和 --notes 的默认输出根目录
fn default_out() -> PathBuf {
    home().join("Documents/LiveNotes")
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

/// 环境变量优先，其次 ~/.config/livenotes/env 里的 KEY=VALUE。
fn config_value(key: &str) -> Option<String> {
    if let Ok(v) = std::env::var(key)
        && !v.is_empty()
    {
        return Some(v);
    }
    let text = std::fs::read_to_string(home().join(".config/livenotes/env")).ok()?;
    text.lines().find_map(|l| {
        let (k, v) = l
            .trim()
            .strip_prefix("export ")
            .unwrap_or(l.trim())
            .split_once('=')?;
        (k.trim() == key).then(|| v.trim().trim_matches(['"', '\'']).to_owned())
    })
}

/// 标题 → 安全的文件名
fn sanitize(name: &str) -> String {
    let s: String = name
        .trim()
        .chars()
        .map(|c| {
            if "/\\:*?\"<>|".contains(c) || c.is_whitespace() {
                '_'
            } else {
                c
            }
        })
        .collect();
    let s = s.trim_matches(['_', '.']).to_owned();
    if s.is_empty() { "untitled".into() } else { s }
}

fn find_bin(name: &str) -> String {
    ["/opt/homebrew/bin", "/usr/local/bin"]
        .iter()
        .map(|d| format!("{d}/{name}"))
        .find(|p| Path::new(p).exists())
        .unwrap_or_else(|| name.to_owned())
}

/// 用 yt-dlp 只下载音轨（已下载过会直接复用），返回 (文件路径, 标题)。
fn download(url: &str, dir: &Path, cookies: Option<&Path>) -> Result<(PathBuf, String)> {
    std::fs::create_dir_all(dir)?;
    let mut cmd = Command::new(find_bin("yt-dlp"));
    cmd.args([
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--no-simulate",
        "--print",
        "after_move:filepath",
        "-o",
    ])
    .arg(dir.join("%(title)s.audio.%(ext)s"));
    if let Some(c) = cookies {
        cmd.arg("--cookies").arg(c);
    }
    let out = cmd
        .arg(url)
        .stderr(Stdio::inherit())
        .output()
        .context("无法运行 yt-dlp（brew install yt-dlp）")?;
    if !out.status.success() {
        bail!("yt-dlp 下载失败");
    }
    let path = String::from_utf8_lossy(&out.stdout)
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|l| PathBuf::from(l.trim()))
        .context("yt-dlp 没有输出文件路径")?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let title = name.split(".audio.").next().unwrap_or(&name).to_owned();
    Ok((path, title))
}

struct Job {
    input: Input,
    /// 字幕文件所在目录
    dir: PathBuf,
    /// 文件名前缀
    stem: String,
    session: Option<Session>,
    /// 实时声纹：给每个语音段标说话人
    speakers: Option<Arc<Mutex<online_spk::OnlineSpeakers>>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.live && args.inputs.len() > 1 {
        bail!("--live 最多接受一个输入（\"-\" 表示 stdin）");
    }
    if !args.live && args.inputs.is_empty() {
        use clap::CommandFactory;
        Args::command().print_help()?;
        std::process::exit(2);
    }
    let models = args
        .models
        .clone()
        .unwrap_or_else(|| home().join(".local/share/livenotes/models"));

    // 默认用 GPU 上的 Qwen；没装 Qwen 服务就退回 Parakeet
    let engine = args.engine.unwrap_or_else(|| {
        if qwen::available(models.parent().unwrap_or(&models)) {
            asr::Engine::Qwen
        } else {
            eprintln!("! 没有安装 Qwen 识别服务（install.sh），改用 Parakeet");
            asr::Engine::Parakeet
        }
    });
    let llm = match config_value("DEEPSEEK_API_KEY") {
        Some(key) => {
            let base = config_value("DEEPSEEK_BASE_URL")
                .unwrap_or_else(|| "https://api.deepseek.com".into());
            Some(DeepSeek::new(key, args.model.clone(), &base))
        }
        None => {
            eprintln!(
                "! 没有找到 DEEPSEEK_API_KEY（环境变量或 ~/.config/livenotes/env），\
                 只生成英文字幕，不补标点、不翻译、不写笔记"
            );
            None
        }
    };
    let mut opt = Options {
        lim: Limits {
            max_dur: args.max_dur,
            max_chars: args.max_chars,
            ..Limits::default()
        },
        // Parakeet 自带标点；SenseVoice 的英文标点很弱，用 LLM 补
        punct: !args.no_punct && engine == asr::Engine::Sensevoice,
        translate: !args.no_translate,
        hint: args.hint.clone(),
        workers: args.workers,
        block_secs: if args.live { 0.0 } else { args.block_secs },
        jsonl: args.jsonl,
        live: args.live,
        labeler: None,
    };
    let custom_prompt = match &args.prompt_file {
        Some(p) => {
            Some(std::fs::read_to_string(p).with_context(|| format!("读取 {}", p.display()))?)
        }
        None => None,
    };
    let want_notes = llm.is_some() && (args.notes || (args.live && !args.no_notes));
    let style = args.style.unwrap_or(if args.live {
        Style::Meeting
    } else {
        Style::Lecture
    });
    let session_cfg = |dir: PathBuf, title: String, clock: Clock| SessionConfig {
        dir,
        title,
        style,
        prompt: custom_prompt.clone(),
        // 实时模式分段小一些，笔记出得快（约 3 分钟一部分）
        chunk_words: args.chunk_words.unwrap_or(match (args.live, style) {
            (true, _) => 450,
            (false, Style::Meeting) => 600,
            (false, Style::Lecture) => 800,
        }),
        clock,
        echo: false,
        hint: None,
        context: None,
        status: None,
    };
    // 课程：没给 --course 返回 None
    let root = args.out_dir.clone().unwrap_or_else(default_out);
    let course = match args.course.as_deref().map(str::trim) {
        Some(c) if !c.is_empty() => Some(course::Course::open(
            &root,
            &sanitize(c),
            c,
            args.hint.as_deref(),
        )?),
        _ => None,
    };
    // 翻译和笔记用的主题提示：命令行优先，其次课程里存的
    let hint = args
        .hint
        .clone()
        .or_else(|| course.as_ref().and_then(|c| c.info.hint.clone()));
    opt.hint = hint.clone();
    // 笔记会话的完整配置（+ 课程上下文、网页进度）
    let notes_cfg =
        |dir: PathBuf, title: String, clock: Clock, status: &Arc<Mutex<NotesStatus>>| {
            let mut cfg = session_cfg(dir.clone(), title, clock);
            cfg.hint = hint.clone();
            cfg.context = course.as_ref().and_then(|c| c.previous_topics(&dir, 2));
            cfg.status = Some(status.clone());
            cfg
        };

    if args.html {
        let dir = PathBuf::from(args.inputs.first().context("请给出会话目录")?);
        let stem = dir
            .file_name()
            .context("目录名无效")?
            .to_string_lossy()
            .into_owned();
        let notes = dir.join(format!("{stem} AI摘要.md"));
        if !notes.exists() {
            bail!("找不到 {}", notes.display());
        }
        let out = dir.join(format!("{stem} 笔记.html"));
        web::export(&notes, &out, &stem)?;
        eprintln!("→ {}", out.display());
        return Ok(());
    }
    if args.enroll {
        let dir = PathBuf::from(args.inputs.first().context("请给出会话目录")?);
        let stem = dir
            .file_name()
            .context("目录名无效")?
            .to_string_lossy()
            .into_owned();
        let saved = diarize::enroll_session(&dir, &stem, &voices_dir())?;
        if saved.is_empty() {
            eprintln!(
                "没有可存的声纹（角色名、占位名和带（2）的名字不会存，先用 --rename-speaker 改成真名）"
            );
        }
        for (name, path) in saved {
            eprintln!("已存入声纹库：{name}（{}）", path.display());
        }
        return Ok(());
    }
    if let Some(spec) = &args.rename_speaker {
        let (from, to) = spec
            .split_once('=')
            .context("格式：--rename-speaker \"旧名=新名\" <会话目录>")?;
        let dir = PathBuf::from(args.inputs.first().context("请给出会话目录")?);
        let stem = dir
            .file_name()
            .context("目录名无效")?
            .to_string_lossy()
            .into_owned();
        let names =
            std::collections::HashMap::from([(from.trim().to_owned(), to.trim().to_owned())]);
        diarize::rename_in_dir(&dir, &stem, &names)?;
        eprintln!(
            "已把「{}」改成「{}」：{}",
            from.trim(),
            to.trim(),
            dir.display()
        );
        return Ok(());
    }
    if args.debug_diarize {
        let t = Instant::now();
        let audio = asr::decode_all(Path::new(&args.inputs[0]))?;
        let d = diarize::Diarizer::new(&models, args.threads, None)?;
        let turns = d.run(&audio)?;
        eprintln!(
            "分离 {:.0}s 音频用时 {:.1}s，{} 段",
            audio.len() as f64 / 16000.0,
            t.elapsed().as_secs_f64(),
            turns.len()
        );
        let n = turns.iter().map(|t| t.speaker + 1).max().unwrap_or(0);
        for s in 0..n {
            let total: f64 = turns
                .iter()
                .filter(|t| t.speaker == s)
                .map(|t| t.end - t.start)
                .sum();
            eprintln!("  说话人{s}: {:.0}s", total);
        }
        for t in turns.iter().take(40) {
            println!("{:7.1}-{:7.1}  S{}", t.start, t.end, t.speaker);
        }
        let profiles = d.profiles(&audio, &turns);
        for (i, a) in profiles.iter().enumerate() {
            for (j, b) in profiles.iter().enumerate().skip(i + 1) {
                if let (Some(a), Some(b)) = (a, b) {
                    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("  相似度 S{i}-S{j}: {:.2}", dot / (na * nb));
                }
            }
        }
        return Ok(());
    }

    let t = Instant::now();
    let asr = asr::Asr::new(
        &models,
        engine,
        args.threads,
        if args.live { 6.0 } else { 20.0 },
    )?
    .with_preproc({
        // 默认：有 GTCRN 模型就给 VAD 降噪；SUBS_PREPROC=none 关闭，=agc / =agc,gtcrn 做对比实验
        let v = std::env::var("SUBS_PREPROC").unwrap_or_else(|_| "gtcrn".into());
        let gtcrn = models.join("gtcrn_simple.onnx");
        preproc::PreprocConfig {
            agc: v.contains("agc"),
            denoise: (v.contains("gtcrn") && gtcrn.exists()).then_some(gtcrn),
        }
    })
    .with_context(hint.as_deref());
    eprintln!("模型加载 {:.1}s", t.elapsed().as_secs_f64());

    if args.live {
        let started = chrono::Local::now();
        let date = started.format("%Y-%m-%d_%H%M");
        let title = match (&course, &args.title) {
            (Some(c), Some(t)) => sanitize(&format!("{}_{date}_{t}", c.info.name)),
            (Some(c), None) => sanitize(&format!("{}_{date}", c.info.name)),
            (None, t) => sanitize(&format!("{date}_{}", t.as_deref().unwrap_or("会议"))),
        };
        let dir = course
            .as_ref()
            .map_or_else(|| root.clone(), |c| c.dir.clone())
            .join(&title);
        std::fs::create_dir_all(&dir)?;
        let stop = Arc::new(AtomicBool::new(false));
        {
            let stop = stop.clone();
            ctrlc::set_handler(move || {
                if stop.swap(true, Ordering::SeqCst) {
                    eprintln!("\n再次 Ctrl+C，直接退出");
                    std::process::exit(130);
                }
                eprintln!("\n⏹  停止录制，正在处理剩余内容（再按一次 Ctrl+C 直接退出）...");
            })?;
        }
        let input = match args.inputs.first().map(String::as_str) {
            Some("-") => Input::Raw(Box::new(std::io::stdin())),
            Some(other) => bail!("--live 的输入只能是 \"-\"（stdin），收到 {other}"),
            None => live_capture(
                !args.mic_only,
                args.mic || args.mic_only,
                &dir.join("录音.wav"),
                stop.clone(),
            )?,
        };
        let status = Arc::new(Mutex::new(NotesStatus::default()));
        let session = match &llm {
            Some(llm) if want_notes => Some(Session::start(
                notes_cfg(
                    dir.clone(),
                    title.clone(),
                    notes::wall_clock(started),
                    &status,
                ),
                llm.clone(),
            )?),
            _ => None,
        };
        let url = session.as_ref().and_then(|s| {
            start_web(
                &args,
                web::WebState {
                    title: title.clone(),
                    notes_path: s.notes_path.clone(),
                    live_json: Some(dir.join(format!("{title}.live.json"))),
                    notes: status.clone(),
                    started,
                    stopping: stop.clone(),
                    live: true,
                },
            )
        });
        if !args.jsonl {
            let src = if args.mic_only {
                "麦克风"
            } else if args.mic {
                "电脑声音 + 麦克风"
            } else {
                "电脑声音"
            };
            println!("\n🎧 正在实时转写{src}，Ctrl+C 结束\n");
            if let Some(u) = &url {
                println!("📖 实时笔记：{u}\n");
                let _ = Command::new("open").arg(u).status();
            }
        }
        if args.jsonl {
            println!(
                "{}",
                serde_json::json!({"type": "session", "dir": dir, "title": title, "url": url})
            );
        }
        let live = live::LiveView::new(
            dir.join(format!("{title}.live.json")),
            !args.jsonl,
            notes::wall_clock(started),
        );
        let overlay = if args.overlay {
            spawn_overlay(&dir.join(format!("{title}.live.json")))
        } else {
            None
        };
        // 实时声纹：每个语音段算声纹、在线聚类；字幕上先标 说话人A/B…，结束后再起名
        let speakers = if args.speakers {
            match online_spk::OnlineSpeakers::new(&models, &voices_dir(), 1) {
                Ok(s) => Some(Arc::new(Mutex::new(s))),
                Err(e) => {
                    eprintln!("  ! 无法区分说话人: {e:#}");
                    None
                }
            }
        } else {
            None
        };
        let spoken: Arc<Mutex<Vec<(String, String)>>> = Default::default();
        let mut live_opt = opt.clone();
        if speakers.is_some() {
            let spoken = spoken.clone();
            live_opt.labeler = Some(Arc::new(move |c: &mut Cue| {
                if let Some(s) = &c.speaker {
                    spoken.lock().unwrap().push((s.clone(), c.text()));
                }
            }));
        }
        let job = Job {
            input,
            stem: title.clone(),
            dir: dir.clone(),
            session,
            speakers: speakers.clone(),
        };
        let result = run_job(job, &asr, llm.as_ref(), &live_opt, Some(&live));
        if let Some(mut o) = overlay {
            let _ = o.kill();
        }
        result?;
        if let Some(spk) = speakers {
            let (summary, relabel) = spk.lock().unwrap().finish();
            if !summary.is_empty() {
                let info = diarize::SpeakerInfo {
                    turns: Vec::new(),
                    labels: summary.iter().map(|s| s.label.clone()).collect(),
                    known: summary.iter().map(|s| s.known).collect(),
                    profiles: summary.iter().map(|s| Some(s.profile.clone())).collect(),
                };
                let lines = std::mem::take(&mut *spoken.lock().unwrap());
                finish_speakers(
                    SpeakerResult {
                        info: &info,
                        secs: summary.iter().map(|s| s.secs).collect(),
                        lines,
                        relabel,
                    },
                    llm.as_ref(),
                    &dir,
                    &title,
                    hint.as_deref(),
                    args.save_voices,
                    args.jsonl,
                )?;
            }
        }
        return Ok(());
    }

    let mut failed = false;
    for raw in &args.inputs {
        eprintln!("▶ {raw}");
        let result = (|| -> Result<()> {
            let is_url = raw.starts_with("http://") || raw.starts_with("https://");
            let base = args.out_dir.clone();
            let (path, auto_title) = if is_url {
                let dl = base.clone().unwrap_or_else(default_out).join("downloads");
                download(raw, &dl, args.cookies.as_deref())?
            } else {
                let p = PathBuf::from(raw);
                if !p.is_file() {
                    bail!("文件不存在");
                }
                let stem = p
                    .file_stem()
                    .context("文件名无效")?
                    .to_string_lossy()
                    .into_owned();
                (p, stem)
            };
            let title = sanitize(match (&args.title, args.inputs.len()) {
                (Some(t), 1) => t,
                _ => &auto_title,
            });
            let (dir, session) = if want_notes {
                let dir = course
                    .as_ref()
                    .map_or_else(|| root.clone(), |c| c.dir.clone())
                    .join(&title);
                let status = Arc::new(Mutex::new(NotesStatus::default()));
                let s = Session::start(
                    notes_cfg(dir.clone(), title.clone(), notes::relative_clock(), &status),
                    llm.clone().unwrap(),
                )?;
                eprintln!("  📂 会话目录: {}", dir.display());
                let url = start_web(
                    &args,
                    web::WebState {
                        title: title.clone(),
                        notes_path: s.notes_path.clone(),
                        live_json: None,
                        notes: status,
                        started: chrono::Local::now(),
                        stopping: Arc::new(AtomicBool::new(false)),
                        live: false,
                    },
                );
                if args.jsonl {
                    println!(
                        "{}",
                        serde_json::json!({"type": "session", "dir": dir, "title": title, "url": url})
                    );
                }
                (dir, Some(s))
            } else {
                let dir = base
                    .or_else(|| path.parent().map(Path::to_path_buf))
                    .unwrap_or_default();
                (dir, None)
            };
            // 说话人识别：后台和识别同时跑，写字幕时再等它的结果
            let spk = args.speakers.then(|| {
                diarize::SpeakerJob::start(
                    path.clone(),
                    models.clone(),
                    4,
                    args.num_speakers,
                    voices_dir(),
                )
            });
            let mut job_opt = opt.clone();
            if let Some(j) = &spk {
                let j = j.clone();
                job_opt.labeler = Some(Arc::new(move |c: &mut Cue| j.label(c)));
            }
            let out_dir = dir.clone();
            run_job(
                Job {
                    input: Input::File(path),
                    dir,
                    stem: title.clone(),
                    session,
                    speakers: None,
                },
                &asr,
                llm.as_ref(),
                &job_opt,
                None,
            )?;
            if let Some(r) = spk.as_deref().and_then(offline_speakers) {
                finish_speakers(
                    r,
                    llm.as_ref(),
                    &out_dir,
                    &title,
                    hint.as_deref(),
                    args.save_voices,
                    args.jsonl,
                )?;
            }
            Ok(())
        })();
        if let Err(e) = result {
            eprintln!("✗ {raw}: {e:#}");
            failed = true;
        }
    }
    if failed {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn live_capture(sys: bool, mic: bool, wav: &Path, stop: Arc<AtomicBool>) -> Result<Input> {
    Ok(Input::Raw(Box::new(capture::Capture::start(
        sys,
        mic,
        Some(wav),
        stop,
    )?)))
}

#[cfg(not(target_os = "macos"))]
fn live_capture(_sys: bool, _mic: bool, _wav: &Path, _stop: Arc<AtomicBool>) -> Result<Input> {
    bail!("实时录音目前只支持 macOS；可以用其他工具输出 16kHz f32le 再 `subs --live -`")
}

/// 开实时笔记网页（--no-web 时不开）
fn start_web(args: &Args, state: web::WebState) -> Option<String> {
    if args.no_web {
        return None;
    }
    match web::serve(Arc::new(state)) {
        Ok(u) => {
            eprintln!("  📖 实时笔记网页：{u}");
            Some(u)
        }
        Err(e) => {
            eprintln!("  ! {e:#}");
            None
        }
    }
}

/// 启动悬浮字幕条（~/.local/bin/subs-overlay），找不到就跳过。
fn spawn_overlay(json: &Path) -> Option<std::process::Child> {
    let bin = home().join(".local/bin/subs-overlay");
    match Command::new(&bin)
        .arg(json)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("  ! 无法启动悬浮字幕条 {}: {e}", bin.display());
            None
        }
    }
}

/// 说话人收尾用到的信息（离线分离和实时声纹两种来源都整理成这个样子）
struct SpeakerResult<'a> {
    info: &'a diarize::SpeakerInfo,
    /// 每个说话人说了多少秒
    secs: Vec<f64>,
    /// 每条字幕 (说话人, 英文)
    lines: Vec<(String, String)>,
    /// 文件里已经写出的标签 → 最终占位名（实时模式结束时会合并、重新编号）
    relabel: std::collections::HashMap<String, String>,
}

/// 说话人收尾：推断名字、替换文件里的标签、写 speakers.json、（可选）存声纹
fn finish_speakers(
    r: SpeakerResult,
    llm: Option<&DeepSeek>,
    dir: &Path,
    stem: &str,
    hint: Option<&str>,
    save_voices: bool,
    jsonl: bool,
) -> Result<()> {
    let info = r.info;
    let lines: Vec<(String, String)> = r
        .lines
        .into_iter()
        .map(|(s, t)| (r.relabel.get(&s).cloned().unwrap_or(s), t))
        .collect();
    let names = match llm {
        Some(llm) => diarize::name_speakers(llm, info, &lines, stem, hint),
        None => Default::default(),
    };
    let final_name = |label: &str| {
        names
            .get(label)
            .cloned()
            .unwrap_or_else(|| label.to_owned())
    };
    let mut table = Vec::new();
    let mut entries = Vec::new();
    for (i, label) in info.labels.iter().enumerate() {
        let name = final_name(label);
        let secs = r.secs.get(i).copied().unwrap_or(0.0);
        let how = if info.known[i] {
            "声纹库"
        } else if names.contains_key(label) {
            "按内容推断"
        } else {
            "未知"
        };
        eprintln!("  🗣  {label} → {name}（{how}，约 {:.0} 分钟）", secs / 60.0);
        table.push(serde_json::json!({"label": label, "name": name, "source": how, "secs": secs}));
        entries.push(diarize::SpeakerEntry {
            name,
            source: how.into(),
            secs,
            embedding: info.profiles[i].clone(),
        });
    }
    // 文件里的标签一次性换成最终名字（同时替换，不会连锁）
    let mut renames: std::collections::HashMap<String, String> = names.clone();
    for (from, to) in &r.relabel {
        renames.insert(from.clone(), final_name(to));
    }
    renames.retain(|from, to| from != to);
    diarize::rename_in_dir(dir, stem, &renames)?;
    diarize::write_sidecar(dir, stem, &entries)?;
    if save_voices {
        for (name, path) in diarize::enroll_session(dir, stem, &voices_dir())? {
            eprintln!("  已存入声纹库：{name}（{}）", path.display());
        }
    } else {
        eprintln!(
            "  名字是自动推断的。核对无误（或用 --rename-speaker 改好）后，运行 subs --enroll \"{}\" 记住声纹",
            dir.display()
        );
    }
    if jsonl {
        println!(
            "{}",
            serde_json::json!({"type": "speakers", "speakers": table})
        );
    }
    Ok(())
}

/// 离线分离的结果 → 收尾信息
fn offline_speakers(job: &diarize::SpeakerJob) -> Option<SpeakerResult<'_>> {
    let info = job.info()?;
    let secs = (0..info.labels.len())
        .map(|i| {
            info.turns
                .iter()
                .filter(|t| t.speaker == i)
                .map(|t| t.end - t.start)
                .sum()
        })
        .collect();
    Some(SpeakerResult {
        info,
        secs,
        lines: job.lines.lock().unwrap().clone(),
        relabel: Default::default(),
    })
}

/// 跑一个任务：字幕流水线 + （可选）会话笔记 + （实时模式）滚动显示。
fn run_job(
    job: Job,
    asr: &asr::Asr,
    llm: Option<&DeepSeek>,
    opt: &Options,
    live: Option<&live::LiveView>,
) -> Result<()> {
    let t = Instant::now();
    let Job {
        input,
        dir,
        stem,
        mut session,
        speakers,
    } = job;
    let mut on_cue = |c: &Cue| -> Result<()> {
        if let Some(l) = live {
            l.cue(c);
        }
        match session.as_mut() {
            Some(s) => s.on_cue(c),
            None => Ok(()),
        }
    };
    let mut on_partial = |text: &str| {
        if let Some(l) = live {
            l.partial(text);
        }
        if opt.jsonl {
            println!("{}", serde_json::json!({"type": "partial", "text": text}));
        }
    };
    let partial: Option<&mut dyn FnMut(&str)> = if live.is_some() {
        Some(&mut on_partial)
    } else {
        None
    };
    let mut on_segment = |samples: &[f32]| speakers.as_ref()?.lock().unwrap().assign(samples);
    let hooks = pipeline::Hooks {
        on_cue: &mut on_cue,
        on_partial: partial,
        speaker_of: if speakers.is_some() {
            Some(&mut on_segment)
        } else {
            None
        },
    };
    let summary = pipeline::run(input, &dir, &stem, asr, llm, opt, hooks)?;
    let took = t.elapsed().as_secs_f64();
    eprintln!(
        "✓ {:.0}s 音频 → {} 条字幕，用时 {took:.0}s{}",
        summary.audio_secs,
        summary.cues,
        if opt.live {
            String::new()
        } else {
            format!("（{:.0}× 实时）", summary.audio_secs / took.max(0.1))
        }
    );
    if let Some(s) = session {
        let (notes, transcript) = (s.notes_path.clone(), s.transcript_path.clone());
        let minutes = s.finish()?;
        let html = dir.join(format!("{stem} 笔记.html"));
        if let Err(e) = web::export(&notes, &html, &stem) {
            eprintln!("  ! 导出网页失败: {e:#}");
        }
        eprintln!(
            "  → {}\n  → {}\n  → {}",
            transcript.display(),
            notes.display(),
            html.display()
        );
        if opt.jsonl {
            println!(
                "{}",
                serde_json::json!({"type": "done", "dir": dir, "notes": notes, "transcript": transcript, "minutes": minutes, "html": html})
            );
        }
        if let Some(m) = minutes {
            eprintln!("  → {}", m.display());
        }
    }
    Ok(())
}
