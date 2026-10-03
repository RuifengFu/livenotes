//! 实时笔记网页：边录边在浏览器里看 AI 笔记（Markdown 渲染 + KaTeX 公式），底部是最新字幕。
//!
//! 只监听 127.0.0.1。页面每 1.5s 轮询 /api/state：笔记文件有变化才重新渲染。
//! 结束时另存一份静态 `<标题> 笔记.html`，以后直接双击打开。

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use pulldown_cmark::{Options, Parser, html};
use serde_json::{Value, json};

use crate::notes::NotesStatus;

const PREFERRED_PORT: u16 = 8765;

pub struct WebState {
    pub title: String,
    pub notes_path: PathBuf,
    pub live_json: Option<PathBuf>,
    pub notes: Arc<Mutex<NotesStatus>>,
    pub started: chrono::DateTime<chrono::Local>,
    /// 已停止录音（正在收尾）
    pub stopping: Arc<AtomicBool>,
    pub live: bool,
}

/// 启动网页服务（后台线程），返回网址
pub fn serve(state: Arc<WebState>) -> Result<String> {
    let server = tiny_http::Server::http(("127.0.0.1", PREFERRED_PORT))
        .or_else(|_| tiny_http::Server::http(("127.0.0.1", 0)))
        .map_err(|e| anyhow::anyhow!("无法启动网页服务: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .map(|a| a.port())
        .unwrap_or(PREFERRED_PORT);
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let url = req.url().to_owned();
            let (path, query) = url.split_once('?').unwrap_or((&url, ""));
            let resp = match path {
                "/" => html_response(page(&state.title, None)),
                "/api/state" => {
                    let since = query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix("v="))
                        .and_then(|v| v.parse::<u64>().ok());
                    json_response(api_state(&state, since))
                }
                _ => tiny_http::Response::from_string("not found").with_status_code(404),
            };
            let _ = req.respond(resp);
        }
    });
    Ok(format!("http://127.0.0.1:{port}/"))
}

fn html_response(body: String) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body).with_header(
        "Content-Type: text/html; charset=utf-8"
            .parse::<tiny_http::Header>()
            .unwrap(),
    )
}

fn json_response(v: Value) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(v.to_string()).with_header(
        "Content-Type: application/json; charset=utf-8"
            .parse::<tiny_http::Header>()
            .unwrap(),
    )
}

fn read_string(p: &Path) -> String {
    let mut s = String::new();
    if let Ok(mut f) = std::fs::File::open(p) {
        let _ = f.read_to_string(&mut s);
    }
    s
}

fn api_state(st: &WebState, since: Option<u64>) -> Value {
    let md = read_string(&st.notes_path);
    // 笔记文件只追加（失败重试时会截掉重写），用内容长度 + 简单哈希当版本号
    let version =
        md.len() as u64 * 1_000_003 + md.bytes().rev().take(64).map(u64::from).sum::<u64>();
    let live = st
        .live_json
        .as_deref()
        .and_then(|p| serde_json::from_str::<Value>(&read_string(p)).ok())
        .unwrap_or(Value::Null);
    let notes = st.notes.lock().unwrap().clone();
    let phase = if st.stopping.load(Ordering::SeqCst) {
        "finishing"
    } else if st.live {
        "recording"
    } else {
        "processing"
    };
    let mut v = json!({
        "v": version,
        "phase": phase,
        "elapsed": (chrono::Local::now() - st.started).num_seconds(),
        "notes": notes,
        "live": live,
    });
    if since != Some(version) {
        v["html"] = Value::String(render_markdown(&strip_title(&md)));
    }
    v
}

/// 去掉笔记开头的「# 标题」和说明行（页面顶栏已有标题）
fn strip_title(md: &str) -> String {
    let mut lines = md.lines().peekable();
    if lines.peek().is_some_and(|l| l.starts_with("# ")) {
        lines.next();
    }
    let rest: Vec<&str> = lines
        .skip_while(|l| l.trim().is_empty() || l.starts_with("> 由 AI"))
        .collect();
    rest.join("\n")
}

/// 把 \( \) / \[ \] 公式改成 $ / $$，交给 pulldown-cmark 的数学扩展（否则反斜杠会被当成转义吃掉）
fn normalize_math(md: &str) -> String {
    let mut out = String::with_capacity(md.len());
    let mut in_code = false;
    for line in md.split_inclusive('\n') {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
        }
        if in_code {
            out.push_str(line);
        } else {
            let line = line
                .replace("\\[", "$$")
                .replace("\\]", "$$")
                .replace("\\(", "$")
                .replace("\\)", "$");
            out.push_str(&strong_cjk(&line));
        }
    }
    out
}

/// `**加粗（English）**后面紧跟中文` 按 CommonMark 规则不算加粗（右括号是标点、后面又不是空格），
/// 中文笔记里很常见。成对的 `**` 直接换成 <strong>。
fn strong_cjk(line: &str) -> String {
    let parts: Vec<&str> = line.split("**").collect();
    // 奇数段 = 偶数个 **；不成对的保持原样；含反引号的行（行内代码）不动
    if parts.len() < 3 || parts.len().is_multiple_of(2) || line.contains('`') {
        return line.to_owned();
    }
    let mut out = String::with_capacity(line.len() + 16);
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str(if i % 2 == 1 { "<strong>" } else { "</strong>" });
        }
        out.push_str(p);
    }
    out
}

pub fn render_markdown(md: &str) -> String {
    let opts = Options::ENABLE_TABLES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_MATH
        | Options::ENABLE_TASKLISTS;
    let md = normalize_math(md);
    let mut out = String::new();
    html::push_html(&mut out, Parser::new_ext(&md, opts));
    out
}

/// 结束后另存一份静态网页（公式用 CDN 上的 KaTeX 渲染）
pub fn export(notes: &Path, out: &Path, title: &str) -> Result<()> {
    let body = render_markdown(&strip_title(&read_string(notes)));
    std::fs::write(out, page(title, Some(&body)))?;
    Ok(())
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `body` 为 None：实时页面（轮询）；Some：静态导出
fn page(title: &str, body: Option<&str>) -> String {
    PAGE.replace("{{TITLE}}", &escape(title))
        .replace("{{LIVE}}", if body.is_none() { "true" } else { "false" })
        .replace("{{BODY}}", body.unwrap_or(""))
}

const PAGE: &str = include_str!("web_page.html");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_latex_delimiters() {
        let html = render_markdown("设 \\(f : D \\to E\\) 连续：\n\n\\[ f(\\bigsqcup x) \\]\n");
        assert!(html.contains("math-inline"), "{html}");
        assert!(html.contains("math-display"), "{html}");
        assert!(html.contains("D \\to E"), "{html}");
    }

    #[test]
    fn bold_before_chinese() {
        let html = render_markdown("与**并发（concurrency）**相关的组件\n");
        assert!(
            html.contains("<strong>并发（concurrency）</strong>相关"),
            "{html}"
        );
        assert!(render_markdown("`a**b`\n").contains("a**b"));
    }

    #[test]
    fn strips_title_block() {
        let md = "# 标题\n\n> 由 AI 根据讲座实时转写生成\n\n\n---\n\n## 内容\n";
        assert_eq!(strip_title(md), "---\n\n## 内容");
    }

    #[test]
    fn code_blocks_keep_backslashes() {
        let md = "```\nlet x = \"\\(\";\n```\n";
        assert!(normalize_math(md).contains("\\("));
    }
}
