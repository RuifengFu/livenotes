//! 实时显示：终端里滚动显示字幕（正在说的英文在最后一行原地刷新），
//! 同时写 `<标题>.live.json` 给悬浮条（subs-overlay）读。

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::json;

use crate::notes::Clock;
use crate::segment::Cue;

struct State {
    finals: VecDeque<(String, String, f64, Option<String>)>,
    partial: String,
    partial_shown: bool,
}

pub struct LiveView {
    state: Mutex<State>,
    json: PathBuf,
    echo: bool,
    clock: Clock,
}

impl LiveView {
    pub fn new(json: PathBuf, echo: bool, clock: Clock) -> Self {
        let view = Self {
            state: Mutex::new(State {
                finals: VecDeque::new(),
                partial: String::new(),
                partial_shown: false,
            }),
            json,
            echo,
            clock,
        };
        view.write(&view.state.lock().unwrap());
        view
    }

    /// 正在说的这段（还没断句翻译）
    pub fn partial(&self, text: &str) {
        let mut st = self.state.lock().unwrap();
        if st.partial == text {
            return;
        }
        st.partial = text.to_owned();
        if self.echo {
            // 只显示最后一截，保证一行放得下
            let chars: Vec<char> = text.chars().collect();
            let tail: String = chars[chars.len().saturating_sub(110)..].iter().collect();
            let mut out = std::io::stdout().lock();
            let _ = write!(out, "\r\x1b[2K          \x1b[3;2m{tail}\x1b[0m");
            let _ = out.flush();
            st.partial_shown = true;
        }
        self.write(&st);
    }

    /// 一条正式字幕（已断句、已翻译）
    pub fn cue(&self, c: &Cue) {
        let mut st = self.state.lock().unwrap();
        let en = c.text();
        st.finals
            .push_back((c.zh.clone(), en.clone(), c.start, c.speaker.clone()));
        while st.finals.len() > 3 {
            st.finals.pop_front();
        }
        // 这条已经出正式结果，就不再显示同样内容的"正在说"
        if st
            .partial
            .starts_with(en.split_whitespace().next().unwrap_or("\u{0}"))
        {
            st.partial.clear();
        }
        if self.echo {
            let mut out = std::io::stdout().lock();
            if st.partial_shown {
                let _ = write!(out, "\r\x1b[2K");
                st.partial_shown = false;
            }
            let _ = writeln!(
                out,
                "\x1b[2m{}\x1b[0m  {}\n          \x1b[2m{}\x1b[0m",
                (self.clock)(c.start),
                c.zh,
                en
            );
            let _ = out.flush();
        }
        self.write(&st);
    }

    fn write(&self, st: &State) {
        let finals: Vec<_> = st
            .finals
            .iter()
            .map(|(zh, en, t, who)| json!({"zh": zh, "en": en, "t": t, "speaker": who}))
            .collect();
        let v = json!({"finals": finals, "partial": st.partial});
        // 先写临时文件再改名，悬浮条不会读到半截
        let tmp = self.json.with_extension("json.tmp");
        if std::fs::write(&tmp, v.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, &self.json);
        }
    }
}
