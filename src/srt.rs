use crate::segment::Cue;

#[derive(Clone, Copy)]
pub enum Track {
    En,
    Zh,
    /// 中文在上、英文在下
    Bilingual,
}

fn timestamp(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// `label`：换人时在开头标出说话人（如 "【Elon Musk】"），同一个人连续说就不标
pub fn entry(n: usize, c: &Cue, track: Track, label: Option<&str>) -> String {
    let p = label.map(|l| format!("【{l}】")).unwrap_or_default();
    let en = format!("{p}{}", c.text());
    let body = match track {
        Track::En => en,
        Track::Zh => format!("{p}{}", c.zh),
        Track::Bilingual if c.zh.is_empty() => en,
        Track::Bilingual => format!("{p}{}\n{}", c.zh, c.text()),
    };
    format!(
        "{n}\n{} --> {}\n{body}\n\n",
        timestamp(c.start),
        timestamp(c.end)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::Word;

    #[test]
    fn formats_srt() {
        let cue = Cue {
            words: vec![
                Word::new("Hello", 3661.5, 3662.0),
                Word::new("world.", 3662.0, 3662.4),
            ],
            start: 3661.5,
            end: 3662.7345,
            zh: "你好 世界".into(),
            seg: 0,
            speaker: None,
        };
        assert_eq!(
            entry(1, &cue, Track::Bilingual, None),
            "1\n01:01:01,500 --> 01:01:02,735\n你好 世界\nHello world.\n\n"
        );
    }
}
