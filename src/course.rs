//! 课程：同一门课的每节课放在 `<输出目录>/<课程>/<课程_日期_时间>/` 下。
//!
//! 课程目录里的 `课程.json` 记下课程名和主题提示（给翻译和笔记用），
//! 写笔记时还会把前两节课笔记的标题交给模型，帮助理解术语、前后衔接。

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const INFO_FILE: &str = "课程.json";

#[derive(Serialize, Deserialize, Default, Debug, PartialEq)]
pub struct CourseInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

pub struct Course {
    pub dir: PathBuf,
    pub info: CourseInfo,
}

impl Course {
    /// 打开（没有就新建）课程目录；给了非空的 hint 就更新课程的主题提示
    pub fn open(root: &Path, dir_name: &str, name: &str, hint: Option<&str>) -> Result<Self> {
        let dir = root.join(dir_name);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(INFO_FILE);
        let mut info: CourseInfo = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_else(|| CourseInfo {
                name: name.to_owned(),
                hint: None,
            });
        let hint = hint.map(str::trim).filter(|h| !h.is_empty());
        if let Some(h) = hint {
            info.hint = Some(h.to_owned());
        }
        if hint.is_some() || !path.exists() {
            std::fs::write(&path, serde_json::to_string_pretty(&info)?)?;
        }
        Ok(Self { dir, info })
    }

    /// 前几节课（按目录名排序，不含 `current`）笔记里的标题
    pub fn previous_topics(&self, current: &Path, sessions: usize) -> Option<String> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&self.dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p != current)
            .collect();
        dirs.sort();
        let mut out = Vec::new();
        for d in dirs.iter().rev().take(sessions).rev() {
            let Some(notes) = std::fs::read_dir(d)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .find(|p| p.to_string_lossy().ends_with("AI摘要.md"))
            else {
                continue;
            };
            let text = std::fs::read_to_string(notes).unwrap_or_default();
            let heads = headings(&text);
            if !heads.is_empty() {
                let name = d.file_name().unwrap_or_default().to_string_lossy();
                out.push(format!("【{name}】\n{}", heads.join("\n")));
            }
        }
        (!out.is_empty()).then(|| out.join("\n\n"))
    }
}

/// 笔记里的一、二级标题（跳过文件标题和课后总结），最多 30 个
fn headings(md: &str) -> Vec<String> {
    let mut out = Vec::new();
    for l in md.lines().skip(1) {
        if l.starts_with("# 课后总结") {
            break;
        }
        if l.starts_with("# ") || l.starts_with("## ") {
            out.push(l.trim_start_matches('#').trim().to_owned());
        }
    }
    out.truncate(30);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_course_and_reads_previous_topics() {
        let root = std::env::temp_dir().join(format!("subs-course-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let c = Course::open(&root, "Haskell", "Haskell", Some("程序语义学")).unwrap();
        assert_eq!(c.info.hint.as_deref(), Some("程序语义学"));
        // 再次打开不给提示：沿用已存的
        let c = Course::open(&root, "Haskell", "Haskell", None).unwrap();
        assert_eq!(c.info.hint.as_deref(), Some("程序语义学"));

        let s1 = c.dir.join("Haskell_2026-10-01_1600");
        std::fs::create_dir_all(&s1).unwrap();
        std::fs::write(
            s1.join("Haskell_2026-10-01_1600 AI摘要.md"),
            "# Haskell_2026-10-01_1600\n\n# 语义学导论\n## 操作语义\n### 小节\n# 课后总结\n## 核心要点\n",
        )
        .unwrap();
        let s2 = c.dir.join("Haskell_2026-10-08_1600");
        std::fs::create_dir_all(&s2).unwrap();
        let ctx = c.previous_topics(&s2, 2).unwrap();
        assert!(ctx.contains("【Haskell_2026-10-01_1600】"));
        assert!(ctx.contains("操作语义"));
        assert!(!ctx.contains("小节") && !ctx.contains("核心要点"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
