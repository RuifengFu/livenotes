# livenotes

英文讲座 / 会议 → **实时 AI 中文笔记** + 中英双语字幕。在 Mac 本地识别语音（Apple GPU），用 DeepSeek 翻译和整理笔记。

- **边听边看笔记**：上课时浏览器打开 `http://127.0.0.1:8765`，每隔几分钟自动追加一部分整理好的中文笔记（Markdown + 公式渲染），底部滚动最新字幕；下课后补一份课后总结，并导出可离线打开的 `笔记.html`
- **识别准**：默认用 [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) 在 Apple GPU 上跑（[MLX](https://github.com/ml-explore/mlx)），课程主题会作为术语提示交给识别模型；[Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) 在 CPU 上负责「正在说」的实时预览，也是 GPU 出问题时的兜底
- **字幕好读**：自动断句，每条不超过 5 秒，按阅读速度排时间轴；输出 `en` / `zh` / `zh-en` 三份 SRT
- **区分说话人**：声纹（CAM++）区分老师 / 学生、主持人 / 嘉宾，结束后根据内容推断名字；确认过的人存进本地声纹库，下次自动认出
- **按课程归档**：同一门课的每节课放在一起，自动沿用课程的术语提示，笔记会参考前两节的内容
- **多种输入**：麦克风（线下课）、电脑声音（Teams / Zoom，ScreenCaptureKit）、本地音视频、B 站 / YouTube 链接（yt-dlp）

## 结构

```
麦克风 / 系统声音 / 文件
   │
   ▼
Rust（subs）：采集 → GTCRN 降噪（只给 VAD）→ Silero VAD 切段 ──► Python 子进程：Qwen3-ASR + 强制对齐（MLX，GPU）
   │                                                         ◄── 带标点的词和时间
   ├─ 断句、排时间轴 → DeepSeek 翻译 → SRT
   ├─ 声纹在线聚类 → 说话人
   └─ 每 ~450 词 → DeepSeek 流式写笔记 → AI摘要.md → 本地网页
```

识别和音频处理用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（Silero VAD、GTCRN、Parakeet、CAM++、pyannote）和 [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr)。

## 安装

需要 Apple Silicon Mac（macOS 15+）、[Rust](https://rustup.rs)、[Homebrew](https://brew.sh)。

```bash
./install.sh
```

会下载模型（约 6GB：Qwen3-ASR 1.7B、对齐模型、Parakeet、SenseVoice、VAD、降噪），用 [uv](https://github.com/astral-sh/uv) 建 Python 环境，编译并安装 `~/.local/bin/subs`。

DeepSeek API key 写进 `~/.config/livenotes/env`：

```
DEEPSEEK_API_KEY=sk-...
```

## 用法

```bash
# 线下课：录麦克风，归入课程，区分说话人
subs --live --mic-only --style lecture --course "编译原理" --hint "LLVM, SSA, 寄存器分配" --speakers

# 线上会议：录电脑声音（Ctrl+C 结束后会补完笔记、生成会议纪要）
subs --live --title 组会

# 视频 / 链接 → 字幕 + 文字稿 + 笔记
subs "https://www.bilibili.com/video/BV..." --notes --speakers
subs lecture.mp4
```

`subs --help` 有完整说明。输出默认在 `~/Documents/LiveNotes/`：字幕、`文字稿.md`、`AI摘要.md`、`笔记.html`、`录音.wav`（实时模式）、`speakers.json`。

其他：

- `--engine parakeet`：只用 CPU（不装 Python 环境也能用）
- `--rename-speaker "说话人B=张老师" <会话目录>`：改说话人名字
- `--enroll <会话目录>`：把确认过名字的人存进声纹库
- `--html <会话目录>`：重新生成网页版笔记
- `--jsonl`：stdout 逐行输出 JSON 事件，方便给其他程序（如菜单栏 App）调用

## 隐私

语音识别、声纹都在本机完成，录音不会上传。字幕文本会发给 DeepSeek 做翻译和整理笔记。实时笔记网页只监听 `127.0.0.1`。
