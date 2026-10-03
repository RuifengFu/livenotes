#!/bin/bash
# 下载模型、编译、安装 subs 和 subs-overlay 到 ~/.local/bin
set -euo pipefail
cd "$(dirname "$0")"

MODELS="${SUBS_MODELS:-$HOME/.local/share/livenotes/models}"
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
PK=sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8
SV=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

mkdir -p "$MODELS"
[ -f "$MODELS/silero_vad.onnx" ] || curl -fSL -o "$MODELS/silero_vad.onnx" "$BASE/silero_vad.onnx"
# GTCRN 降噪（只给 VAD 用，0.5MB）：远处小声、嘈杂教室里切段更准
[ -f "$MODELS/gtcrn_simple.onnx" ] || curl -fSL -o "$MODELS/gtcrn_simple.onnx" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
# Parakeet：CPU 上的快速识别，实时模式用它显示「正在说」；--engine parakeet 时也用它出正式字幕
[ -f "$MODELS/$PK/encoder.int8.onnx" ] || curl -fSL "$BASE/$PK.tar.bz2" | tar xj -C "$MODELS"
# 可选引擎：SenseVoice（--engine sensevoice）
if [ ! -f "$MODELS/$SV/model.onnx" ]; then
  curl -fSL "$BASE/$SV.tar.bz2" | tar xj -C "$MODELS"
  rm -f "$MODELS/$SV/model.int8.onnx" # 用 fp32 版，准确率更好
fi

# 默认识别引擎：Qwen3-ASR-1.7B，在 Apple GPU 上跑（MLX，Python 子进程，uv 管理）
DATA="$HOME/.local/share/livenotes"
command -v uv >/dev/null || brew install uv
[ -x "$DATA/qwen-venv/bin/python" ] || uv venv -q -p 3.12 "$DATA/qwen-venv"
uv pip install -q -p "$DATA/qwen-venv/bin/python" -r qwen/pyproject.toml
install -m 644 qwen/qwen_server.py "$DATA/qwen_server.py"
# 预先下载模型（约 4.5GB，在 ~/.cache/huggingface）
"$DATA/qwen-venv/bin/python" -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ('Qwen/Qwen3-ASR-1.7B', 'Qwen/Qwen3-ForcedAligner-0.6B')]"

command -v ffmpeg >/dev/null || brew install ffmpeg
command -v yt-dlp >/dev/null || brew install yt-dlp

source "$HOME/.cargo/env"
cargo build --release
mkdir -p "$HOME/.local/bin"
install -m 755 target/release/subs "$HOME/.local/bin/subs"
swiftc -O overlay/SubsOverlay.swift -o target/subs-overlay
install -m 755 target/subs-overlay "$HOME/.local/bin/subs-overlay"
echo "已安装 $HOME/.local/bin/subs 和 subs-overlay"

mkdir -p "$HOME/.config/livenotes"
[ -f "$HOME/.config/livenotes/env" ] ||
  echo "提示：DeepSeek key 可以写进 ~/.config/livenotes/env：DEEPSEEK_API_KEY=sk-..."
