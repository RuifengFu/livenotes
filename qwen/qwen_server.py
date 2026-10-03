"""Qwen3-ASR 识别服务：在 Apple GPU（MLX）上跑，给 Rust 的 subs 当子进程用。

协议（stdin / stdout）：
- 启动完成后输出一行 {"ready": true, "model": ...}
- 请求：一行 JSON 头 {"id": 1, "n": 样本数, "context": "术语提示", "max_tokens": 120}，
  紧接着 n 个 16kHz 单声道 float32 小端样本
- 回复：一行 JSON {"id": 1, "text": "...", "words": [["Hello,", 0.12, 0.40], ...], "truncated": false}
  出错时 {"id": 1, "error": "..."}
日志只写 stderr。
"""

import json
import os
import re
import sys
import time

import numpy as np

MODEL = os.environ.get("SUBS_QWEN_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER = os.environ.get("SUBS_QWEN_ALIGNER", "Qwen/Qwen3-ForcedAligner-0.6B")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def bare(s: str) -> str:
    return re.sub(r"[^\w']", "", s).lower()


def strip_context_leak(text: str, context: str, min_run: int = 4) -> str:
    """安静或听不清的段落里，模型偶尔会把术语提示原样念出来。
    识别结果里连续 min_run 个以上的词和提示一字不差，就删掉这一串。"""
    ctx = [bare(w) for w in context.split()]
    ctx = [w for w in ctx if w]
    toks = text.split()
    if len(ctx) < min_run or len(toks) < min_run:
        return text
    b = [bare(t) for t in toks]
    drop = set()
    i = 0
    while i < len(toks):
        best = 0
        for j in range(len(ctx)):
            k = 0
            while i + k < len(b) and j + k < len(ctx) and b[i + k] == ctx[j + k]:
                k += 1
            best = max(best, k)
        if best >= min_run:
            drop.update(range(i, i + best))
            i += best
        else:
            i += 1
    return " ".join(t for n, t in enumerate(toks) if n not in drop)


def attach_punctuation(text: str, aligned) -> list:
    """对齐器给出的词不带标点；按顺序对回识别文本里带标点的词，拿到它们的时间。"""
    tokens = text.split()
    out = []
    j = 0
    for tok in tokens:
        b = bare(tok)
        if not b:
            # 纯标点：贴到上一个词后面
            if out:
                out[-1][0] += tok
            continue
        # 在附近找同一个词（对齐器偶尔拆词/合词，找不到就沿用前后时间）
        k = next((i for i in range(j, min(j + 4, len(aligned))) if bare(aligned[i].text) == b), None)
        if k is None and j < len(aligned) and b.startswith(bare(aligned[j].text)):
            k = j
        if k is not None:
            w = aligned[k]
            out.append([tok, float(w.start_time), float(w.end_time)])
            j = k + 1
        else:
            t = out[-1][2] if out else (float(aligned[j].start_time) if j < len(aligned) else 0.0)
            out.append([tok, t, t])
    # 保证时间单调，终点不早于起点
    last = 0.0
    for w in out:
        w[1] = max(w[1], last)
        w[2] = max(w[2], w[1])
        last = w[1]
    return out


def main():
    import mlx.core as mx  # noqa: F401  确认 MLX / Metal 可用
    from mlx_qwen3_asr import ForcedAligner, Session

    t = time.time()
    session = Session(MODEL)
    aligner = ForcedAligner(ALIGNER)
    aligner._ensure_loaded()
    # 预热一次，避免第一段特别慢
    session.transcribe(np.zeros(16000, dtype=np.float32), language="English", max_new_tokens=8)
    log(f"qwen: {MODEL} 加载 {time.time() - t:.1f}s")
    out = sys.stdout
    out.write(json.dumps({"ready": True, "model": MODEL}) + "\n")
    out.flush()

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        req = json.loads(line)
        n = int(req["n"])
        raw = stdin.read(n * 4)
        if len(raw) < n * 4:
            break
        audio = np.frombuffer(raw, dtype="<f4").copy()
        try:
            r = session.transcribe(
                audio,
                language="English",
                context=req.get("context") or "",
                max_new_tokens=int(req.get("max_tokens") or 32 + n // 2000),
            )
            text = strip_context_leak(r.text.strip(), req.get("context") or "")
            words = attach_punctuation(text, aligner.align(audio, text, "English")) if text else []
            resp = {"id": req["id"], "text": text, "words": words, "truncated": bool(r.truncated)}
        except Exception as e:  # 单段失败不影响后面
            resp = {"id": req["id"], "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
