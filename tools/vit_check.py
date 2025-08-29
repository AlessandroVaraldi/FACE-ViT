#!/usr/bin/env python3
"""vit_check.py
Run a ViT ONNX model on a PPM image and print results in the same format as main.c.

Key points vs main.c:
- You can choose preprocessing with --preproc:
  * imagenet: (x/255 - mean)/std  (default; MEAN/STD from YAML or ImageNet)
  * c_like:   (x - 128)/128       (closer to main.c which does uint8-128 -> int8)
  * none:     x/255               (no mean/std)
- It trusts the ONNX model's output dimension (ignores OUT_DIM in YAML if different).
- Softmax is float, then quantized to int [0..127] with a final correction so the sum==127,
  mirroring kernel_softmax.c behavior.

Usage:
  python vit_check.py --onnx vit_model.onnx --config vit_config.yaml --ppm image.ppm [--labels labels.txt] [--preproc imagenet|c_like|none]
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    print("ERROR: onnxruntime is required. Install with: pip install onnxruntime", file=sys.stderr)
    raise

try:
    import yaml
except Exception:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


def parse_args():
    p = argparse.ArgumentParser(description="Run ViT ONNX on a PPM and print C-like output.")
    p.add_argument("--onnx", required=True, type=Path, help="Path to vit_model.onnx")
    p.add_argument("--config", required=True, type=Path, help="Path to vit_config.yaml")
    p.add_argument("--ppm", required=True, type=Path, help="Path to input image (.ppm)")
    p.add_argument("--labels", type=Path, default=None, help="Optional labels file (one label per line)")
    p.add_argument("--preproc", choices=["imagenet","c_like","none"], default="imagenet",
                   help="Preprocessing: imagenet=(x/255 - mean)/std; c_like=(x-128)/128; none=x/255")
    return p.parse_args()


# --- Simple PPM loader (P6 binary and P3 ASCII) ---
def load_ppm(path: Path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in [b'P6', b'P3']:
            raise ValueError(f"Unsupported PPM format: {magic!r}. Only P6 (binary) and P3 (ASCII) supported.")

        def _read_token():
            token = b""
            c = f.read(1)
            # consume whitespace/comments
            while c:
                if c in b" \t\r\n":
                    c = f.read(1); continue
                if c == b'#':
                    # skip until end of line
                    while c and c not in b"\r\n":
                        c = f.read(1)
                    c = f.read(1); continue
                break
            # read token until whitespace
            while c and c not in b" \t\r\n":
                token += c
                c = f.read(1)
            return token

        width = int(_read_token()); height = int(_read_token()); maxval = int(_read_token())
        if maxval <= 0 or maxval > 65535:
            raise ValueError(f"Invalid maxval in PPM: {maxval}")

        if magic == b'P6':
            depth = 2 if maxval > 255 else 1
            expected = width * height * 3 * depth
            data = f.read(expected)
            if len(data) != expected:
                raise ValueError(f"PPM data truncated: expected {expected} bytes, got {len(data)}")
            if depth == 1:
                arr = np.frombuffer(data, dtype=np.uint8)
            else:
                arr = np.frombuffer(data, dtype=">u2")
                arr = (arr.astype(np.uint32) * 255 + (maxval // 2)) // maxval
                arr = arr.astype(np.uint8)
        else:
            text = f.read().split()
            vals = list(map(int, text))
            arr = np.array(vals, dtype=np.int32)
            if arr.size != width * height * 3:
                raise ValueError("Invalid P3 PPM data size")
            if maxval != 255:
                arr = (arr.astype(np.uint32) * 255 + (maxval // 2)) // maxval
            arr = arr.astype(np.uint8)

        return arr.reshape((height, width, 3))


def maybe_resize(img_hw3: np.ndarray, size_hw):
    H, W, _ = img_hw3.shape
    target_h, target_w = size_hw
    if (H, W) == (target_h, target_w):
        return img_hw3
    if PIL_OK:
        return np.asarray(Image.fromarray(img_hw3).resize((target_w, target_h), Image.BILINEAR))
    # nearest neighbor fallback
    y_idx = (np.linspace(0, H - 1, target_h)).round().astype(int)
    x_idx = (np.linspace(0, W - 1, target_w)).round().astype(int)
    return img_hw3[y_idx][:, x_idx]


def infer_input_size(session):
    inp = session.get_inputs()[0]
    shape = inp.shape
    resolved = [d if isinstance(d, int) else 1 for d in shape]
    if len(resolved) != 4:
        raise ValueError(f"Unexpected ONNX input rank: {shape}")
    nchw = True
    if resolved[1] not in (1, 3) and resolved[-1] in (1, 3):
        nchw = False
    if nchw:
        C, H, W = resolved[1], resolved[2], resolved[3]
        layout = "NCHW"
    else:
        H, W, C = resolved[1], resolved[2], resolved[3]
        layout = "NHWC"
    return (H, W, C, layout, inp.name)


def load_labels(labels_path: Path, out_dim: int):
    if labels_path is None or not labels_path.exists():
        return [f"class_{i}" for i in range(out_dim)]
    labels = [line.rstrip("\n") for line in labels_path.read_text(encoding="utf-8").splitlines()]
    if len(labels) != out_dim:
        print(f"WARNING: labels length ({len(labels)}) != OUT_DIM ({out_dim}); using generic names.", file=sys.stderr)
        return [f"class_{i}" for i in range(out_dim)]
    return labels


def softmax(x: np.ndarray, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x, dtype=np.float64)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def quantize_softmax_to_int8(probs: np.ndarray) -> np.ndarray:
    """Quantize probs to int [0..127] and enforce sum==127 (like main.c)."""
    q = np.round(probs * 127.0).astype(np.int32)
    q = np.clip(q, 0, 127)
    diff = 127 - int(q.sum())
    if diff != 0:
        j = len(q) - 1
        v = int(q[j]) + diff
        if v < 0: v = 0
        if v > 127: v = 127
        q[j] = v
    return q.astype(np.int32)


def main():
    args = parse_args()

    # Load config (for MEAN/STD defaults)
    cfg = yaml.safe_load(args.config.read_text())
    MEAN = cfg.get("MEAN", [0.485, 0.456, 0.406])  # ImageNet defaults
    STD  = cfg.get("STD",  [0.229, 0.224, 0.225])

    # Load ONNX and infer input geometry
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    H, W, C, layout, inp_name = infer_input_size(sess)
    if C != 3:
        raise ValueError(f"Model expects {C} channels, but script assumes 3-channel RGB input.")

    # Load & prepare image
    img = load_ppm(args.ppm)  # HxWx3 uint8
    img = maybe_resize(img, (H, W))
    img_u8 = img.astype(np.float32)

    if args.preproc == "imagenet":
        img = img_u8 / 255.0
        mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3)
        std  = np.array(STD, dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
    elif args.preproc == "c_like":
        img = (img_u8 - 128.0) / 128.0
    else:  # none
        img = img_u8 / 255.0

    if layout == "NCHW":
        x = np.transpose(img, (2, 0, 1))[None, ...]  # 1x3xHxW
    else:
        x = img[None, ...]  # 1xHxWx3

    # Forward
    outputs = sess.run(None, {inp_name: x})
    logits = np.array(outputs[0])
    if logits.ndim > 2:
        logits = logits.reshape(logits.shape[0], -1)
    if logits.shape[0] != 1:
        print(f"WARNING: batch size is {logits.shape[0]}, using the first sample only.", file=sys.stderr)
    logits = logits[0]
    OUT_DIM = logits.shape[-1]

    # Softmax -> int8 probs (sum==127 like C)
    probs = softmax(logits, axis=-1)
    out_q = quantize_softmax_to_int8(probs)

    # Labels
    labels = load_labels(args.labels, OUT_DIM)

    # Print exactly like main.c: "<index> <class_name>: <int>"
    for i in range(OUT_DIM):
        print(f"{i} {labels[i]}: {int(out_q[i])}")

    # top-5 (stderr)
    top5_idx = np.argsort(-probs)[:5]
    msg = "TOP-5: " + ", ".join([f"#{int(i)} {labels[int(i)]} ({probs[int(i)]:.4f})" for i in top5_idx])
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()

# python tools/vit_check.py --onnx models/vit_model_finetune.onnx --config models/vit_config_finetune.yaml --ppm images/shark32.ppm --preproc c_like