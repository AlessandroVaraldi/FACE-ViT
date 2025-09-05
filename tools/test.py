#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live webcam tester for TinyViTDet (anchor-free per patch).

- Loads model hyper-params from the export JSON (MODEL: img_size, patch, dim, layers, heads, dff).
- Loads weights from a training checkpoint .pth (by default model base; --use-ema to load EMA).
- Preprocess matches validation: Resize(shorter_side=int(img*256/224)) + CenterCrop(img_size) + Normalize.
- Postprocess: sigmoid on p_logits, threshold + NMS, draw boxes on either canvas or original frame.

Keys:
  [ / ]     decrease/increase confidence threshold by 0.02
  ; / '     decrease/increase NMS IoU by 0.02
  o         toggle drawing on original vs canvas
  s         save current frame to disk
  q/ESC     quit

Example:
  python tools/webcam_test.py \
    --config checkpoints/tinyvitdet_faces_export.json \
    --weights checkpoints/best_tinyvitdet_faces.pth \
    --use-ema --device cuda --amp bf16 --view original
"""

from __future__ import annotations
import argparse, json, time, math, os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---- import TinyViTDet from your utilities ----
from train_utils import TinyViTDet  # ensure PYTHONPATH includes repo root

IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------------- Geometry / NMS -------------------------
def boxes_cxcywh_to_xyxy(b: torch.Tensor, img_size: int) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    x1 = (cx - 0.5 * w) * img_size
    y1 = (cy - 0.5 * h) * img_size
    x2 = (cx + 0.5 * w) * img_size
    y2 = (cy + 0.5 * h) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union

def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    keep = []
    idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0]
        keep.append(i.item())
        if idx.numel() == 1:
            break
        cur = boxes[i].unsqueeze(0)
        rest = boxes[idx[1:]]
        iou = box_iou_xyxy(cur, rest).squeeze(0)
        idx = idx[1:][iou <= iou_thr]
    return torch.as_tensor(keep, device=boxes.device, dtype=torch.long)

# ------------------------- Pre/Post utils -------------------------
def resize_and_center_crop_rgb(rgb: np.ndarray, img_size: int):
    """
    rgb: HxWx3 (uint8), RGB
    Returns:
      canvas_rgb: img_size x img_size x 3 (uint8)
      meta: (W0,H0,new_w,new_h,left,top,scale)
    """
    H0, W0 = rgb.shape[:2]
    target_short = int(round(img_size * 256 / 224))  # match validation path
    if H0 <= W0:
        scale = target_short / max(H0, 1e-6)
        new_h = target_short
        new_w = int(round(W0 * scale))
    else:
        scale = target_short / max(W0, 1e-6)
        new_w = target_short
        new_h = int(round(H0 * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    top  = max(0, (new_h - img_size) // 2)
    left = max(0, (new_w - img_size) // 2)
    crop = resized[top:top + img_size, left:left + img_size].copy()
    return crop, (W0, H0, new_w, new_h, left, top, scale)

def normalize_to_tensor(rgb_canvas: np.ndarray, device: torch.device) -> torch.Tensor:
    # HWC uint8 -> float32 0..1 -> normalize -> CHW 1x3xHxW
    arr = rgb_canvas.astype(np.float32) / 255.0
    arr = (arr - IMNET_MEAN) / IMNET_STD
    chw = np.transpose(arr, (2, 0, 1)).copy()
    t = torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=torch.float32)
    return t

def map_canvas_boxes_to_original(boxes_canvas: np.ndarray, meta):
    """boxes_canvas: Nx4 in pixel coords of canvas (img_size), meta as above; returns xyxy in original frame space."""
    W0, H0, new_w, new_h, left, top, scale = meta
    b = boxes_canvas.astype(np.float32).copy()
    # back to resized space
    b[:, [0, 2]] = b[:, [0, 2]] + left
    b[:, [1, 3]] = b[:, [1, 3]] + top
    # back to original space
    inv = 1.0 / max(scale, 1e-9)
    b[:, [0, 2]] = b[:, [0, 2]] * inv
    b[:, [1, 3]] = b[:, [1, 3]] * inv
    # clamp
    b[:, 0] = np.clip(b[:, 0], 0, W0 - 1)
    b[:, 2] = np.clip(b[:, 2], 0, W0 - 1)
    b[:, 1] = np.clip(b[:, 1], 0, H0 - 1)
    b[:, 3] = np.clip(b[:, 3], 0, H0 - 1)
    return b

# ------------------------- Checkpoint loading -------------------------
def _strip_prefix(k: str) -> str:
    for pref in ("module._orig_mod.", "_orig_mod.", "module."):
        if k.startswith(pref):
            return k[len(pref):]
    return k

def load_model(config_path: str, weights_path: str, use_ema: bool, device: torch.device, amp: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    mcfg = cfg["MODEL"]
    img_size = int(mcfg["img_size"])
    model = TinyViTDet(
        img=img_size, patch=int(mcfg["patch"]), d=int(mcfg["dim"]),
        depth=int(mcfg["layers"]), heads=int(mcfg["heads"]), dff=int(mcfg["dff"]),
        dp=0.0, do=0.0
    ).to(device).eval()

    ckpt = torch.load(weights_path, map_location="cpu")
    # prefer EMA if requested and present
    sd = ckpt.get("ema_state_dict") if (use_ema and "ema_state_dict" in ckpt) else ckpt.get("state_dict", ckpt)
    cleaned = { _strip_prefix(k): v for k, v in sd.items() if _strip_prefix(k) in model.state_dict() }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    # runtime postproc defaults from JSON
    post = cfg.get("POST", {})
    conf_thr = float(post.get("conf_thr", 0.25))
    nms_iou  = float(post.get("nms_iou", 0.50))

    amp_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if amp == "fp16" else None)
    return model, img_size, conf_thr, nms_iou, amp_dtype

# ------------------------- Drawing -------------------------
def draw_boxes(img_bgr: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, conf_thr: float):
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int).tolist()
        s = float(scores[i])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{s:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, label, (x1 + 2, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def overlay_hud(img_bgr: np.ndarray, fps: float, conf_thr: float, nms_iou: float, view_mode: str):
    txt = f"{fps:.1f} FPS  conf={conf_thr:.2f}  nms={nms_iou:.2f}  view={view_mode}"
    cv2.putText(img_bgr, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(img_bgr, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# ------------------------- Main loop -------------------------
def main():
    ap = argparse.ArgumentParser("TinyViTDet webcam tester")
    ap.add_argument("--config", type=str, required=True, help="export JSON from training (…_export.json)")
    ap.add_argument("--weights", type=str, required=True, help="checkpoint .pth (last/best)")
    ap.add_argument("--use-ema", action="store_true", help="load EMA weights if available")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", choices=["off","bf16","fp16"], default="bf16")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    ap.add_argument("--nms", type=float, default=None, help="override NMS IoU")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--view", choices=["original","canvas"], default="original")
    # Camera args
    ap.add_argument("--cam", type=str, default="0",
                    help="Camera index (e.g. '0') OR device path (e.g. '/dev/video0') OR a GStreamer pipeline string")
    ap.add_argument("--backend", choices=["auto", "v4l2", "gstreamer", "dshow", "mf"], default="v4l2",
                    help="Capture backend to use")
    ap.add_argument("--fourcc", type=str, default="", help="Pixel format, e.g. MJPG, YUY2, H264 (if supported)")
    ap.add_argument("--fps", type=int, default=0, help="Try to set FPS (0 = leave default)")
    args = ap.parse_args()

    import re
    def is_gst_pipeline(s: str) -> bool:
        return isinstance(s, str) and ("!" in s or s.strip().startswith(("v4l2src", "filesrc", "videotestsrc")))
    def normalize_cam(cam: str, backend: str):
        # Convert '/dev/videoX' to integer index for V4L2/CAP_ANY backends
        if cam.isdigit():
            return int(cam)
        if cam.startswith("/dev/video"):
            m = re.search(r"/dev/video(\d+)$", cam)
            if m and backend in ("v4l2", "auto"):
                return int(m.group(1))
        return cam

    # CuDNN auto-tuner + device
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    # Backend map
    backend_map = {
        "auto": cv2.CAP_ANY,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
        "dshow": cv2.CAP_DSHOW,
        "mf": cv2.CAP_MSMF
    }
    backend = backend_map[args.backend]

    # Prepare camera argument and attempts
    cam_arg = normalize_cam(args.cam, args.backend)
    attempts = []

    if args.backend == "v4l2":
        # Prefer integer index for V4L2
        if isinstance(cam_arg, int):
            attempts.append(("v4l2:index", cam_arg, cv2.CAP_V4L2))
            attempts.append(("any:index",  cam_arg, cv2.CAP_ANY))
        elif isinstance(cam_arg, str) and cam_arg.startswith("/dev/video"):
            # If user insisted on path, still try it, then CAP_ANY with index
            attempts.append(("v4l2:path", cam_arg, cv2.CAP_V4L2))
            m = re.search(r"/dev/video(\d+)$", cam_arg)
            if m:
                idx = int(m.group(1))
                attempts.append(("any:index", idx, cv2.CAP_ANY))
        else:
            attempts.append(("v4l2:arg", cam_arg, cv2.CAP_V4L2))
    elif args.backend == "gstreamer":
        attempts.append(("gst:arg", cam_arg, cv2.CAP_GSTREAMER))
        if isinstance(cam_arg, str) and cam_arg.startswith("/dev/video") and not is_gst_pipeline(cam_arg):
            # Build a simple v4l2src pipeline as fallback
            caps = []
            if args.width and args.height:
                caps.append(f"width={args.width},height={args.height}")
            if args.fps:
                caps.append(f"framerate={args.fps}/1")
            caps_str = ",".join(caps)
            if caps_str:
                caps_str = "," + caps_str
            gst_pipe = f"v4l2src device={cam_arg} ! video/x-raw{caps_str} ! videoconvert ! appsink"
            attempts.append(("gst:auto-pipe", gst_pipe, cv2.CAP_GSTREAMER))
    else:
        # auto / other backends
        attempts.append(("any:arg", cam_arg, cv2.CAP_ANY))

    # Try to open capture
    cap = None
    chosen = None
    for tag, arg, be in attempts:
        cap_try = cv2.VideoCapture(arg, be)
        if cap_try.isOpened():
            cap = cap_try
            chosen = (tag, arg, be)
            break
        cap_try.release()

    if cap is None:
        raise RuntimeError(f"Cannot open camera: {args.cam} (backend={args.backend}). "
                           f"Tried: {[(t,a) for t,a,_ in attempts]}")

    # Configure stream properties
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    print(f"[camera] opened via {chosen[0]} -> {chosen[1]}")

    # Load model + thresholds
    model, img_size, conf_thr0, nms_iou0, amp_dtype = load_model(
        args.config, args.weights, args.use_ema, device, args.amp
    )
    conf_thr = float(args.conf) if args.conf is not None else conf_thr0
    nms_iou  = float(args.nms)  if args.nms  is not None else nms_iou0

    print("Press q/ESC to quit. [/]=conf  ;/'=nms  o=toggle view  s=save")
    t_prev = time.time()
    view_mode = args.view
    frame_id = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            canvas_rgb, meta = resize_and_center_crop_rgb(frame_rgb, img_size)
            inp = normalize_to_tensor(canvas_rgb, device)

            with torch.no_grad(), torch.autocast(
                device_type=("cuda" if device.type == "cuda" else "cpu"),
                dtype=amp_dtype,
                enabled=(amp_dtype is not None and device.type == "cuda")
            ):
                p_logits, boxes01 = model(inp)
                scores = torch.sigmoid(p_logits[0])
                boxes01 = boxes01[0]

            keep = torch.nonzero(scores > conf_thr, as_tuple=False).flatten()
            if keep.numel() > 0:
                bx = boxes_cxcywh_to_xyxy(boxes01[keep], img_size).clamp(0, img_size - 1)
                sc = scores[keep]
                if bx.shape[0] > 0:
                    idx = nms_xyxy(bx, sc, nms_iou)
                    bx = bx[idx]; sc = sc[idx]
                if bx.shape[0] > args.max_det:
                    ord_idx = torch.argsort(sc, descending=True)[:args.max_det]
                    bx = bx[ord_idx]; sc = sc[ord_idx]
                boxes_canvas = bx.detach().cpu().numpy()
                scores_np = sc.detach().cpu().numpy()
            else:
                boxes_canvas = np.empty((0,4), dtype=np.float32)
                scores_np = np.empty((0,), dtype=np.float32)

            if view_mode == "original":
                img_show = frame_bgr.copy()
                if boxes_canvas.size:
                    boxes_orig = map_canvas_boxes_to_original(boxes_canvas, meta)
                    draw_boxes(img_show, boxes_orig, scores_np, conf_thr)
            else:
                img_show = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)
                if boxes_canvas.size:
                    draw_boxes(img_show, boxes_canvas, scores_np, conf_thr)

            t_now = time.time()
            fps = 1.0 / max(t_now - t_prev, 1e-9)
            t_prev = t_now
            overlay_hud(img_show, fps, conf_thr, nms_iou, view_mode)

            cv2.imshow("TinyViTDet Webcam", img_show)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('['):
                conf_thr = max(0.0, conf_thr - 0.02)
            elif k == ord(']'):
                conf_thr = min(1.0, conf_thr + 0.02)
            elif k == ord(';'):
                nms_iou = max(0.05, nms_iou - 0.02)
            elif k == ord('\''):
                nms_iou = min(0.95, nms_iou + 0.02)
            elif k == ord('o'):
                view_mode = "canvas" if view_mode == "original" else "original"
            elif k == ord('s'):
                out_path = f"webcam_{frame_id:06d}.jpg"
                cv2.imwrite(out_path, img_show)
                print(f"[save] {out_path}")
            frame_id += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
