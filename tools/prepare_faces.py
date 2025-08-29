#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare WIDER FACE for TinyViTDet:
  - Download train/val via torchvision (requires: pip install torchvision gdown)
  - Create root/images/{train,val}/... with symlinks (fallback to hardlink/copy)
  - Export root/{train_annots.json,val_annots.json} (XYXY pixel boxes)
  - Also create root/images/test/... if raw WIDER_test images are available;
    otherwise create an empty folder and print a warning.

Examples:
  python prepare_faces.py --root ./faces
  python prepare_faces.py --root ./faces --min-size 8 --limit-train 5000 --limit-val 1000
"""

from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path
from typing import List, Dict

# ---- dependencies ----
try:
    from torchvision.datasets import WIDERFace
except Exception as e:
    print("ERROR: torchvision not available. Install with: pip install torchvision", file=sys.stderr)
    raise

def xywh_to_xyxy(box):
    x, y, w, h = [int(v) for v in box]
    return [x, y, x + max(0, w), y + max(0, h)]

def safe_link(src: Path, dst: Path):
    """Create a symlink; if it fails, try hardlink; if that fails, copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)           # symlink
    except Exception:
        try:
            os.link(src, dst)         # hardlink
        except Exception:
            shutil.copy2(src, dst)    # copy

def build_split(torch_root: Path, split: str, out_images: Path,
                min_size: int, limit: int | None) -> List[Dict]:
    """
    Convert the WIDERFace split into JSON annots and link images into out_images/<split>/...
    Filters invalid boxes and tiny faces (min side < min_size).
    """
    ds = WIDERFace(str(torch_root), split=split, download=True)  # downloads if needed
    records: List[Dict] = []
    n = len(ds)
    take = n if limit is None else min(limit, n)

    for i in range(take):
        # Access dataset internals to get absolute image path + annotations
        info = ds.img_info[i]
        img_path = Path(info["img_path"])
        ann = info["annotations"]  # contains keys like: bbox, invalid, etc.

        bboxes = ann["bbox"]           # tensor Nx4 (x,y,w,h)
        invalid = ann.get("invalid", None)
        boxes_xyxy: List[List[int]] = []

        for j in range(bboxes.shape[0]):
            if invalid is not None and int(invalid[j]) == 1:
                continue
            x, y, w, h = [int(v) for v in bboxes[j].tolist()]
            if w <= 0 or h <= 0:
                continue
            if min(w, h) < min_size:
                continue
            boxes_xyxy.append(xywh_to_xyxy((x, y, w, h)))

        # Keep the hierarchy after ".../images/"
        try:
            rel_after_images = img_path.as_posix().split("/images/", 1)[1]
        except Exception:
            rel_after_images = img_path.name

        rel_dst = Path(split) / rel_after_images   # prefix with split
        dst = out_images / rel_dst
        safe_link(img_path, dst)

        records.append({"file": rel_dst.as_posix(), "boxes": boxes_xyxy})

    return records

def link_wider_test_images(torch_root: Path, out_images: Path, limit: int | None = None) -> int:
    """
    Link WIDER_test images (no annotations) into images/test/...
    Returns the number of images linked. If raw test images are not found, returns 0.
    """
    src_root = torch_root / "widerface" / "WIDER_test" / "images"
    if not src_root.exists():
        # torchvision usually does not download WIDER_test; we handle this gracefully.
        (out_images / "test").mkdir(parents=True, exist_ok=True)
        print("[WARN] WIDER_test images not found in cache. Created empty images/test/.")
        return 0

    count = 0
    for img_path in src_root.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        rel_after_images = img_path.as_posix().split("/images/", 1)[1]
        rel_dst = Path("test") / rel_after_images
        dst = out_images / rel_dst
        safe_link(img_path, dst)
        count += 1
        if limit and count >= limit:
            break

    print(f"[INFO] Linked {count} test images into {out_images/'test'}")
    return count

def main():
    ap = argparse.ArgumentParser("Prepare WIDER FACE")
    ap.add_argument("--root", type=str, required=True, help="output folder (creates images/ + json)")
    ap.add_argument("--min-size", type=int, default=4, help="discard faces with min side < this (pixels)")
    ap.add_argument("--limit-train", type=int, default=None, help="limit number of train images (debug)")
    ap.add_argument("--limit-val", type=int, default=None, help="limit number of val images (debug)")
    ap.add_argument("--limit-test", type=int, default=None, help="limit number of test images (debug, if available)")
    args = ap.parse_args()

    root = Path(args.root)
    out_images = root / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    # Where torchvision stores WIDERFace (keep it inside root to avoid duplication elsewhere)
    torch_root = root / "_widerface_raw"

    print("[1/4] Ensuring WIDER FACE (train/val) are available via torchvision…")
    _ = WIDERFace(str(torch_root), split="train", download=True)
    _ = WIDERFace(str(torch_root), split="val", download=True)
    print("OK.")

    print("[2/4] Building train split (links + annotations)…")
    train_recs = build_split(torch_root, "train", out_images, args.min_size, args.limit_train)
    with open(root / "train_annots.json", "w") as f:
        json.dump(train_recs, f)

    print("[3/4] Building val split (links + annotations)…")
    val_recs = build_split(torch_root, "val", out_images, args.min_size, args.limit_val)
    with open(root / "val_annots.json", "w") as f:
        json.dump(val_recs, f)

    print("[4/4] Creating images/test/…")
    _ = link_wider_test_images(torch_root, out_images, args.limit_test)

    # Quick stats
    def stats(recs):
        n_imgs = len(recs)
        n_boxes = sum(len(r["boxes"]) for r in recs)
        return n_imgs, n_boxes

    ti, tb = stats(train_recs)
    vi, vb = stats(val_recs)
    print(f"DONE ✓\n  Train: {ti} images, {tb} faces\n  Val:   {vi} images, {vb} faces\nOutput: {root}")

if __name__ == "__main__":
    main()

# python3 prepare_faces.py --root ./faces --limit-train 2000 --limit-val 500