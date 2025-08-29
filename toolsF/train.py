#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TinyViTDet training script (face detection, anchor-free per-patch).

- Model: TinyViTDet (backbone ViT con testa per-token -> p_logit, (cx,cy,w,h) in [0,1])
- Data: face_det_loaders(root, img_size, bs, workers) -> (train_loader, val_loader)
- Loss: focal BCE (cls) + SmoothL1 (box sui positivi)
- Val: loss e Precision/Recall/F1 @ IoU=0.5 con NMS
- Extras: AMP (bf16/fp16), torch.compile, EMA, resume, grad-accum, cosine LR + warmup,
          checkpoint (best/last), ONNX export (p_logits, boxes)

Usage (example):
  python train.py --data /path/to/dataset --epochs 50 --img-size 256 --patch 16 --bs 128 --pretrained checkpoints/tinyvit_cls.pth --compile --amp bf16
"""

from __future__ import annotations
import argparse, os, math, time, json, random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from train_utils import TinyViTDet, face_det_loaders

torch.set_float32_matmul_precision("high")

# ------------------------- Utility -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def smart_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj

def smart_load_pretrained(model: nn.Module, path: str):
    if not path: return
    sd = torch.load(path, map_location="cpu")
    sd = smart_state_dict(sd)
    msd = model.state_dict()
    copied = 0
    new_sd = {}
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            new_sd[k] = v; copied += v.numel()
    model.load_state_dict({**msd, **new_sd})
    print(f"[pretrained] Copied {copied} compatible parameters from: {path}")

# ------------------------- Detection helpers -------------------------
def build_targets(batch_boxes: List[torch.Tensor], img_size: int, patch: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(batch_boxes)
    G = img_size // patch
    N = G * G
    y_p = torch.zeros(B, N, dtype=torch.float32, device=device)
    y_b = torch.zeros(B, N, 4, dtype=torch.float32, device=device)
    for b, boxes in enumerate(batch_boxes):
        if boxes.numel() == 0: 
            continue
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) * 0.5 / img_size
            cy = (y1 + y2) * 0.5 / img_size
            w  = (x2 - x1) / img_size
            h  = (y2 - y1) / img_size
            gx = int(min(max((cx * img_size) // patch, 0), G - 1))
            gy = int(min(max((cy * img_size) // patch, 0), G - 1))
            idx = gy * G + gx
            y_p[b, idx] = 1.0
            y_b[b, idx, :] = torch.tensor([cx, cy, w, h], device=device)
    return y_p, y_b

def focal_bce(logits: torch.Tensor, target: torch.Tensor, alpha=0.25, gamma=2.0) -> torch.Tensor:
    # logits, target: (B, N)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = p * target + (1 - p) * (1 - target)
    w  = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt).pow(gamma) * ce).mean()

def boxes_cxcywh_to_xyxy(b: torch.Tensor, img_size: int) -> torch.Tensor:
    # b: (K,4) in [0,1] (cx,cy,w,h) -> xyxy in pixel
    cx, cy, w, h = b.unbind(-1)
    x1 = (cx - 0.5 * w) * img_size
    y1 = (cy - 0.5 * h) * img_size
    x2 = (cx + 0.5 * w) * img_size
    y2 = (cy + 0.5 * h) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (M,4), b: (N,4)
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

# ------------------------- Train / Val -------------------------
def train_one_epoch(model, loader, opt, scaler, scheduler, ema, args, device, epoch):
    """
    Fix:
      - Warmup LR per-step (lineare) basato sui passi effettivi dopo l'accumulo
      - Step dell'optimizer anche sull'ultimo batch se non multiplo di args.accum
      - Grad clipping dopo unscale_ quando AMP è attivo
    """
    import math
    model.train()
    t0 = time.time()
    loss_sum = loss_c_sum = loss_b_sum = 0.0
    step_eff = 0  # passi effettivi (dopo accumulo)
    opt.zero_grad(set_to_none=True)

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)

    # --- helper: warmup per-step (lineare) ---
    steps_per_epoch = (len(loader) + max(1, args.accum) - 1) // max(1, args.accum)  # ceil
    warmup_steps_total = steps_per_epoch * max(0, args.warmup_epochs)

    def apply_warmup_lr(global_step_eff: int):
        """Imposta lr = args.lr * factor, con factor lineare in [1/warmup_steps_total .. 1]."""
        if warmup_steps_total <= 0:
            return
        if global_step_eff < warmup_steps_total:
            factor = float(global_step_eff + 1) / float(warmup_steps_total)
            for g in opt.param_groups:
                g["lr"] = args.lr * factor

    # --- helper: un singolo optimizer step (dopo accum) ---
    def do_optimizer_step():
        nonlocal step_eff
        # warmup LR per-step (prima dello step, così lo step usa il nuovo LR)
        global_step_eff = (epoch - 1) * steps_per_epoch + step_eff
        apply_warmup_lr(global_step_eff)

        # grad clipping
        if args.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # step + scaler
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        # EMA
        if ema is not None:
            ema.update_parameters(model)

        # Scheduler (solo se passato; tipicamente dopo il warmup)
        if scheduler is not None:
            scheduler.step()

        step_eff += 1

    for it, batch in enumerate(loader):
        imgs, boxes_list = batch
        if isinstance(imgs, (list, tuple)):  # robustezza a collate custom
            imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(device, non_blocking=True)
        boxes_list = [b.to(device) for b in boxes_list]

        with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"),
                            dtype=amp_dtype, enabled=(args.amp != "off" and device.type == "cuda")):
            p_pred, b_pred = model(imgs)  # (B,N), (B,N,4)
            y_p, y_b = build_targets(boxes_list, args.img_size, args.patch, device)

            loss_cls = focal_bce(p_pred, y_p, alpha=args.focal_alpha, gamma=args.focal_gamma)
            pos = (y_p > 0.5)
            if pos.any():
                loss_box = F.smooth_l1_loss(b_pred[pos], y_b[pos], reduction="mean")
            else:
                loss_box = p_pred.sum() * 0.0

            loss = args.lambda_cls * loss_cls + args.lambda_box * loss_box

        # backward (con/ senza scaler)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # accumulo
        do_step_now = ((it + 1) % max(1, args.accum) == 0)
        if do_step_now:
            do_optimizer_step()

        # metriche aggregate
        loss_sum  += float(loss.item())
        loss_c_sum+= float(loss_cls.item())
        loss_b_sum+= float(loss_box.item())

    # flush: se l'ultimo batch non ha triggerato lo step a causa dell'accumulo
    if (len(loader) % max(1, args.accum)) != 0:
        do_optimizer_step()

    dt = time.time() - t0
    n_iter = len(loader)
    print(f"[E{epoch:03d}] train  loss {loss_sum/n_iter:.4f}  cls {loss_c_sum/n_iter:.4f}  box {loss_b_sum/n_iter:.4f}  time {dt:.1f}s  steps {step_eff}")
    return loss_sum / n_iter

@torch.no_grad()
def validate(model, loader, args, device, use_ema=True):
    model.eval()
    net = model if not (hasattr(model, "module")) else model.module

    if hasattr(net, "_ema_src") and use_ema:
        pass

    loss_m, loss_c_m, loss_b_m = 0.0, 0.0, 0.0
    TP = FP = FN = 0

    for batch in loader:
        imgs, boxes_list = batch
        if isinstance(imgs, (list, tuple)): imgs = torch.stack(imgs, dim=0)
        imgs = imgs.to(device, non_blocking=True)
        gt_list = [b.to(device) for b in boxes_list]

        p_pred, b_pred = model(imgs)  # (B,N), (B,N,4)
        y_p, y_b = build_targets(gt_list, args.img_size, args.patch, device)
        loss_cls = focal_bce(p_pred, y_p, alpha=args.focal_alpha, gamma=args.focal_gamma)
        pos = (y_p > 0.5)
        if pos.any():
            loss_box = F.smooth_l1_loss(b_pred[pos], y_b[pos], reduction="mean")
        else:
            loss_box = p_pred.sum() * 0.0
        loss = args.lambda_cls * loss_cls + args.lambda_box * loss_box
        loss_m   += loss.item()
        loss_c_m += loss_cls.item()
        loss_b_m += loss_box.item()

        # --- metriche semplici @IoU=0.5 ---
        B, N = p_pred.shape
        scores = torch.sigmoid(p_pred)
        for b in range(B):
            s = scores[b]  # (N,)
            bb = b_pred[b] # (N,4) in [0,1]
            keep = torch.nonzero(s > args.conf_thr, as_tuple=False).flatten()
            if keep.numel():
                bx = boxes_cxcywh_to_xyxy(bb[keep], args.img_size).clamp(0, args.img_size-1)
                sc = s[keep]
                keep2 = nms_xyxy(bx, sc, args.nms_iou)
                bx = bx[keep2]; sc = sc[keep2]
            else:
                bx = torch.empty((0,4), device=device)

            gt = gt_list[b]  # (M,4) xyxy pixel
            if gt.numel() == 0 and bx.numel() == 0:
                continue
            if gt.numel() == 0 and bx.numel() > 0:
                FP += bx.shape[0]
                continue
            if gt.numel() > 0 and bx.numel() == 0:
                FN += gt.shape[0]
                continue

            ious = box_iou_xyxy(bx, gt)  # (K,M)
            # matching greedy
            matched_gt = set()
            matched_pr = set()
            for i in range(ious.shape[0]):
                j = torch.argmax(ious[i]).item()
                if ious[i, j].item() >= 0.5 and j not in matched_gt:
                    TP += 1
                    matched_gt.add(j)
                    matched_pr.add(i)
            FP += (bx.shape[0] - len(matched_pr))
            FN += (gt.shape[0] - len(matched_gt))

    n = len(loader)
    prec = TP / (TP + FP + 1e-9)
    rec  = TP / (TP + FN + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    print(f"[VAL] loss {loss_m/n:.4f}  cls {loss_c_m/n:.4f}  box {loss_b_m/n:.4f}  P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}")
    return (loss_m / n), dict(precision=prec, recall=rec, f1=f1)


def _unwrap_and_clean_state_dict(m: torch.nn.Module) -> dict:
    base = m
    if hasattr(base, "module"):     # AveragedModel (EMA)
        base = base.module
    if hasattr(base, "_orig_mod"):  # torch.compile wrapper
        base = base._orig_mod
    sd = base.state_dict()
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module._orig_mod.", "_orig_mod.", "module."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v
    cleaned.pop("n_averaged", None)
    return cleaned

def export_onnx(args, trained_model, device):
    # recreate a "clean" model on CPU
    export_model = TinyViTDet(
        args.img_size, args.patch, args.dim, args.layers, args.heads, args.dff,
        dp=args.drop_path, do=args.dropout
    ).to("cpu").eval()

    # take weights from trained model (EMA if present) and clean the keys
    sd = _unwrap_and_clean_state_dict(trained_model)
    missing, unexpected = export_model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[export] warning: missing={len(missing)} unexpected={len(unexpected)}")

    dummy = torch.zeros(1, 3, args.img_size, args.img_size, device="cpu")

    torch.onnx.export(
        export_model, dummy, args.export_onnx,
        input_names=["images"], output_names=["p_logits", "boxes"],
        opset_version=18, do_constant_folding=False, dynamo=True,
        # if you want dynamic batch size:
        # dynamic_axes={"images": {0: "N"}, "p_logits": {0: "N"}, "boxes": {0: "N"}}
    )

    cfg = {
        "TASK": "FACEDET",
        "MODEL": {
            "img_size": args.img_size, "patch": args.patch, "dim": args.dim,
            "layers": args.layers, "heads": args.heads, "dff": args.dff
        },
        "POST": {"conf_thr": args.conf_thr, "nms_iou": args.nms_iou}
    }
    with open(Path(args.save_dir) / f"{args.name}_export.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[export] ONNX salvato -> {args.export_onnx}")

# ------------------------- Test-time Evaluation -------------------------
@torch.no_grad()
def evaluate_test(args, trained_model, device):
    """
    Valutazione deterministica sul test set in spazio canvas (img_size x img_size):
      - Applica Resize(shorter_side=int(img*256/224)) + CenterCrop(img_size) anche alle box GT.
      - Usa NMS e soglie da args.test_* se presenti (altrimenti args.conf_thr / args.nms_iou).
      - Riporta P/R/F1 @conf e mAP@0.5, mAP@[0.5:0.95].
    """
    import json
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    # ---- costanti IMAGENET per normalizzazione ----
    IMNET_MEAN = (0.485, 0.456, 0.406)
    IMNET_STD  = (0.229, 0.224, 0.225)

    root = Path(args.data)
    test_json = root / "test_annots.json"
    img_dir = root / "images"
    if not test_json.exists():
        print("[TEST] Skipped: test_annots.json not found.")
        return

    img_size = args.img_size
    target_short = int(round(img_size * 256 / 224))  # come nel val preprocess
    conf_thr = args.test_conf if getattr(args, "test_conf", None) is not None else args.conf_thr
    nms_iou  = args.test_nms  if getattr(args, "test_nms",  None) is not None else args.nms_iou
    max_det  = max(1, getattr(args, "test_max_det", 300))

    # ---- dataset test con trasformazioni APPLICATE ANCHE ALLE BOX ----
    class FaceDetTestCanvas(Dataset):
        def __init__(self, ann_path, img_dir, img_size, target_short):
            with open(ann_path, "r", encoding="utf-8") as f:
                self.items = json.load(f)
            self.img_dir = Path(img_dir)
            self.img_size = img_size
            self.target_short = target_short

        def __len__(self): return len(self.items)

        def _resize_and_crop(self, pil_img):
            W0, H0 = pil_img.size
            # scale so min(H',W') = target_short
            if H0 < W0:
                new_h = self.target_short
                new_w = int(round(W0 * (self.target_short / H0)))
            else:
                new_w = self.target_short
                new_h = int(round(H0 * (self.target_short / W0)))
            img_res = pil_img.resize((new_w, new_h), Image.BICUBIC)
            # center crop to (img_size, img_size)
            top  = max(0, (new_h - self.img_size) // 2)
            left = max(0, (new_w - self.img_size) // 2)
            img_crop = img_res.crop((left, top, left + self.img_size, top + self.img_size))
            return img_crop, (W0, H0, new_w, new_h, left, top)

        def _transform_boxes(self, boxes_xyxy, meta):
            # boxes_xyxy: np.ndarray (M,4) in spazio originale (W0,H0)
            # meta = (W0,H0,new_w,new_h,left,top)
            _, _, _, _, left, top = meta
            # scala: fattore uniforme applicato prima del crop
            # (lo ricaviamo da new_h/new_w vs H0/W0 a seconda del lato limitante)
            # Più semplicemente: rapporto tra new dimension e old dimension sul lato usato
            # Ma per i box basta moltiplicare per (new_w/W0, new_h/H0) e poi sottrarre (left,top)
            W0, H0, new_w, new_h, _, _ = meta
            sx = new_w / max(W0, 1e-6)
            sy = new_h / max(H0, 1e-6)
            b = boxes_xyxy.astype(np.float32).copy()
            b[:, [0, 2]] = b[:, [0, 2]] * sx - left
            b[:, [1, 3]] = b[:, [1, 3]] * sy - top
            # clamp su canvas
            b[:, 0] = np.clip(b[:, 0], 0, self.img_size - 1)
            b[:, 1] = np.clip(b[:, 1], 0, self.img_size - 1)
            b[:, 2] = np.clip(b[:, 2], 0, self.img_size - 1)
            b[:, 3] = np.clip(b[:, 3], 0, self.img_size - 1)
            # filtra box degeneri
            w = (b[:, 2] - b[:, 0]) > 1e-3
            h = (b[:, 3] - b[:, 1]) > 1e-3
            keep = w & h
            return b[keep]

        def __getitem__(self, i):
            rec = self.items[i]
            path = self.img_dir / rec["file"]
            pil = Image.open(path).convert("RGB")
            crop, meta = self._resize_and_crop(pil)

            # to tensor + normalize (senza torchvision v2 per coerenza)
            arr = np.asarray(crop).astype(np.float32) / 255.0  # HWC RGB 0..1
            arr = (arr - np.asarray(IMNET_MEAN, np.float32)) / np.asarray(IMNET_STD, np.float32)
            img_t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW

            gt = np.asarray(rec.get("boxes", []), dtype=np.float32)  # xyxy nello spazio originale
            gt_canvas = self._transform_boxes(gt, meta)              # xyxy nello spazio canvas
            gt_t = torch.from_numpy(gt_canvas) if gt_canvas.size else torch.zeros((0, 4), dtype=torch.float32)
            return img_t, gt_t

    def _collate(batch):
        imgs, gts = zip(*batch)
        return list(imgs), list(gts)

    ds = FaceDetTestCanvas(test_json, img_dir, img_size, target_short)
    if len(ds) == 0:
        print("[TEST] Empty test set.")
        return

    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers,
                    pin_memory=True, persistent_workers=(args.workers > 0), collate_fn=_collate)

    model = trained_model.eval()

    # ---- accumulatori per metriche ----
    iou_thresholds = np.arange(0.50, 0.95 + 1e-9, 0.05)
    total_gts = {float(t): 0 for t in iou_thresholds}

    # per AP accumuliamo predizioni globalmente
    # records: lista di dict {score, img_id, gt_idx_best, iou_best}
    records = []

    # per P/R/F1 a soglia fissa
    TP = FP = FN = 0

    # ---- inferenza ----
    for img_batch, gt_list in dl:
        if isinstance(img_batch, (list, tuple)):
            imgs = torch.stack(img_batch, dim=0)
        else:
            imgs = img_batch
        imgs = imgs.to(device, non_blocking=True)

        p_logits, b_pred = model(imgs)  # (B,N), (B,N,4 in [0,1])
        scores_all = torch.sigmoid(p_logits).detach().cpu()     # (B,N)
        boxes01_all = b_pred.detach().cpu()                     # (B,N,4)

        B, N = scores_all.shape
        for b in range(B):
            scores = scores_all[b]            # (N,)
            boxes01 = boxes01_all[b]          # (N,4) cxcywh
            gt = gt_list[b].to(torch.float32) # (M,4) xyxy (canvas)

            # --- predizioni per AP (soglia molto bassa + NMS) ----
            keep_all = torch.nonzero(scores > 1e-6, as_tuple=False).flatten()
            if keep_all.numel():
                bx_pix = boxes_cxcywh_to_xyxy(boxes01[keep_all], img_size).clamp(0, img_size - 1)
                sc = scores[keep_all]
                # NMS per ridurre duplicati
                if bx_pix.shape[0] > 0:
                    idx = nms_xyxy(bx_pix, sc, nms_iou)
                    bx_pix = bx_pix[idx]; sc = sc[idx]
                # limita max_det (coerente con test)
                if bx_pix.shape[0] > max_det:
                    ord_idx = torch.argsort(sc, descending=True)[:max_det]
                    bx_pix = bx_pix[ord_idx]; sc = sc[ord_idx]
                # registra per AP
                if gt.numel():
                    ious = box_iou_xyxy(bx_pix, gt)  # (K,M)
                    iou_best, gt_best = ious.max(dim=1)
                    for k in range(bx_pix.shape[0]):
                        records.append(dict(score=float(sc[k]), img_id=len(records),  # unique per record
                                            img_idx=len(records),  # dummy unique; not used for 'used' key below
                                            gt_idx_best=int(gt_best[k].item()),
                                            iou_best=float(iou_best[k].item()),
                                            image_id_global=b))  # b dentro batch; sostituito sotto
                else:
                    for k in range(bx_pix.shape[0]):
                        records.append(dict(score=float(sc[k]), img_id=len(records),
                                            img_idx=len(records), gt_idx_best=-1, iou_best=0.0, image_id_global=b))
            # --- P/R/F1 @conf_thr (con NMS) ----
            keep = torch.nonzero(scores > conf_thr, as_tuple=False).flatten()
            if keep.numel():
                bx = boxes_cxcywh_to_xyxy(boxes01[keep], img_size).clamp(0, img_size - 1)
                sc = scores[keep]
                idx = nms_xyxy(bx, sc, nms_iou)
                bx = bx[idx]; sc = sc[idx]
                # limita max_det
                if bx.shape[0] > max_det:
                    ord_idx = torch.argsort(sc, descending=True)[:max_det]
                    bx = bx[ord_idx]; sc = sc[ord_idx]
            else:
                bx = torch.empty((0, 4), dtype=torch.float32)

            if gt.numel() == 0 and bx.numel() > 0:
                FP += bx.shape[0]
            elif gt.numel() > 0 and bx.numel() == 0:
                FN += gt.shape[0]
            elif gt.numel() > 0 and bx.numel() > 0:
                ious = box_iou_xyxy(bx, gt)  # (K,M)
                used = set()
                matched = 0
                for i in range(ious.shape[0]):
                    j = int(torch.argmax(ious[i]).item())
                    if ious[i, j].item() >= 0.5 and j not in used:
                        matched += 1
                        used.add(j)
                TP += matched
                FP += (bx.shape[0] - matched)
                FN += (gt.shape[0] - matched)

        # aggiorna conteggio GT globali per AP
        for t in iou_thresholds:
            total_gts[float(t)] += sum(int(g.shape[0]) for g in gt_list)

    # ---- AP su tutta la dataset ----
    def compute_ap_for_threshold(thr: float) -> float:
        if total_gts[thr] == 0:
            return 0.0
        # ordina predizioni per score desc
        recs = sorted(records, key=lambda r: -r["score"])
        tp_flags = []
        used_pairs = set()  # (image_index_global, gt_idx)
        for r in recs:
            # NOTA: records sono creati per-batch. 'image_id_global' è b nel batch,
            # ma poiché le immagini cambiano ogni batch, basta usare (id progressivo, gt_idx) come chiave unica.
            # Usiamo 'img_id' (progressivo globale) come image key.
            img_key = r["img_id"]
            if r["gt_idx_best"] >= 0 and r["iou_best"] >= thr and (img_key, r["gt_idx_best"]) not in used_pairs:
                tp_flags.append(1)
                used_pairs.add((img_key, r["gt_idx_best"]))
            else:
                tp_flags.append(0)
        if len(tp_flags) == 0:
            return 0.0
        tp = np.cumsum(tp_flags, dtype=np.float64)
        fp = np.cumsum([1 - x for x in tp_flags], dtype=np.float64)
        recall = tp / max(1, total_gts[thr])
        precision = tp / np.maximum(tp + fp, 1e-9)
        # envelope
        mpre = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.trapz(mpre, recall))
        return ap

    ap_05 = compute_ap_for_threshold(0.50)
    aps = [compute_ap_for_threshold(float(t)) for t in iou_thresholds]
    map_5095 = float(np.mean(aps)) if aps else 0.0

    # ---- P/R/F1 finali ----
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"[TEST] images={len(ds)}  conf={conf_thr}  nms={nms_iou}  max_det={max_det}")
    print(f"[TEST] P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}  |  mAP@0.5 {ap_05:.3f}  mAP@[.5:.95] {map_5095:.3f}")

# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser("TinyViTDet training")
    # Dati / modello
    ap.add_argument("--data", type=str, required=True, help="root con images/ e *_annots.json")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Train
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Loss
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--lambda-box", type=float, default=5.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)

    # Val / NMS
    ap.add_argument("--conf-thr", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--val-interval", type=int, default=1)

    # Extra
    ap.add_argument("--pretrained", type=str, default="", help="checkpoint per inizializzare backbone")
    ap.add_argument("--resume", type=str, default="", help="riprende il training da un checkpoint")
    ap.add_argument("--save-dir", type=str, default="checkpoints")
    ap.add_argument("--name", type=str, default="tinyvitdet")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-onnx", type=str, default="", help="path .onnx da esportare a fine training")

    # --- Test-time evaluation options ---
    ap.add_argument("--test-conf", type=float, default=None, help="confidence threshold for test metrics (default: --conf-thr)")
    ap.add_argument("--test-nms", type=float, default=None, help="NMS IoU for test metrics (default: --nms-iou)")
    ap.add_argument("--test-max-det", type=int, default=300, help="maximum detections per image at test-time")


    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Dataloaders
    ld_tr, ld_val = face_det_loaders(args.data, args.img_size, args.bs, args.workers)

    # Model
    net = TinyViTDet(args.img_size, args.patch, args.dim, args.layers, args.heads, args.dff,
                     dp=args.drop_path, do=args.dropout)
    smart_load_pretrained(net, args.pretrained)
    print(f"Model params: {count_params(net)/1e6:.2f} M")
    net = net.to(device)

    if args.compile and hasattr(torch, "compile"):
        net = torch.compile(net)  # torch >= 2.0
        print("[info] torch.compile attivato")

    # EMA
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)) if args.ema else None

    # Optim / sched
    opt = AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
    # Cosine LR after warmup
    import math
    total_steps = math.ceil(len(ld_tr) / max(1, args.accum)) * max(1, args.epochs - args.warmup_epochs)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-6)

    # AMP scaler
    scaler = torch.amp.GradScaler(device="cuda", enabled=(args.amp == "fp16"))

    # Resume
    start_epoch = 1
    best_metric = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema_state_dict") is not None:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt.get("epoch", 1)
        best_metric = ckpt.get("best_metric", best_metric)
        print(f"[resume] ripreso da {args.resume} @ epoch {start_epoch} (best={best_metric:.4f})")

    # Warmup schedule helper
    def warmup_lr_lambda(it_step, warmup_steps):
        if warmup_steps <= 0: return 1.0
        return min(1.0, (it_step + 1) / warmup_steps)

    # Train loop
    global_step = 0
    warmup_steps = (len(ld_tr) // max(1, args.accum)) * args.warmup_epochs
    meta_path = Path(args.save_dir) / f"meta_{args.name}.json"
    last_path = Path(args.save_dir) / f"last_{args.name}.pth"
    best_path = Path(args.save_dir) / f"best_{args.name}.pth"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            net.train()

            tr_loss = train_one_epoch(net, ld_tr, opt, scaler, scheduler if epoch > args.warmup_epochs else None, ema, args, device, epoch)

            # Val
            if (epoch % args.val_interval) == 0:
                val_loss, vstats = validate(ema if ema is not None else net, ld_val, args, device, use_ema=True)
            else:
                val_loss, vstats = float("nan"), dict(precision=float("nan"), recall=float("nan"), f1=float("nan"))

            # Aggiorna best prima di serializzare meta
            improved = False
            if (epoch % args.val_interval) == 0 and (val_loss < best_metric):
                best_metric = val_loss
                improved = True

            # Save last
            ckpt = {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": (scaler.state_dict() if scaler is not None else None),
                "best_metric": best_metric
            }
            if ema is not None:
                ckpt["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt, last_path)

            with open(meta_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": val_loss,
                    "precision": vstats["precision"],
                    "recall": vstats["recall"],
                    "f1": vstats["f1"],
                    "best_metric": best_metric
                }, f, indent=2)

            # Save best (min val_loss)
            if improved:
                torch.save(ckpt, best_path)
            print(f"[E{epoch:03d}] saved last. {'✓ BEST' if improved else ''}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    if args.export_onnx:
        export_onnx(args, ema if ema is not None else net, device)

    evaluate_test(args, ema if ema is not None else net, device)

if __name__ == "__main__":
    main()

#  python tools/train.py --data ./faces --img-size 256 --patch 16 --epochs 300 --bs 256 --workers 4 --lr 3e-4 --ema --amp bf16 --compile --save-dir checkpoints --name tinyvitdet_faces --export-onnx checkpoints/tinyvitdet_faces.onnx