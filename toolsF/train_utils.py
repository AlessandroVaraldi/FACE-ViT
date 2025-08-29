#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities per TinyViTDet:
  - Backbone ViT minimale (PatchEmbed, Block con MHSA+MLP, DropPath)
  - Testa detection anchor-free per token (p_logit, cx,cy,w,h in [0,1])
  - Dataset WIDER-like (images/ + {train,val}_annots.json) con transforms.v2
  - Dataloaders con collate per liste di box
Compatibile con torchvision 0.23 (usa tv_tensors.BoundingBoxes e v2).
"""

from __future__ import annotations
import os, math, json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision.tv_tensors import BoundingBoxes

# ----------------------- Costanti & util -----------------------
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def low_prio_worker_init(_):
    """Abbassa (best-effort) la priorità CPU/I/O dei worker del DataLoader."""
    try:
        os.nice(10)
    except Exception:
        pass
    try:
        import psutil  # opzionale
        p = psutil.Process(os.getpid())
        if hasattr(p, "ionice"):
            try:
                p.ionice(psutil.IOPRIO_CLASS_BE, value=7)  # very low I/O prio
            except Exception:
                pass
    except Exception:
        pass

# ----------------------- DropPath (stochastic depth) -----------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1) per token sequence
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ----------------------- MLP -----------------------
class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ----------------------- Attention -----------------------
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim deve essere multiplo di heads"
        self.dim = dim
        self.heads = heads
        self.dk = dim // heads
        self.scale = self.dk ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        for m in (self.qkv, self.proj):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B,N,3C)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dk).permute(2, 0, 3, 1, 4)  # (3,B,h,N,dk)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,h,N,dk)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,N,N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B,h,N,dk)
        out = out.transpose(1, 2).reshape(B, N, C)  # (B,N,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ----------------------- Transformer Block -----------------------
class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dff: int, drop_path: float = 0.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dff, drop)

        nn.init.zeros_(self.norm1.bias); nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm2.bias); nn.init.ones_(self.norm2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ----------------------- Patch Embedding -----------------------
class PatchEmbed(nn.Module):
    """
    Converte l'immagine in una sequenza di token di patch.
    """
    def __init__(self, img_size: int, patch: int, in_chans: int, dim: int):
        super().__init__()
        assert img_size % patch == 0, "img_size deve essere multiplo di patch"
        self.img_size = img_size
        self.patch = patch
        self.grid = img_size // patch
        self.np = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) -> (B,N,D)
        x = self.proj(x)               # (B,D,G,G)
        x = x.flatten(2).transpose(1, 2)  # (B,N,D)
        return x

# ----------------------- TinyViTDet -----------------------
class TinyViTDet(nn.Module):
    """
    Detection anchor-free, per token (griglia = img_size//patch)^2.
    Output:
      - p_logits: (B, N)
      - boxes:   (B, N, 4) in [0,1] (cx, cy, w, h) riferiti all'immagine
    """
    def __init__(self, img: int, patch: int, d: int, depth: int, heads: int, dff: int,
                 dp: float = 0.0, do: float = 0.0):
        super().__init__()
        self.img, self.patch = img, patch
        self.embed = PatchEmbed(img, patch, 3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed.np + 1, d))

        dpr = torch.linspace(0, dp, steps=depth).tolist()
        self.layers = nn.ModuleList([Block(d, heads, dff, dpr[i], do) for i in range(depth)])

        self.head_det = nn.Linear(d, 5)  # (p_logit, cx, cy, w, h)

        # init
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head_det.weight, std=0.02)
        nn.init.zeros_(self.head_det.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        x = self.embed(x)  # (B,N,D)
        x = torch.cat([self.cls.expand(B, 1, -1), x], dim=1)  # prepend [CLS]
        x = x + self.pos_embed
        for blk in self.layers:
            x = blk(x)
        t = x[:, 1:, :]       # rimuovi [CLS]
        pred = self.head_det(t)    # (B,N,5)
        p_logits = pred[..., 0]    # (B,N)
        boxes = pred[..., 1:].sigmoid()  # (B,N,4) in [0,1]
        return p_logits, boxes

# ----------------------- Dataset -----------------------
class FaceDetDataset(Dataset):
    """
    Struttura attesa:
      root/
        images/<split>/...   (file immagine, es. .jpg/.png)
        train_annots.json    [{"file":"train/<subdir>/img.jpg","boxes":[[x1,y1,x2,y2], ...]}, ...]
        val_annots.json      idem, con "val/..."
    """
    def __init__(self, root: str | Path, img_size: int, split: str = "train"):
        super().__init__()
        self.root = Path(root)
        self.img_dir = self.root / "images"
        assert split in ("train", "val"), "split deve essere 'train' o 'val'"
        self.split = split
        ann_path = self.root / f"{split}_annots.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotazioni non trovate: {ann_path}")
        with open(ann_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.img_size = img_size
        self.bb_fmt = "XYXY"

        # Pipeline v2 (applica le stesse trasformazioni a image & boxes)
        if split == "train":
            self.t = T.Compose([
                T.RandomResizedCrop(
                    self.img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33),
                    interpolation=T.InterpolationMode.BICUBIC, antialias=True
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.SanitizeBoundingBoxes(labels_getter=None),  # <-- FIX
                T.ToImage(),                                   # -> Tensor (C,H,W), uint8
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(IMNET_MEAN, IMNET_STD),
            ])
        else:
            long_side = int(self.img_size * 256 / 224)
            self.t = T.Compose([
                T.Resize(long_side, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(self.img_size),
                T.SanitizeBoundingBoxes(labels_getter=None),  # <-- FIX
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(IMNET_MEAN, IMNET_STD),
            ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rec = self.items[i]
        # Percorso relativo dentro images/ (es. "train/0--Parade/xxx.jpg")
        rel_file = rec["file"]
        path = self.img_dir / rel_file
        img = Image.open(path).convert("RGB")
        H, W = img.height, img.width

        boxes = torch.as_tensor(rec.get("boxes", []), dtype=torch.float32)  # (M,4) XYXY (pixel)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)

        bb = BoundingBoxes(boxes, format=self.bb_fmt, canvas_size=(H, W))
        img_t, boxes_t = self.t(img, bb)   # <-- passa due argomenti
        boxes_t = boxes_t.to(torch.float32) if boxes_t.numel() else torch.empty((0, 4), dtype=torch.float32)
        return img_t, boxes_t

# ----------------------- Loader helper -----------------------
def _collate_det(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    batch: lista di (img_t, boxes_t)
    ritorna (lista_img, lista_boxes) per flessibilità (stack in train.py)
    """
    imgs, boxes = zip(*batch)  # tuple di tensors
    return list(imgs), list(boxes)

def face_det_loaders(root: str | Path, img: int, bs: int, workers: int = 4):
    """
    Crea DataLoader train/val per FaceDetDataset.
    """
    tr = FaceDetDataset(root, img, "train")
    va = FaceDetDataset(root, img, "val")

    # kwargs comuni
    kw_common = dict(
        pin_memory=True,
        persistent_workers=(workers > 0),
        worker_init_fn=low_prio_worker_init if workers > 0 else None,
        collate_fn=_collate_det,
    )
    if workers > 0:
        kw_common["prefetch_factor"] = 2

    ld_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=workers, **kw_common)
    ld_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=workers, **kw_common)
    return ld_tr, ld_va

# ----------------------- Esportazioni -----------------------
__all__ = [
    "TinyViTDet",
    "FaceDetDataset",
    "face_det_loaders",
    "IMNET_MEAN",
    "IMNET_STD",
    "low_prio_worker_init",
    "PatchEmbed",
    "Block",
    "DropPath",
]
