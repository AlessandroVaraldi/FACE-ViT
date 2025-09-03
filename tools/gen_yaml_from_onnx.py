#!/usr/bin/env python3
# Rebuild TinyViTDet YAML (for INT8 packer) from export.json + ONNX
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import onnx
import yaml

def np_init_map(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    T = {}
    for t in model.graph.initializer:
        T[t.name] = onnx.numpy_helper.to_array(t)
    # also collect Constant nodes
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output:
            for a in n.attribute:
                if a.name == "value":
                    T[n.output[0]] = onnx.numpy_helper.to_array(a.t)
    return T

def get_init(T: Dict[str, np.ndarray], keys: Iterable[str], required=True) -> Optional[np.ndarray]:
    for k in keys:
        if k in T:
            v = T[k]
            return v if isinstance(v, np.ndarray) else onnx.numpy_helper.to_array(v)
    if required:
        raise KeyError(f"Missing initializer; tried: {list(keys)}")
    return None

def infer_from_onnx(T: Dict[str, np.ndarray]) -> Dict[str, int]:
    out: Dict[str, int] = {}

    # DMODEL & TOKENS from pos_embed if available
    pos = get_init(T, ("pos_embed", "pos_embedding", "positional_embedding", "pos_embed.weight"), required=False)
    if pos is not None:
        if pos.ndim == 3 and pos.shape[0] == 1:
            pos = pos[0]
        tokens, dmodel = int(pos.shape[0]), int(pos.shape[1])
        out["TOKENS"] = tokens
        out["DMODEL"] = dmodel

    # OUT_DIM & DMODEL from head_det or head
    head_w = get_init(T, ("head_det.weight", "head.weight", "head_det.out.weight"), required=False)
    if head_w is not None:
        out_dim, dmodel2 = int(head_w.shape[0]), int(head_w.shape[1])
        out["OUT_DIM"] = out_dim
        out.setdefault("DMODEL", dmodel2)

    # PATCH_DIM from PatchEmbed conv
    pe_w = get_init(T,
        ("embed.proj.weight", "patch_embed.proj.weight", "patch_embed.weight", "embed.weight"),
        required=False)
    if pe_w is not None:
        if pe_w.ndim == 4:
            # O x I x kH x kW
            _, C, kH, kW = pe_w.shape
            out["PATCH_DIM"] = int(C * kH * kW)
        elif pe_w.ndim == 2:
            # DMODEL x PATCH_DIM
            out["PATCH_DIM"] = int(pe_w.shape[1])

    # DFF from FFN1 weight (mlp.fc1)
    ffn1_w = get_init(T, ("layers.0.ffn.fc1.weight", "layers.0.mlp.fc1.weight"), required=False)
    if ffn1_w is not None:
        out["DFF"] = int(ffn1_w.shape[0])

    return out

def main():
    ap = argparse.ArgumentParser("Generate TinyViTDet YAML for int8 packer (no retrain)")
    ap.add_argument("--export", required=True, help="..._export.json written by training/export_onnx")
    ap.add_argument("--onnx",   required=True, help="model.onnx exported from training")
    ap.add_argument("--out",    required=True, help="output YAML path")
    args = ap.parse_args()

    exp = json.loads(Path(args.export).read_text())
    model_json = exp.get("MODEL", {})
    img_size = int(model_json["img_size"])
    patch    = int(model_json["patch"])
    dmodel   = int(model_json["dim"])
    layers   = int(model_json["layers"])
    heads    = int(model_json["heads"])
    dff      = int(model_json["dff"])

    # grid & tokens from JSON (fallbacks)
    grid    = img_size // patch
    patches = grid * grid
    tokens  = patches + 1  # CLS + patches

    # ONNX sanity/inference
    onnx_model = onnx.load(args.onnx)
    T = np_init_map(onnx_model)
    inferred = infer_from_onnx(T)

    # Merge (prefer ONNX for shapes if present & mismatch)
    DMODEL = inferred.get("DMODEL", dmodel)
    if DMODEL != dmodel:
        print(f"[WARN] DMODEL from JSON={dmodel} vs ONNX={DMODEL} — using ONNX.")

    OUT_DIM   = inferred.get("OUT_DIM", 5)  # TinyViTDet: 5 (p, cx, cy, w, h)
    PATCH_DIM = inferred.get("PATCH_DIM", 3 * patch * patch)

    TOKENS_onx = inferred.get("TOKENS")
    if TOKENS_onx is not None and TOKENS_onx != tokens:
        print(f"[WARN] TOKENS from JSON-grid={tokens} vs ONNX={TOKENS_onx} — using ONNX.")
        tokens  = TOKENS_onx
        patches = tokens - 1

    cfg = {
        # minimal core for the packer
        "DMODEL": DMODEL,
        "DFF":    dff,
        "HEADS":  heads,
        "LAYERS": layers,
        "OUT_DIM": OUT_DIM,
        "PATCH_DIM": PATCH_DIM,
        "TOKENS": tokens,
        "PATCHES": patches,

        # helpful extras (not strictly required by the packer)
        "IMG":   img_size,
        "PATCH": patch,
        "POSENC": "APE" if "TOKENS" in inferred else "NONE",

        # keep EPS_SHIFT default used by your engine
        "EPS_SHIFT": 12,

        # no 'shifts' here — let onnx2int8 compute them with --auto-shift
        # "shifts": {...}
    }

    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"✅ Wrote {args.out}")
    print("   Tip: run packer with --auto-shift to fill fixed-point shifts.")

if __name__ == "__main__":
    main()

# python tools/gen_yaml_from_onnx.py  --export checkpoints/tinyvitdet_faces_export.json --onnx models/tinyvitdet.onnx --out models/tinyvitdet.yaml