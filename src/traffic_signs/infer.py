#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: infer.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Run inference on a single image or directory of images.

Usage: 
python -m traffic_signs.infer --checkpoint runs/baseline/checkpoints/best.h5 --input path/to/img_or_dir --class_map_json runs/baseline/class_map.json

Notes: 
- Outputs predictions JSON in run folder
===================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from .config import DEFAULTS


def parse_args():
    p = argparse.ArgumentParser(description="Inference for Traffic Signs CNN")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Image file or directory")
    p.add_argument("--img_size", type=int, default=DEFAULTS.img_size)
    p.add_argument("--class_map_json", type=str, required=True)
    return p.parse_args()


def load_image(path: Path, img_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def gather_images(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        exts = ["*.png", "*.jpg", "*.jpeg", "*.ppm"]
        files = []
        for e in exts:
            files.extend(input_path.rglob(e))
        return sorted(files)
    else:
        return [input_path]


def main():
    args = parse_args()
    ckpt = Path(args.checkpoint)
    model = keras.models.load_model(ckpt)

    with open(args.class_map_json) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    input_path = Path(args.input)
    files = gather_images(input_path)
    if not files:
        raise FileNotFoundError("No images found for inference")

    imgs = np.stack([load_image(p, args.img_size) for p in files], axis=0)
    probs = model.predict(imgs, verbose=0)
    preds = probs.argmax(axis=1)

    results = [
        {
            "file": str(p),
            "pred_id": int(pid),
            "pred_label": class_map.get(int(pid), str(pid)),
            "confidence": float(probs[i, pid]),
        }
        for i, (p, pid) in enumerate(zip(files, preds))
    ]

    out_json = ckpt.parent.parent / "inference_results.json"
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results[:5], indent=2))
    print(f"Saved all results to: {out_json}")


if __name__ == "__main__":
    main()
