#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: evaluate.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Evaluation script to load a checkpoint and compute metrics on validation/test.

Usage: 
python -m traffic_signs.evaluate --data_dir /path/to/GTSRB --checkpoint runs/baseline/checkpoints/best.h5

Notes: 
- Prints classification report and confusion matrix
===================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from .config import Config, DEFAULTS
from .data import build_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Traffic Signs CNN")
    p.add_argument("--data_dir", type=str, default=DEFAULTS.data_dir)
    p.add_argument("--train_dir", type=str, default=DEFAULTS.train_dir)
    p.add_argument("--val_dir", type=str, default=DEFAULTS.val_dir)
    p.add_argument("--test_dir", type=str, default=DEFAULTS.test_dir)
    p.add_argument("--batch_size", type=int, default=DEFAULTS.batch_size)
    p.add_argument("--img_size", type=int, default=DEFAULTS.img_size)
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(**vars(args))

    # Load data
    train_ds, val_ds, test_ds, class_map = build_datasets(cfg)
    eval_ds = test_ds or val_ds

    # Load model
    model = keras.models.load_model(cfg.checkpoint)

    # Predict
    y_true = np.concatenate([y.numpy() for _, y in eval_ds])
    y_pred_prob = model.predict(eval_ds, verbose=0)
    y_pred = y_pred_prob.argmax(axis=1)

    print(keras.utils.plot_model(model, show_shapes=True, dpi=72))

    # Reports
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix shape:", cm.shape)

    # Save artifacts
    out_dir = Path(DEFAULTS.out_dir) / (args.checkpoint.split("/")[-3] if "runs" in args.checkpoint else "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")


if __name__ == "__main__":
    main()
