#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: train.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Training entrypoint. Builds datasets, model, optimizers, and runs training.

Usage: 
python -m traffic_signs.train --data_dir /path/to/GTSRB --epochs 30 --batch_size 64

Notes: 
- Supports mixed precision and cosine decay
===================================================================
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from .config import Config, DEFAULTS
from .data import build_datasets
from .model import build_cnn
from .callbacks import build_callbacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Traffic Signs CNN")
    p.add_argument("--data_dir", type=str, default=DEFAULTS.data_dir)
    p.add_argument("--train_dir", type=str, default=DEFAULTS.train_dir)
    p.add_argument("--val_dir", type=str, default=DEFAULTS.val_dir)
    p.add_argument("--test_dir", type=str, default=DEFAULTS.test_dir)

    p.add_argument("--img_size", type=int, default=DEFAULTS.img_size)
    p.add_argument("--channels", type=int, default=DEFAULTS.channels)
    p.add_argument("--num_classes", type=int, default=DEFAULTS.num_classes)
    p.add_argument("--batch_size", type=int, default=DEFAULTS.batch_size)
    p.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    p.add_argument("--seed", type=int, default=DEFAULTS.seed)
    p.add_argument("--mixed_precision", action="store_true", default=DEFAULTS.mixed_precision)

    p.add_argument("--learning_rate", type=float, default=DEFAULTS.learning_rate)
    p.add_argument("--weight_decay", type=float, default=DEFAULTS.weight_decay)

    p.add_argument("--augment", action="store_true", default=DEFAULTS.augment)
    p.add_argument("--color_jitter", type=float, default=DEFAULTS.color_jitter)
    p.add_argument("--rotation_deg", type=float, default=DEFAULTS.rotation_deg)
    p.add_argument("--translate", type=float, default=DEFAULTS.translate)
    p.add_argument("--zoom", type=float, default=DEFAULTS.zoom)

    p.add_argument("--exp_name", type=str, default=DEFAULTS.exp_name)
    p.add_argument("--out_dir", type=str, default=DEFAULTS.out_dir)
    p.add_argument("--monitor", type=str, default=DEFAULTS.monitor)
    p.add_argument("--patience", type=int, default=DEFAULTS.patience)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(**vars(args))

    # Mixed precision
    if cfg.mixed_precision:
        from tensorflow.keras import mixed_precision as mp

        policy = mp.Policy("mixed_float16")
        mp.set_global_policy(policy)

    # Seed
    tf.keras.utils.set_random_seed(cfg.seed)

    # Data
    train_ds, val_ds, test_ds, class_map = build_datasets(cfg)

    # Model
    model = build_cnn(cfg.img_size, cfg.channels, cfg.num_classes)

    # Optimizer with weight decay
    optimizer = keras.optimizers.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
    ]
    loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Callbacks
    out_dir = cfg.artifacts_dir()
    callbacks = build_callbacks(out_dir, cfg.monitor, cfg.patience)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "last.h5")
    with (out_dir / "config.json").open("w") as f:
        json.dump(cfg.as_dict(), f, indent=2)

    print("Training complete. Best checkpoint saved to:", out_dir / "checkpoints" / "best.h5")


if __name__ == "__main__":
    main()
