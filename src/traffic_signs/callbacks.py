#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: callbacks.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Factory functions for common Keras callbacks used in training.

Usage: 
from traffic_signs.callbacks import build_callbacks

Notes: 
- Includes ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, LR scheduler
===================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow import keras


def build_callbacks(out_dir: Path, monitor: str = "val_accuracy", patience: int = 7) -> List[keras.callbacks.Callback]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckp_dir = out_dir / "checkpoints"
    ckp_dir.mkdir(parents=True, exist_ok=True)
    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckp_dir / "best.h5"),
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode="max" if "acc" in monitor else "min",
        ),
        keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(str(out_dir / "train_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "tb")),
    ]

    # Cosine decay via built-in schedule or ReduceLROnPlateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
        )
    )
    return callbacks
