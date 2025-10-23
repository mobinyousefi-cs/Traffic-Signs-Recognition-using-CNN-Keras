#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: data.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Dataset utilities using tf.data for training/validation/testing.

Usage: 
from traffic_signs.data import build_datasets

Notes: 
- Expects GTSRB layout with class subfolders
===================================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import tensorflow as tf
from tensorflow import keras

from .augmentations import build_augment, build_preprocess
from .config import Config

AUTOTUNE = tf.data.AUTOTUNE


def _class_names_from_dir(directory: str) -> Dict[int, str]:
    classes = sorted([p.name for p in Path(directory).iterdir() if p.is_dir()])
    return {i: name for i, name in enumerate(classes)}


def _count_images(directory: str) -> int:
    return sum(1 for _ in Path(directory).rglob("*.png")) + sum(1 for _ in Path(directory).rglob("*.ppm")) + sum(1 for _ in Path(directory).rglob("*.jpg"))


def build_datasets(cfg: Config):
    """Build tf.data pipelines for train/val/test based on config."""
    img_size = cfg.img_size

    # Derive subdirs if only data_dir provided
    train_dir = cfg.train_dir or (Path(cfg.data_dir) / "Train" if cfg.data_dir else None)
    val_dir = cfg.val_dir  # optional user-provided
    test_dir = cfg.test_dir or (Path(cfg.data_dir) / "Test" if cfg.data_dir else None)

    if train_dir is None or not Path(train_dir).exists():
        raise FileNotFoundError("Training directory not found. Provide --train_dir or --data_dir with Train/")

    # If no explicit val_dir, split from train
    if val_dir is None:
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=cfg.batch_size,
            validation_split=0.15,
            subset="both",
            seed=cfg.seed,
        )
        train_ds, val_ds = train_ds
    else:
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
        val_ds = keras.utils.image_dataset_from_directory(
            val_dir,
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )

    if test_dir and Path(test_dir).exists():
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            label_mode="int",
            image_size=(img_size, img_size),
            batch_size=cfg.batch_size,
            shuffle=False,
        )
    else:
        test_ds = None

    # Save class map
    class_map = _class_names_from_dir(str(train_dir))
    out_dir = cfg.artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "class_map.json").open("w") as f:
        json.dump(class_map, f, indent=2)

    # Augment / preprocess pipelines
    if cfg.augment:
        aug = build_augment(cfg.img_size, cfg.color_jitter, cfg.rotation_deg, cfg.translate, cfg.zoom)
        train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
        preprocess = tf.identity  # already scaled by augment
    else:
        preprocess_layer = build_preprocess(cfg.img_size)
        train_ds = train_ds.map(lambda x, y: (preprocess_layer(x), y), num_parallel_calls=AUTOTUNE)
        preprocess = preprocess_layer

    val_ds = val_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)

    # Cache + prefetch
    def tune(ds):
        return ds.cache().prefetch(AUTOTUNE)

    return tune(train_ds), tune(val_ds), (tune(test_ds) if test_ds is not None else None), class_map
