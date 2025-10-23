#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: config.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Centralized configuration utilities and defaults for training, evaluation,
and inference. CLI flags override these defaults.

Usage: 
python -m traffic_signs.train --help

Notes: 
- Designed to be a single source of truth
===================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Paths
    data_dir: Optional[str] = None
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None

    # Training
    img_size: int = 48
    channels: int = 3
    num_classes: int = 43
    batch_size: int = 64
    epochs: int = 30
    seed: int = 42
    mixed_precision: bool = False

    # Optimizer / LR
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cosine_decay: bool = True

    # Augmentations
    augment: bool = True
    color_jitter: float = 0.2
    rotation_deg: float = 10.0
    translate: float = 0.1
    zoom: float = 0.1

    # Logging / Artifacts
    exp_name: str = "baseline"
    out_dir: str = "runs"

    # Checkpointing
    save_best_only: bool = True
    monitor: str = "val_accuracy"
    patience: int = 7

    # Inference
    class_map_json: Optional[str] = None
    checkpoint: Optional[str] = None

    def artifacts_dir(self) -> Path:
        return Path(self.out_dir) / self.exp_name

    def checkpoints_dir(self) -> Path:
        return self.artifacts_dir() / "checkpoints"

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULTS = Config()
