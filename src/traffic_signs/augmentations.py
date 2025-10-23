#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: augmentations.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Data preprocessing and augmentation layers using tf.keras.layers.

Usage: 
Imported in data.py and model.py.

Notes: 
- Keeps augmentations lightweight for portability
===================================================================
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def build_augment(img_size: int, color_jitter: float, rotation: float, translate: float, zoom: float):
    layers = [
        keras.layers.Resizing(img_size, img_size),
        keras.layers.Rescaling(1.0 / 255.0),
    ]
    if color_jitter > 0:
        layers.append(keras.layers.RandomBrightness(factor=color_jitter))
        layers.append(keras.layers.RandomContrast(factor=color_jitter))
    if rotation > 0:
        layers.append(keras.layers.RandomRotation(factor=rotation / 180.0))
    if translate > 0:
        layers.append(keras.layers.RandomTranslation(translate, translate))
    if zoom > 0:
        layers.append(keras.layers.RandomZoom((-zoom, zoom)))
    return keras.Sequential(layers, name="augment")


def build_preprocess(img_size: int):
    return keras.Sequential(
        [
            keras.layers.Resizing(img_size, img_size),
            keras.layers.Rescaling(1.0 / 255.0),
        ],
        name="preprocess",
    )
