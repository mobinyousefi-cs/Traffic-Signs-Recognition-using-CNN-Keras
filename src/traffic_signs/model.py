#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: model.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
CNN architectures for GTSRB. Includes a compact SE-CNN baseline.

Usage: 
from traffic_signs.model import build_cnn

Notes: 
- Keeps parameter count < 2M for fast inference
===================================================================
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


def se_block(x: keras.layers.Layer, ratio: int = 8):
    filters = x.shape[-1]
    se = keras.layers.GlobalAveragePooling2D()(x)
    se = keras.layers.Dense(max(filters // ratio, 8), activation="relu")(se)
    se = keras.layers.Dense(filters, activation="sigmoid")(se)
    se = keras.layers.Reshape((1, 1, filters))(se)
    return keras.layers.Multiply()([x, se])


def conv_bn(x, filters, k=3, s=1, act=True):
    x = keras.layers.Conv2D(filters, k, s, padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    if act:
        x = keras.layers.Activation("relu")(x)
    return x


def residual_block(x, filters, downsample=False):
    shortcut = x
    s = 2 if downsample else 1
    x = conv_bn(x, filters, 3, s)
    x = conv_bn(x, filters, 3, 1, act=False)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = conv_bn(shortcut, filters, 1, s, act=False)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation("relu")(x)
    x = se_block(x)
    return x


def build_cnn(img_size: int = 48, channels: int = 3, num_classes: int = 43) -> keras.Model:
    inputs = keras.Input((img_size, img_size, channels))

    x = conv_bn(inputs, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = keras.layers.SpatialDropout2D(0.2)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="se_res_cnn")
    return model
