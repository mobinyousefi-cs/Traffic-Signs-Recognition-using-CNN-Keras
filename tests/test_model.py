#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: test_model.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Unit tests for model construction and a forward pass with dummy data.

Usage: 
pytest -q

Notes: 
- Ensures model compiles and predicts with correct shapes
===================================================================
"""
import numpy as np
from tensorflow import keras

from traffic_signs.model import build_cnn


def test_build_and_forward():
    model = build_cnn(img_size=48, channels=3, num_classes=43)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    x = np.random.rand(2, 48, 48, 3).astype("float32")
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 43)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-4)
