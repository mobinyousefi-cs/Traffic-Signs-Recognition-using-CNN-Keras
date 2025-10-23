#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: test_config.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Unit tests for configuration object.

Usage: 
pytest -q

Notes: 
- Ensures defaults are sane
===================================================================
"""
from traffic_signs.config import DEFAULTS, Config


def test_defaults_valid():
    assert DEFAULTS.img_size > 0
    assert DEFAULTS.num_classes == 43
    assert DEFAULTS.batch_size > 0


def test_override():
    c = Config(img_size=64, num_classes=10)
    assert c.img_size == 64
    assert c.num_classes == 10
