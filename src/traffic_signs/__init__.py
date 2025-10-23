#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================== 
Project: Traffic Signs Recognition (GTSRB) 
File: __init__.py 
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs) 
Created: 2025-10-23 
Updated: 2025-10-23 
License: MIT License (see LICENSE file for details)
=================================================================== 

Description: 
Package initializer for the traffic_signs module.

Usage: 
Imported implicitly when using the package.

Notes: 
- Exposes key functions for convenience
===================================================================
"""
from .config import Config
from .model import build_cnn
from .train import main as train_main
from .evaluate import main as evaluate_main
from .infer import main as infer_main

__all__ = [
    "Config",
    "build_cnn",
    "train_main",
    "evaluate_main",
    "infer_main",
]
