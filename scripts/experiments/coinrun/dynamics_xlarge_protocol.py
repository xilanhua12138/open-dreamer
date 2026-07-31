#!/usr/bin/env python3
"""Frozen CR-DYN-0012 capacity-only extension of CR-DYN-0011."""

from __future__ import annotations


XLARGE_ARCHITECTURE = {
    "depth": 9,
    "d_model": 640,
    "n_heads": 10,
    "n_kv_heads": 1,
    "n_register": 32,
}

XLARGE_PROTOCOL = {
    "parameters": 52_801_152,
    "batch_size": 16,
    "short_T": 64,
    "long_T": 128,
    "long_ratio": 0.1,
    "context_length": 128,
    "k_max": 256,
    "bootstrap_start": 100_000,
    "bootstrap_fraction": 0.25,
    "max_steps": 200_000,
}
