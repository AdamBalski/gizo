#!/usr/bin/env python3
"""Shared morphological operations built on Pillow filters."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _ensure_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    return mask.astype(bool)


def _apply_filter(mask: np.ndarray, flt: ImageFilter.Filter) -> np.ndarray:
    img = Image.fromarray(_ensure_bool(mask).astype(np.uint8) * 255, mode="L")
    filtered = img.filter(flt)
    return np.asarray(filtered) > 0


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = _ensure_bool(mask)
    for _ in range(iterations):
        result = _apply_filter(result, ImageFilter.MaxFilter(3))
    return result


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = _ensure_bool(mask)
    for _ in range(iterations):
        result = _apply_filter(result, ImageFilter.MinFilter(3))
    return result


def binary_close(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_erode(binary_dilate(mask, iterations), iterations)


def binary_open(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, iterations), iterations)
