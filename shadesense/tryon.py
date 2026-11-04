"""Module 7: virtual lipstick try-on by recolouring the lip mask in CIELAB.

Shifts the lips' average colour toward the lipstick while keeping each pixel's deviation from that
average, so texture, creases and highlights survive. A feathered mask avoids hard edges.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .colour import hex_to_lab, lab_to_rgb, rgb_to_lab


def apply_lipstick(rgb: np.ndarray, lips: np.ndarray, hex_colour: str, strength: float = 0.75,
                   feather_px: int = 3) -> np.ndarray:
    if lips.sum() == 0:
        return rgb.copy()
    lab = rgb_to_lab(rgb)
    target = hex_to_lab(hex_colour)
    mean = lab[lips].mean(0)
    shifted = lab.copy()
    shifted[..., 0] += 0.7 * (target[0] - mean[0])     # keep some of the natural lightness variation
    shifted[..., 1:] += target[1:] - mean[1:]
    m = lips.astype(np.float64)
    # full strength inside the lips, soft fall-off only outside the edge (thin lips would otherwise wash out)
    alpha = np.maximum(m, ndimage.gaussian_filter(m, feather_px)) if feather_px else m
    alpha = (np.clip(alpha, 0, 1) * strength)[..., None]
    out = lab * (1 - alpha) + shifted * alpha
    return lab_to_rgb(out)
