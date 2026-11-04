"""Colour helpers shared by every module: hex/RGB/CIELAB conversion, Delta E 2000, Monk Skin Tone, undertone."""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab

# Monk Skin Tone scale, MST 1 (lightest) to MST 10 (deepest). Reference swatches published by Google
# (skintone.google); double-check against the official page before the final report.
MST_HEX = ["#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
           "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"]


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(v))) for v in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_lab(rgb) -> np.ndarray:
    """uint8 or float [0,1] RGB, any shape (...,3) -> CIELAB (D65)."""
    rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float64) / 255.0
    return rgb2lab(rgb)


def lab_to_rgb(lab) -> np.ndarray:
    """CIELAB (...,3) -> uint8 RGB."""
    return np.round(np.clip(lab2rgb(np.asarray(lab, dtype=np.float64)), 0, 1) * 255).astype(np.uint8)


def hex_to_lab(h: str) -> np.ndarray:
    return rgb_to_lab(hex_to_rgb(h)[None, None])[0, 0]


def lab_to_hex(lab) -> str:
    return rgb_to_hex(lab_to_rgb(np.asarray(lab)[None, None])[0, 0])


def delta_e(lab1, lab2) -> np.ndarray:
    """Perceptual colour difference (CIEDE2000). Below ~2 is hard for people to see. Broadcasts."""
    a, b = np.broadcast_arrays(np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64))
    return deltaE_ciede2000(a, b)


MST_LAB = np.stack([hex_to_lab(h) for h in MST_HEX])


def mst_from_lab(lab) -> int:
    """Nearest Monk Skin Tone swatch by Delta E 2000 (1-10)."""
    return int(np.argmin(delta_e(MST_LAB, np.asarray(lab)[None])) + 1)


def hue_angle(lab) -> float:
    """CIELAB hue angle in degrees: ~0 = red/pink, ~90 = yellow."""
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)


def undertone_rule(lab, cool_below: float = 55.0, warm_above: float = 65.0) -> str:
    """Rule-based undertone from the skin hue angle (the baseline; replace with a trained classifier when labels exist)."""
    h = hue_angle(lab)
    if h < cool_below:
        return "cool"
    if h > warm_above:
        return "warm"
    return "neutral"
