"""Module 3: lighting correction, plus the lighting changes and corruptions used to test robustness."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _as_float(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) / 255.0 if rgb.dtype == np.uint8 else rgb.astype(np.float64)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)


def shades_of_gray(rgb: np.ndarray, p: float = 6, mask: np.ndarray | None = None, max_gain: float = 1.25) -> np.ndarray:
    """White balance assuming the scene's p-norm average is gray. p=1 is gray-world; large p approaches white-patch.

    Works in linear RGB so the per-channel gains behave like a change of light. `mask` limits which
    pixels estimate the illuminant (default: all non-saturated pixels). Gains are capped at
    [1/max_gain, max_gain]: a strongly coloured scene (red wall, flag, hair) otherwise gets "corrected"
    into a wrong skin colour.
    """
    lin = _srgb_to_linear(_as_float(rgb))
    sel = np.ones(lin.shape[:2], bool) if mask is None else mask.copy()
    sel &= lin.max(-1) < 0.98            # saturated pixels carry no information about the light
    px = lin[sel] if sel.any() else lin.reshape(-1, 3)
    illum = np.power(np.mean(np.power(px, p), axis=0), 1 / p) + 1e-6
    gains = np.clip(illum.mean() / illum, 1 / max_gain, max_gain)
    return _to_uint8(_linear_to_srgb(lin * gains))


def gray_world(rgb: np.ndarray, mask: np.ndarray | None = None, max_gain: float = 1.25) -> np.ndarray:
    return shades_of_gray(rgb, p=1, mask=mask, max_gain=max_gain)


def correct(rgb: np.ndarray, method: str = "shades_of_gray", p: float = 6, mask: np.ndarray | None = None,
            max_gain: float = 1.25) -> np.ndarray:
    """`mask`: pixels used to estimate the light. Pass the area OUTSIDE the face: skin itself is not gray,
    and including it pulls every face toward neutral (removing real warmth)."""
    if method == "none":
        return rgb
    if method == "gray_world":
        return gray_world(rgb, mask, max_gain)
    if method == "shades_of_gray":
        return shades_of_gray(rgb, p, mask, max_gain)
    raise ValueError(f"unknown lighting method {method!r}")


def relight(rgb: np.ndarray, gains=(1.0, 1.0, 1.0), brightness: float = 1.0) -> np.ndarray:
    """Simulate a different light source: per-channel gains and overall intensity, applied in linear RGB."""
    lin = _srgb_to_linear(_as_float(rgb))
    return _to_uint8(_linear_to_srgb(lin * np.asarray(gains) * brightness))


def corrupt(rgb: np.ndarray, spec: dict) -> np.ndarray:
    kind = spec["kind"]
    img = Image.fromarray(rgb)
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=spec.get("quality", 30))
        return np.asarray(Image.open(buf).convert("RGB"))
    if kind == "blur":
        return np.asarray(img.filter(ImageFilter.GaussianBlur(spec.get("sigma", 2.0))))
    raise ValueError(f"unknown corruption {kind!r}")


# --------------------------------------------------------------------------- white-card calibration
def find_card(rgb: np.ndarray, background: np.ndarray, min_area: float = 0.002, top_fraction: float = 0.10,
              box: tuple[float, float, float, float] | None = None) -> np.ndarray | None:
    """Mask of the white card / paper held next to the face in a reference photo.

    `box` (x0, y0, x1, y1 as fractions of width/height) gives the card position by hand. Otherwise the card
    is the largest connected bright region outside the face: among the brightest `top_fraction` of background
    pixels (and at least 25% brighter than the background median), ignoring clipped ones. Returns None when nothing large enough is found.
    """
    from scipy import ndimage

    h, w = rgb.shape[:2]
    if box is not None:
        x0, y0, x1, y1 = box
        m = np.zeros((h, w), bool)
        m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
        m &= rgb.max(-1) < 250
    else:
        lum = _srgb_to_linear(_as_float(rgb)) @ np.array([0.2126, 0.7152, 0.0722])
        cand = background & (rgb.max(-1) < 250)
        if cand.sum() < min_area * h * w:
            return None
        # bright AND clearly brighter than the typical background (a plain wall is otherwise all "top 10%")
        thr = max(np.quantile(lum[cand], 1 - top_fraction), 1.25 * np.median(lum[cand]))
        lab, n = ndimage.label(cand & (lum > thr))
        if n == 0:
            return None
        sizes = ndimage.sum(np.ones_like(lum), lab, index=np.arange(1, n + 1))
        m = lab == (np.argmax(sizes) + 1)
        m = ndimage.binary_erosion(m, iterations=2) if m.sum() > 200 else m
    return m if m.sum() >= min_area * h * w else None


def card_calibrate(rgb: np.ndarray, card: np.ndarray, white: float = 0.8, fix_brightness: bool = True) -> np.ndarray:
    """Rescale linear RGB so the card becomes neutral white (`white` = its linear value under daylight).

    The card has a known colour, so this removes both the light colour and the camera's white balance /
    exposure. That makes the reference colour independent of the methods being tested (no circular
    ground truth). With fix_brightness=False only the colour cast is removed and exposure is kept.
    """
    lin = _srgb_to_linear(_as_float(rgb))
    c = np.median(lin[card], axis=0) + 1e-6
    gains = (white / c) if fix_brightness else (c.mean() / c)
    return _to_uint8(_linear_to_srgb(lin * gains))
