"""Module 4: skin-tone features.

Baseline:  median CIELAB colour of the skin pixels (3 numbers, no training).
Proposed:  frozen DINOv2 embedding of the masked skin crop, mapped to CIELAB by a small ridge
           regression head trained to predict the reference-light colour. Both end in a LAB colour,
           so the same matcher serves both and any difference comes from the representation.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .colour import rgb_to_lab

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def skin_lab(rgb: np.ndarray, skin: np.ndarray, trim=(10, 90)) -> np.ndarray:
    """Median LAB of skin pixels, ignoring the darkest and brightest (shadow, highlight) by L*."""
    lab = rgb_to_lab(rgb[skin])
    if len(lab) == 0:
        raise ValueError("empty skin mask")
    lo, hi = np.percentile(lab[:, 0], trim)
    keep = (lab[:, 0] >= lo) & (lab[:, 0] <= hi)
    return np.median(lab[keep] if keep.sum() >= 10 else lab, axis=0)


def skin_crop(rgb: np.ndarray, skin: np.ndarray, size: int = 224) -> np.ndarray:
    """Square crop around the skin mask; non-skin pixels set to mid-gray so DINOv2 sees only skin."""
    ys, xs = np.nonzero(skin)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
    crop = rgb[y0:y0 + side, x0:x0 + side].copy()
    crop[~skin[y0:y0 + side, x0:x0 + side]] = 128
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.BICUBIC))


class DinoFeaturizer:
    def __init__(self, hf_id: str = "facebook/dinov2-small", size: int = 224, device: str = "cpu"):
        import torch
        from transformers import AutoModel

        self.torch, self.size, self.device = torch, size, device
        self.model = AutoModel.from_pretrained(hf_id).eval().to(device)
        self.dim = self.model.config.hidden_size

    def __call__(self, crops: list[np.ndarray]) -> np.ndarray:
        """crops: list of uint8 [size,size,3] -> [N, D] CLS embeddings."""
        x = (np.stack(crops).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = self.torch.as_tensor(x, dtype=self.torch.float32).permute(0, 3, 1, 2).to(self.device)
        with self.torch.inference_mode():
            return self.model(pixel_values=x).pooler_output.cpu().numpy()


def make_colour_head(alpha: float = 1.0):
    """DINOv2 embedding -> CIELAB under reference light."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
