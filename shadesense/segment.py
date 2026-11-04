"""Module 2: find the face and cut out skin and lip masks.

Three interchangeable backends, all returning FaceMasks:
  landmarks  MediaPipe Face Landmarker (478 points) -> polygons. Fast, no training. (baseline)
  parser     SegFormer face parser trained on CelebAMask-HQ -> per-pixel classes. (proposed)
  oracle     ground-truth masks from disk, to evaluate later stages without segmentation errors.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# MediaPipe Face Mesh landmark indices (canonical topology).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


@dataclass
class FaceMasks:
    skin: np.ndarray   # bool [H, W]
    lips: np.ndarray   # bool [H, W]

    @property
    def background(self) -> np.ndarray:
        """Pixels well away from the face: used to estimate the light colour."""
        ys, xs = np.nonzero(self.skin | self.lips)
        out = np.ones(self.skin.shape, bool)
        pad_y, pad_x = (ys.max() - ys.min()) // 4, (xs.max() - xs.min()) // 4
        out[max(0, ys.min() - pad_y): ys.max() + pad_y, max(0, xs.min() - pad_x): xs.max() + pad_x] = False
        return out if out.sum() > 0.05 * out.size else ~(self.skin | self.lips)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.skin | self.lips)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def refine_lips(rgb: np.ndarray, lips: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Clean a lip mask so try-on paints lips only.

    Drops teeth (bright, nearly colourless) and surrounding skin (lips are redder: higher a* than the
    person's own skin). Falls back to the teeth-only cleanup if the colour test would remove most of the mask.
    """
    from .colour import rgb_to_lab

    if not lips.any():
        return lips
    lab = rgb_to_lab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    no_teeth = lips & ~((lab[..., 0] > 60) & (chroma < 18))
    if not skin.any():
        return no_teeth
    redder = no_teeth & (lab[..., 1] > np.median(lab[skin][:, 1]) + 4)
    redder = ndimage.binary_closing(redder, iterations=2) & no_teeth
    out = redder if redder.sum() >= 0.3 * no_teeth.sum() else no_teeth
    return _largest_parts(out)


def _largest_parts(mask: np.ndarray, keep: float = 0.15) -> np.ndarray:
    """Drop specks: keep connected parts at least `keep` x the size of the biggest (upper + lower lip survive)."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= keep * sizes.max()))


def polygon_mask(shape, points) -> np.ndarray:
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(map(float, p)) for p in points], fill=1)
    return np.asarray(img).astype(bool)


def masks_from_landmarks(shape, pts: np.ndarray, erode_px: int = 3) -> FaceMasks:
    """pts: [N, 2] pixel coordinates in MediaPipe Face Mesh order."""
    face = polygon_mask(shape, pts[FACE_OVAL])
    lips = polygon_mask(shape, pts[LIPS_OUTER]) & ~polygon_mask(shape, pts[LIPS_INNER])
    holes = np.zeros(shape[:2], bool)
    for idx in (LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW, LIPS_OUTER):
        holes |= polygon_mask(shape, pts[idx])
    holes = ndimage.binary_dilation(holes, iterations=max(1, erode_px * 2))
    skin = face & ~holes
    if erode_px:
        skin = ndimage.binary_erosion(skin, iterations=erode_px)
    return FaceMasks(skin=skin, lips=lips)


class LandmarkSegmenter:
    def __init__(self, url: str, models_dir: str | Path, erode_px: int = 3, **_):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(models_dir) / "face_landmarker.task"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[segment] downloading MediaPipe face landmarker -> {path}")
            urllib.request.urlretrieve(url, path)
        opts = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(path)), num_faces=1)
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.erode_px = erode_px

    def __call__(self, rgb: np.ndarray, image_id: str | None = None) -> FaceMasks | None:
        res = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
        m = masks_from_landmarks(rgb.shape, pts, self.erode_px)
        m.lips = refine_lips(rgb, m.lips, m.skin)
        return m


class ParserSegmenter:
    SKIN, LIPS = ("skin", "nose"), ("u_lip", "l_lip")

    def __init__(self, hf_id: str, erode_px: int = 3, **_):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        self.proc = SegformerImageProcessor.from_pretrained(hf_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(hf_id).eval()
        names = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.skin_ids = [names[n] for n in self.SKIN if n in names]
        self.lip_ids = [names[n] for n in self.LIPS if n in names]
        if not self.skin_ids or not self.lip_ids:
            raise ValueError(f"unexpected label names in {hf_id}: {sorted(names)}")
        self.erode_px = erode_px

    def __call__(self, rgb: np.ndarray, image_id: str | None = None) -> FaceMasks | None:
        torch = self.torch
        with torch.inference_mode():
            logits = self.model(**self.proc(images=Image.fromarray(rgb), return_tensors="pt")).logits
            labels = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear",
                                                     align_corners=False).argmax(1)[0].numpy()
        skin, lips = np.isin(labels, self.skin_ids), np.isin(labels, self.lip_ids)
        if skin.sum() < 500:
            return None
        if self.erode_px:
            skin = ndimage.binary_erosion(skin, iterations=self.erode_px)
        return FaceMasks(skin=skin, lips=refine_lips(rgb, lips, skin))


class OracleSegmenter:
    """Reads <masks_dir>/<image_id>_skin.png and _lips.png."""

    def __init__(self, masks_dir: str | Path, **_):
        if not masks_dir:
            raise ValueError("oracle segmentation needs dataset.masks_dir")
        self.dir = Path(masks_dir)

    def load(self, image_id: str) -> FaceMasks | None:
        sp, lp = self.dir / f"{image_id}_skin.png", self.dir / f"{image_id}_lips.png"
        if not sp.exists():
            return None
        return FaceMasks(skin=np.asarray(Image.open(sp)) > 127, lips=np.asarray(Image.open(lp)) > 127)

    def __call__(self, rgb: np.ndarray, image_id: str | None = None) -> FaceMasks | None:
        return self.load(image_id) if image_id else None


def load_segmenter(cfg: dict, backend: str | None = None):
    s = cfg["segment"]
    backend = backend or s["backend"]
    if backend == "landmarks":
        return LandmarkSegmenter(s["landmarker_url"], cfg["paths"]["models"], s["erode_px"])
    if backend == "parser":
        return ParserSegmenter(s["parser_hf_id"], s["erode_px"])
    if backend == "oracle":
        return OracleSegmenter(cfg["dataset"]["masks_dir"])
    raise ValueError(f"unknown segment backend {backend!r}")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0
