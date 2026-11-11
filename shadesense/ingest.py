"""Module 1: load images, reject unusable ones, split by identity, write a manifest + data-quality report.

Manifest columns: image_id, path, identity, condition, is_reference, split, width, height,
blur_var, clip_fraction, valid, reason, [mst], [undertone]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage

from .colour import MST_HEX, hex_to_rgb, lab_to_hex, lab_to_rgb, mst_from_lab, rgb_to_lab, undertone_rule
from .config import paths
from .lighting import relight

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """RGB uint8, EXIF rotation applied, longest side resized to `size`."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if size and max(img.size) > size:
        img.thumbnail((size, size), Image.LANCZOS)
    return np.asarray(img)


def quality(rgb: np.ndarray) -> dict:
    gray = rgb.astype(np.float64).mean(-1)
    return {
        "height": rgb.shape[0], "width": rgb.shape[1],
        "blur_var": float(ndimage.laplace(gray).var()),
        "clip_fraction": float(((rgb.max(-1) >= 254) | (rgb.max(-1) <= 1)).mean()),
        "mean_intensity": float(gray.mean()),
    }


def scan(images_dir: Path, reference_condition: str) -> pd.DataFrame:
    """<identity>/<condition>.<ext>; files directly in images_dir are treated as one identity each."""
    rows = []
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        rel = p.relative_to(images_dir)
        identity = rel.parts[0] if len(rel.parts) > 1 else p.stem
        condition = p.stem if len(rel.parts) > 1 else reference_condition
        rows.append({"image_id": str(rel.with_suffix("")), "path": str(p), "identity": identity, "condition": condition})
    if not rows:
        raise FileNotFoundError(f"no images under {images_dir}")
    df = pd.DataFrame(rows)
    df["is_reference"] = df.condition == reference_condition
    return df


def split_by_identity(df: pd.DataFrame, test_fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(df.identity.unique()))
    test = set(rng.choice(ids, max(1, round(len(ids) * test_fraction)), replace=False))
    df = df.copy()
    df["split"] = np.where(df.identity.isin(test), "test", "train")
    return df


def assert_identity_disjoint(df: pd.DataFrame) -> None:
    leaked = df.groupby("identity").split.nunique()
    if (leaked > 1).any():
        raise AssertionError(f"identities in both splits: {leaked[leaked > 1].index[:5].tolist()}")


def ingest(cfg: dict, face_detector=None) -> pd.DataFrame:
    ds, ic = cfg["dataset"], cfg["ingest"]
    p = paths(cfg)
    df = split_by_identity(scan(Path(ds["images_dir"]), ds["reference_condition"]), ds["test_fraction"], cfg["seed"])
    assert_identity_disjoint(df)

    stats = []
    for row in df.itertuples():
        try:
            rgb = load_image(row.path, ds["image_size"])
        except Exception as e:
            stats.append({"valid": False, "reason": f"unreadable: {type(e).__name__}"})
            continue
        q = quality(rgb)
        reason = ""
        if min(q["height"], q["width"]) < ic["min_side"]:
            reason = "too_small"
        elif q["blur_var"] < ic["min_blur_var"]:
            reason = "blurry"
        elif q["clip_fraction"] > ic["max_clip_fraction"]:
            reason = "bad_exposure"
        elif ic["require_face"] and face_detector is not None and face_detector(rgb, row.image_id) is None:
            reason = "no_face"
        stats.append({**q, "valid": not reason, "reason": reason})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(stats)], axis=1)

    for col, key in (("mst", "mst_labels"), ("undertone", "undertone_labels")):
        if ds.get(key):
            df = df.merge(pd.read_csv(ds[key])[["image_id", col]], on="image_id", how="left")

    p["manifest"].parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p["manifest"], index=False)
    df[~df.valid][["image_id", "path", "reason"]].to_csv(p["rejections"], index=False)
    v = df[df.valid]
    report = {
        "n_images": len(df), "n_valid": int(df.valid.sum()),
        "rejected_by_reason": df[~df.valid].reason.value_counts().to_dict(),
        "n_identities": int(v.identity.nunique()),
        "identities_per_split": v.groupby("split").identity.nunique().to_dict(),
        "images_per_split": v.split.value_counts().to_dict(),
        "conditions": v.condition.value_counts().to_dict(),
        "identities_without_reference": sorted(set(v.identity) - set(v[v.is_reference].identity)),
        "resolution_median": [float(v.width.median()), float(v.height.median())] if len(v) else None,
        "blur_var_median": float(v.blur_var.median()) if len(v) else None,
    }
    if "mst" in v:
        report["mst_counts"] = v.mst.value_counts().sort_index().to_dict()
    p["quality"].write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return df


def load_manifest(cfg: dict, valid_only: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(paths(cfg)["manifest"])
    return df[df.valid].reset_index(drop=True) if valid_only else df


# --------------------------------------------------------------------------- synthetic demo data
def _ellipse(h, w, cy, cx, ry, rx):
    yy, xx = np.mgrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1


# Photos per synthetic person: (condition, mean light gains, exposure range, white card in shot?)
SYNTH_CONDITIONS = [
    ("neutral", (1.0, 1.0, 1.0), (0.85, 1.0), True),       # reference: daylight + white card, mild camera cast
    ("daylight", (1.0, 1.0, 1.0), (0.85, 1.0), False),
    ("warm_lamp", (1.12, 1.0, 0.82), (0.7, 0.95), False),
    ("cool_led", (0.88, 1.0, 1.12), (0.8, 1.0), False),
]


def make_synthetic(out_dir: str | Path, masks_dir: str | Path, n_identities: int = 40, size: int = 256, seed: int = 0) -> pd.DataFrame:
    """Cartoon faces with known skin colour and masks, spread across all 10 MST tones.

    Each person gets the photos in SYNTH_CONDITIONS: a reference photo with a white card next to the
    face and a slightly wrong camera white balance, plus photos under other lights without the card.

    Lets the whole pipeline (oracle segmentation, lighting, features, matching, evaluation) run on a
    laptop with no downloads. It is a plumbing test, not evidence about real faces.
    """
    out_dir, masks_dir = Path(out_dir), Path(masks_dir)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_identities):
        mst = i % 10
        base = rgb_to_lab(hex_to_rgb(MST_HEX[mst])[None, None])[0, 0]
        skin_lab = base + rng.normal(0, [2.0, 2.5, 3.0])          # individual variation, incl. undertone
        skin_rgb = lab_to_rgb(skin_lab[None, None])[0, 0].astype(np.float64)
        h = w = size
        # near-neutral background with a slight tint (typical indoor wall); gray-world assumes this
        img = np.empty((h, w, 3)); img[:] = rng.uniform(70, 200) + rng.normal(0, 2, 3)
        face = _ellipse(h, w, h * 0.5, w * 0.5, h * 0.36, w * 0.27)
        shade = 1 - 0.18 * ((np.mgrid[:h, :w][1] - w / 2) / (w * 0.27)) ** 2   # soft side shading
        tex = rng.normal(0, 4, (h, w, 1))
        img[face] = (skin_rgb * shade[..., None] + tex)[face]
        eyes = _ellipse(h, w, h * 0.42, w * 0.40, h * 0.025, w * 0.05) | _ellipse(h, w, h * 0.42, w * 0.60, h * 0.025, w * 0.05)
        brows = _ellipse(h, w, h * 0.36, w * 0.40, h * 0.012, w * 0.07) | _ellipse(h, w, h * 0.36, w * 0.60, h * 0.012, w * 0.07)
        lips = _ellipse(h, w, h * 0.68, w * 0.5, h * 0.035, w * 0.10)
        img[eyes] = [35, 30, 30]
        img[brows] = [55, 40, 35]
        lip_rgb = np.clip(skin_rgb * [0.85, 0.55, 0.6] + [30, 0, 5], 0, 255)
        img[lips] = lip_rgb + rng.normal(0, 4, (lips.sum(), 3))
        skin_mask = face & ~ndimage.binary_dilation(eyes | brows | lips, iterations=3)

        card = np.zeros((h, w), bool)
        card[int(h * 0.40):int(h * 0.62), int(w * 0.01):int(w * 0.085)] = True   # beside the face, outside it

        ident = f"person{i:03d}"
        (out_dir / ident).mkdir(parents=True, exist_ok=True)
        (masks_dir / ident).mkdir(parents=True, exist_ok=True)
        for cond, gains, (e_lo, e_hi), with_card in SYNTH_CONDITIONS:
            scene = img.copy()
            if with_card:
                scene[card] = 231.0 + rng.normal(0, 1.0, (card.sum(), 3))    # sRGB 231 = linear 0.8, white paper
            g = np.asarray(gains) * (1 + rng.normal(0, 0.04, 3))
            photo = relight(np.clip(scene, 0, 255).astype(np.uint8), g, rng.uniform(e_lo, e_hi))
            Image.fromarray(photo).save(out_dir / ident / f"{cond}.png")
            Image.fromarray(skin_mask.astype(np.uint8) * 255).save(masks_dir / ident / f"{cond}_skin.png")
            Image.fromarray(lips.astype(np.uint8) * 255).save(masks_dir / ident / f"{cond}_lips.png")
            rows.append({"image_id": f"{ident}/{cond}", "mst": mst_from_lab(skin_lab),
                         "undertone": undertone_rule(skin_lab), "true_hex": lab_to_hex(skin_lab)})
    labels = pd.DataFrame(rows)
    labels.to_csv(out_dir.parent / f"{out_dir.name}_labels.csv", index=False)
    return labels


def make_demo_catalogue(path: str | Path, n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic foundation catalogue covering the MST range. Brand names are fake (DemoBrand A/B/C)."""
    rng = np.random.default_rng(seed)
    mst_lab = np.stack([rgb_to_lab(hex_to_rgb(h)[None, None])[0, 0] for h in MST_HEX])
    t = np.sort(rng.uniform(0, 9, n))
    lab = np.stack([np.interp(t, np.arange(10), mst_lab[:, j]) for j in range(3)], 1) + rng.normal(0, [1.5, 2.0, 2.5], (n, 3))
    df = pd.DataFrame({"brand": [f"DemoBrand {'ABC'[i % 3]}" for i in range(n)],
                       "product": "Demo Foundation", "shade": [f"{100 + i * 4}" for i in range(n)],
                       "hex": [lab_to_hex(x) for x in lab]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
