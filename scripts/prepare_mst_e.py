"""Arrange the MST-E dataset (Google's Monk Skin Tone Examples) for ShadeSense.

MST-E has photos of a small number of people, each labelled with a Monk Skin Tone value, taken under
many lighting conditions. Download it yourself (accept its licence), then point this script at the
image folder and the label CSV. Column names are detected; override them with the flags if needed.

    python scripts/prepare_mst_e.py --images data/raw/mst_e_download --labels data/raw/mst_e_download/labels.csv

Writes symlinks  data/raw/mst_e/<subject>/<image stem>.<ext>   (no copies, nothing leaves your laptop)
       labels    data/raw/mst_e_labels.csv                     (image_id,mst)
Then:  python -m shadesense pipeline --config configs/mst_e.yaml

MST-E has no white-card reference photo, so shade accuracy is not computed on it. It is used for the
fairness checks: Monk Skin Tone prediction accuracy (exact, within 1) per method and per skin-tone group.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _pick(cols, given, *names):
    if given:
        return given
    low = {c.lower(): c for c in cols}
    for n in names:
        for c_low, c in low.items():
            if c_low == n or n in c_low:
                return c
    raise SystemExit(f"could not find a column like {names} in {list(cols)}; pass it with a flag")


def _mst_value(v) -> float:
    m = re.search(r"\d+", str(v))
    return float(m.group()) if m else float("nan")


def prepare(images: Path, labels: Path, out: Path, labels_out: Path, image_col=None, subject_col=None, mst_col=None) -> pd.DataFrame:
    df = pd.read_csv(labels)
    image_col = _pick(df.columns, image_col, "image", "file", "path", "filename")
    mst_col = _pick(df.columns, mst_col, "mst", "monk", "skin_tone", "tone")
    try:
        subject_col = _pick(df.columns, subject_col, "subject", "person", "participant", "identity", "id")
    except SystemExit:
        subject_col = None
    files = {p.name: p for p in images.rglob("*") if p.suffix.lower() in EXTS}
    rows = []
    for r in df.itertuples(index=False):
        rec = r._asdict() if hasattr(r, "_asdict") else dict(zip(df.columns, r))
        name = Path(str(rec[image_col])).name
        src = files.get(name)
        if src is None:
            continue
        subject = str(rec[subject_col]) if subject_col else src.parent.name
        subject = re.sub(r"[^A-Za-z0-9_-]", "_", subject)
        dst = out / subject / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        rows.append({"image_id": f"{subject}/{src.stem}", "mst": _mst_value(rec[mst_col])})
    lab = pd.DataFrame(rows).dropna()
    lab["mst"] = lab.mst.astype(int)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    lab.to_csv(labels_out, index=False)
    print(f"{len(lab)} images of {lab.image_id.str.split('/').str[0].nunique()} subjects -> {out}, labels {labels_out}")
    print(lab.mst.value_counts().sort_index().to_string())
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="data/raw/mst_e")
    ap.add_argument("--labels-out", default="data/raw/mst_e_labels.csv")
    ap.add_argument("--image-col"); ap.add_argument("--subject-col"); ap.add_argument("--mst-col")
    a = ap.parse_args()
    prepare(Path(a.images), Path(a.labels), Path(a.out), Path(a.labels_out), a.image_col, a.subject_col, a.mst_col)


if __name__ == "__main__":
    main()
