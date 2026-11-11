"""Turn one-row-per-person labels into the one-row-per-photo CSVs the pipeline reads.

Usage: python scripts/expand_labels.py [--people docs/templates/people.csv]
       [--images data/raw/team_selfies] [--out data/labels]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def expand(people: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    rows = [{"image_id": str(p.relative_to(images_dir).with_suffix("")), "identity": p.parent.name}
            for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in EXTS]
    photos = pd.DataFrame(rows, columns=["image_id", "identity"])
    missing = sorted(set(photos.identity) - set(people.identity))
    if missing:
        print(f"warning: no labels for {missing}")
    return photos.merge(people, on="identity", how="inner")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", default="docs/templates/people.csv")
    ap.add_argument("--images", default="data/raw/team_selfies")
    ap.add_argument("--out", default="data/labels")
    a = ap.parse_args()
    people = pd.read_csv(a.people, dtype={"identity": str})
    df = expand(people, Path(a.images))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.dropna(subset=["mst"])[["image_id", "mst"]].astype({"mst": int}).to_csv(out / "mst.csv", index=False)
    df.dropna(subset=["undertone"])[["image_id", "undertone"]].to_csv(out / "undertone.csv", index=False)
    print(f"{len(df)} photos labelled -> {out}/mst.csv, {out}/undertone.csv")


if __name__ == "__main__":
    main()
