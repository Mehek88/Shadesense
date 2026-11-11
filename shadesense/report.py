"""Module 8b: material for the report, run after `evaluate`.

  failures          worst predictions per method, as a table plus a picture grid, for failure analysis
  foundation_check  people who know their real foundation shade: is the catalogue colour of that shade
                    close to their measured skin colour, and is the error a constant offset (e.g. the
                    catalogue's website swatches are all a bit too light)?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .colour import delta_e, hex_to_rgb
from .config import paths
from .evaluate import _truth, render_view
from .ingest import load_image, load_manifest
from .match import ShadeMatcher
from .store import load_catalogue, read_features


def _catalogue_csv(cfg: dict) -> str:
    from pathlib import Path

    return cfg["catalogue"]["csv"] if Path(cfg["catalogue"]["csv"]).exists() else cfg["catalogue"]["demo_csv"]


# --------------------------------------------------------------------------- failures
def failures(cfg: dict, n: int | None = None) -> pd.DataFrame:
    p = paths(cfg)
    res = p["results"]
    per_path = res / "per_image.csv"
    if not per_path.exists():
        print("[report] run `evaluate` first")
        return pd.DataFrame()
    per = pd.read_csv(per_path)
    n = n or cfg["evaluate"].get("failures", 8)
    methods = [m for m in ("baseline", "proposed") if m in set(per.method)]
    worst = (per[per.method.isin(methods)].sort_values("de_skin", ascending=False)
             .groupby("method").head(n).sort_values(["method", "de_skin"], ascending=[True, False]))
    worst = worst.assign(reason="")       # fill in by hand: shadow, hair over cheek, mask error, strong colour cast ...
    worst.to_csv(res / "failures.csv", index=False)
    if len(worst):
        _failure_grid(cfg, worst, methods, n, res / "failures.png")
    print(f"[report] {len(worst)} failure cases -> {res / 'failures.csv'}, {res / 'failures.png'}")
    return worst


def _failure_grid(cfg: dict, worst: pd.DataFrame, methods: list[str], n: int, out) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    paths_by_id = load_manifest(cfg).set_index("image_id").path.to_dict()
    fig, axes = plt.subplots(len(methods), n, figsize=(2.1 * n, 2.6 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        rows = worst[worst.method == method].head(n).reset_index(drop=True)
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(rows):
                continue
            row = rows.iloc[c]
            img = render_view(load_image(paths_by_id[row.image_id], cfg["dataset"]["image_size"]), row.view, cfg["evaluate"]) \
                if not row.view.startswith("photo:") else load_image(paths_by_id[row.image_id], cfg["dataset"]["image_size"])
            ax.imshow(img)
            h, w = img.shape[:2]
            s = w * 0.2
            for k, (hx, lbl) in enumerate(((row.true_hex, "true"), (row.pred_hex, "pred"))):
                ax.add_patch(Rectangle((k * s, h - s), s, s, color=np.asarray(hex_to_rgb(hx)) / 255, ec="white", lw=1.5))
                ax.text(k * s + s / 2, h - s - 3, lbl, color="white", fontsize=6, ha="center", va="bottom",
                        bbox={"facecolor": "black", "alpha": 0.4, "pad": 1, "lw": 0})
            ax.set_title(f"{row.image_id}\n{row.view} · MST {int(row.mst)}\nΔE {row.de_skin:.1f}", fontsize=6)
        axes[r, 0].text(-0.08, 0.5, method, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=10, fontweight="bold")
    fig.suptitle("Worst skin-colour estimates (left swatch: true colour, right: predicted)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- foundation check
def _norm(s) -> str:
    return str(s).strip().lower() if pd.notna(s) else ""


def find_shade(cat: pd.DataFrame, brand: str, shade: str) -> int | None:
    """Row of the catalogue for a person's real foundation: brand contains `brand`, shade name matches."""
    b, s = _norm(brand), _norm(shade)
    if not s:
        return None
    rows = cat[cat.brand.map(_norm).str.contains(b, regex=False)] if b else cat
    for col in ("shade", "product"):
        exact = rows[rows[col].map(_norm) == s]
        if len(exact):
            return int(exact.index[0])
    part = rows[rows.shade.map(_norm).str.contains(s, regex=False)]
    return int(part.index[0]) if len(part) else None


def foundation_check(cfg: dict) -> dict | None:
    from pathlib import Path

    path = cfg["dataset"].get("foundation_labels")
    if not path or not Path(path).exists():
        return None
    p = paths(cfg)
    lab = pd.read_csv(path, dtype={"identity": str})
    lab = lab.rename(columns={"brand": "foundation_brand", "shade": "foundation_shade"})
    lab = lab[lab.foundation_shade.notna()] if "foundation_shade" in lab else lab.iloc[0:0]
    meta, vecs = read_features(p["features"])
    truth, src = _truth(meta, vecs, cfg)
    cat = load_catalogue(_catalogue_csv(cfg))
    matcher = ShadeMatcher(cat, cfg["match"]["candidates"])
    rows = []
    for r in lab.itertuples():
        ident = str(r.identity)
        idx = find_shade(cat, getattr(r, "foundation_brand", ""), r.foundation_shade)
        rec = {"identity": ident, "foundation_brand": getattr(r, "foundation_brand", ""), "foundation_shade": r.foundation_shade,
               "in_catalogue": idx is not None, "has_reference": ident in truth}
        if idx is not None and ident in truth:
            t, c = truth[ident], matcher.lab[idx]
            ranked = matcher.top_k_ids(t, matcher.candidates)
            rec.update({"truth_source": src[ident], "dL": c[0] - t[0], "da": c[1] - t[1], "db": c[2] - t[2],
                        "delta_e": float(delta_e(c, t)), "catalogue_hex": cat.hex[idx],
                        "rank_of_real_shade": ranked.index(idx) + 1 if idx in ranked else None,
                        "real_shade_in_top3": idx in ranked[:cfg["match"]["k"]]})
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(p["results"] / "foundation_check.csv", index=False)
    ok = df[df.get("delta_e", pd.Series(dtype=float)).notna()] if "delta_e" in df else df.iloc[0:0]
    out = {"n_people": len(df), "n_found_in_catalogue": int(df.in_catalogue.sum()) if len(df) else 0, "n_compared": len(ok)}
    if len(ok):
        off = ok[["dL", "da", "db"]].astype(float)
        mean = off.mean()
        std = off.std(ddof=1) if len(off) > 1 else pd.Series(np.nan, index=off.columns)
        sem = std / np.sqrt(len(off))
        out.update({"mean_offset_Lab": mean.round(2).to_dict(), "std_offset_Lab": std.round(2).to_dict(),
                    "mean_delta_e": float(ok.delta_e.mean()), "real_shade_in_top3": float(ok.real_shade_in_top3.mean()),
                    # an offset is "constant" when the mean is well outside its own noise (|mean| > 2 standard
                    # errors); needs at least 2 people, otherwise None
                    "constant_offset": {k: (bool(abs(mean[k]) > 2 * sem[k]) if len(off) > 1 else None) for k in off.columns}})
    (p["results"] / "foundation_check.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[report] foundation check on {out['n_compared']} people -> {p['results'] / 'foundation_check.csv'}")
    return out


def report(cfg: dict) -> None:
    failures(cfg)
    foundation_check(cfg)


# --------------------------------------------------------------------------- card check (for the data team)
def check_cards(cfg: dict) -> pd.DataFrame:
    """Find the white card in every reference photo and save a preview with the card outlined in green.

    Look through results/<ds>/cards/ before running the pipeline. For any photo where the card is missing
    or the wrong area is outlined, add a row to the `dataset.card_boxes` CSV (image_id,x0,y0,x1,y1 as
    fractions of the width and height) and run this again.
    """
    from PIL import Image
    from scipy import ndimage

    from .evaluate import load_card_boxes
    from .lighting import find_card
    from .segment import load_segmenter

    out_dir = paths(cfg)["results"] / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    seg, boxes = load_segmenter(cfg), load_card_boxes(cfg)
    rows = []
    for r in load_manifest(cfg).query("is_reference").itertuples():
        rgb = load_image(r.path, cfg["dataset"]["image_size"])
        masks = seg(rgb, r.image_id)
        card = None if masks is None else find_card(rgb, masks.background & ~masks.skin & ~masks.lips, box=boxes.get(r.image_id))
        rec = {"image_id": r.image_id, "face_found": masks is not None, "card_found": card is not None,
               "card_area_pct": round(100 * card.mean(), 2) if card is not None else 0.0,
               "source": "hand box" if r.image_id in boxes else "automatic"}
        preview = rgb.copy()
        if card is not None:
            edge = card & ~ndimage.binary_erosion(card, iterations=2)
            preview[ndimage.binary_dilation(edge, iterations=2)] = [0, 220, 0]
            rec["card_rgb"] = "#%02x%02x%02x" % tuple(np.median(rgb[card], axis=0).astype(int))
        Image.fromarray(preview).save(out_dir / f"{r.image_id.replace('/', '__')}.jpg", quality=85)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cards.csv", index=False)
    missing = df[~df.card_found].image_id.tolist() if len(df) else []
    print(f"[check-cards] card found in {int(df.card_found.sum()) if len(df) else 0}/{len(df)} reference photos -> {out_dir}")
    if missing:
        print("  no card found (add to dataset.card_boxes):", ", ".join(missing))
    return df
