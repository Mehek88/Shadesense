"""Module 8: offline experiments, always reporting baseline (LAB) and proposed (DINOv2) side by side.

  extract   segment every image, plus synthetic re-lit / corrupted copies of each reference image,
            and store LAB + DINOv2 features             -> data/store/<ds>/features.parquet
  train     fit the DINOv2 -> LAB colour head (and an undertone classifier if labels exist), train split only
  evaluate  shade accuracy under lighting changes, per Monk Skin Tone group, segmentation IoU,
            undertone accuracy, latency                  -> results/<ds>/*.csv, summary.json, *.png

Ground truth: each identity's colour is the median skin LAB of its reference-light image
(`dataset.reference_condition`). The question is whether the system returns the same shade when
the light changes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .colour import delta_e, mst_from_lab
from .config import paths, seed_everything
from .features import DinoFeaturizer, make_colour_head, skin_crop, skin_lab
from .ingest import load_image, load_manifest
from .lighting import corrupt, correct, relight
from .match import ShadeMatcher
from .segment import OracleSegmenter, iou, load_segmenter
from .store import load_catalogue, read_features, write_features

METHOD_COLOURS = {"raw": "#898781", "baseline": "#2a78d6", "proposed": "#eb6834"}


# --------------------------------------------------------------------------- extract
def _views(row, rgb, ev):
    yield "orig", rgb
    if row.is_reference:
        for name, spec in ev["relight"].items():
            yield f"relight:{name}", relight(rgb, spec["gains"], spec["brightness"])
        for name, spec in ev["corruptions"].items():
            yield f"corrupt:{name}", corrupt(rgb, spec)


def extract(cfg: dict, use_dino: bool = True) -> pd.DataFrame:
    seed_everything(cfg["seed"])
    p, ds, f = paths(cfg), cfg["dataset"], cfg["features"]
    df = load_manifest(cfg)
    seg = load_segmenter(cfg)
    gt = OracleSegmenter(ds["masks_dir"]) if ds.get("masks_dir") and cfg["segment"]["backend"] != "oracle" else None
    dino = DinoFeaturizer(f["dino_hf_id"], f["dino_size"]) if use_dino else None

    rows, crops = [], []
    for row in df.itertuples():
        rgb = load_image(row.path, ds["image_size"])
        for view, img in _views(row, rgb, cfg["evaluate"]):
            t0 = time.perf_counter()
            masks = seg(img, row.image_id)
            seg_ms = (time.perf_counter() - t0) * 1000
            rec = {"image_id": row.image_id, "identity": row.identity, "condition": row.condition,
                   "is_reference": row.is_reference, "split": row.split, "view": view,
                   "mst_label": getattr(row, "mst", np.nan), "undertone_label": getattr(row, "undertone", None),
                   "seg_ms": seg_ms, "found": masks is not None and masks.skin.sum() >= 50}
            if not rec["found"]:
                rows.append({**rec, "lab_raw": np.zeros(3), "lab_corr": np.zeros(3)})
                crops.append(None)
                continue
            rec["skin_px"] = int(masks.skin.sum())
            rec["lab_raw"] = skin_lab(img, masks.skin, f["trim_percentiles"])
            corrected = correct(img, cfg["lighting"]["method"], cfg["lighting"]["p_norm"], mask=masks.background,
                                 max_gain=cfg["lighting"]["max_gain"])
            rec["lab_corr"] = skin_lab(corrected, masks.skin, f["trim_percentiles"])
            if gt is not None and view == "orig":
                ref = gt.load(row.image_id)
                if ref is not None:
                    rec["iou_skin"], rec["iou_lips"] = iou(masks.skin, ref.skin), iou(masks.lips, ref.lips)
            rows.append(rec)
            crops.append(skin_crop(img, masks.skin, f["dino_size"]) if dino else None)
    out = pd.DataFrame(rows)
    if dino:
        dim = dino.dim
        emb = np.zeros((len(out), dim), np.float32)
        idx = [i for i, c in enumerate(crops) if c is not None]
        t0 = time.perf_counter()
        for s in range(0, len(idx), 32):
            chunk = idx[s:s + 32]
            emb[chunk] = dino([crops[i] for i in chunk])
        out["dino_ms"] = (time.perf_counter() - t0) * 1000 / max(1, len(idx))
        out["dino"] = list(emb)
    write_features(p["features"], out)
    print(f"[extract] {len(out)} views from {len(df)} images, {int(out.found.sum())} with a face -> {p['features']}")
    return out


# --------------------------------------------------------------------------- truth + train
def _truth(meta: pd.DataFrame, vecs: dict) -> dict[str, np.ndarray]:
    ref = meta.index[(meta.view == "orig") & meta.is_reference & meta.found]
    return {meta.identity[i]: vecs["lab_raw"][i] for i in ref}


def train(cfg: dict) -> None:
    p = paths(cfg)
    meta, vecs = read_features(p["features"])
    truth = _truth(meta, vecs)
    if "dino" in vecs:
        m = ((meta.split == "train") & meta.found & meta.identity.isin(truth)).to_numpy()
        y = np.stack([truth[i] for i in meta.identity[m]])
        head = make_colour_head(cfg["features"]["ridge_alpha"]).fit(vecs["dino"][m], y)
        joblib.dump(head, p["head"])
        print(f"[train] colour head on {m.sum()} views ({meta.identity[m].nunique()} identities) -> {p['head']}")
    lab_ok = meta.undertone_label.notna() & (meta.split == "train") & (meta.view == "orig") & meta.found
    if lab_ok.sum() >= 6 and meta.undertone_label[lab_ok].nunique() > 1:
        clf = LogisticRegression(max_iter=1000).fit(vecs["lab_corr"][lab_ok.to_numpy()], meta.undertone_label[lab_ok])
        joblib.dump(clf, p["undertone_clf"])
        print(f"[train] undertone classifier on {lab_ok.sum()} labelled images -> {p['undertone_clf']}")


# --------------------------------------------------------------------------- metrics
def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired successes (a, b bool arrays over the same cases)."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    pval = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "p_value": float(pval)}


def mst_group(mst: float, groups: dict) -> str:
    for name, (lo, hi) in groups.items():
        if lo <= mst <= hi:
            return name
    return "unknown"


def _summ(df: pd.DataFrame, keys: list[str], n_boot: int) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        lo, hi = bootstrap_ci(g.top3.to_numpy().astype(float), n_boot)
        rows.append({**dict(zip(keys, k)), "n": len(g), "top1": g.top1.mean(), "top3": g.top3.mean(),
                     "top3_lo": lo, "top3_hi": hi, "mean_de_skin": g.de_skin.mean(), "mean_de_shade": g.de_shade.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evaluate
def evaluate(cfg: dict, latency: bool = True) -> dict:
    p, ev = paths(cfg), cfg["evaluate"]
    res = p["results"]
    res.mkdir(parents=True, exist_ok=True)
    meta, vecs = read_features(p["features"])
    truth = _truth(meta, vecs)
    csv = cfg["catalogue"]["csv"] if Path(cfg["catalogue"]["csv"]).exists() else cfg["catalogue"]["demo_csv"]
    matcher = ShadeMatcher(load_catalogue(csv), cfg["match"]["candidates"])
    k = cfg["match"]["k"]

    test = meta.index[(meta.split == "test") & meta.found & meta.identity.isin(truth)].to_numpy()
    estimates = {"raw": vecs["lab_raw"][test], "baseline": vecs["lab_corr"][test]}
    if "dino" in vecs and p["head"].exists():
        estimates["proposed"] = joblib.load(p["head"]).predict(vecs["dino"][test])

    # Monk Skin Tone group per identity: hand label if present, otherwise estimated from the reference colour.
    ref_rows = meta[(meta.view == "orig") & meta.is_reference]
    mst_lab = ref_rows.set_index("identity").mst_label.to_dict()
    mst_src = "label" if pd.notna(list(mst_lab.values())).any() else "estimated"
    mst = {i: (mst_lab.get(i) if pd.notna(mst_lab.get(i)) else mst_from_lab(t)) for i, t in truth.items()}

    rows = []
    for j, r in enumerate(test):
        ident = meta.identity[r]
        t = truth[ident]
        true_shade = matcher.top_k_ids(t, 1)[0]
        for method, est in estimates.items():
            ids = matcher.top_k_ids(est[j], k)
            rows.append({"image_id": meta.image_id[r], "identity": ident, "view": meta.view[r],
                         "view_type": meta.view[r].split(":")[0], "method": method,
                         "mst": mst[ident], "mst_group": mst_group(mst[ident], ev["mst_groups"]),
                         "top1": ids[0] == true_shade, "top3": true_shade in ids,
                         "de_skin": float(delta_e(est[j], t)),
                         "de_shade": float(delta_e(matcher.lab[ids[0]], t))})
    per = pd.DataFrame(rows)
    per.to_csv(res / "per_image.csv", index=False)

    by_view = _summ(per, ["method", "view"], ev["bootstrap"])
    by_view.to_csv(res / "by_view.csv", index=False)
    lit = per[per.view_type == "relight"]
    by_mst = _summ(lit, ["method", "mst_group"], ev["bootstrap"])
    by_mst.to_csv(res / "by_mst_group.csv", index=False)
    overall = _summ(per.assign(scope=np.where(per.view_type == "relight", "lighting changes",
                                              np.where(per.view_type == "corrupt", "blur / JPEG", "original photos"))),
                    ["method", "scope"], ev["bootstrap"])
    overall.to_csv(res / "overall.csv", index=False)

    summary = {"n_test_identities": int(per.identity.nunique()), "mst_source": mst_src,
               "catalogue": csv, "n_shades": len(matcher.cat)}
    if "proposed" in estimates and len(lit):
        a = lit[lit.method == "proposed"].sort_values(["image_id", "view"]).top3.to_numpy()
        b = lit[lit.method == "baseline"].sort_values(["image_id", "view"]).top3.to_numpy()
        summary["H1_mcnemar_top3_proposed_vs_baseline"] = mcnemar(a, b)
    if len(by_mst):
        gaps = by_mst.groupby("method").top3.agg(lambda s: float(s.max() - s.min()))
        summary["H2_top3_gap_between_mst_groups"] = gaps.to_dict()

    if "iou_skin" in meta:
        seg = meta[meta.iou_skin.notna()]
        pd.DataFrame([{"backend": cfg["segment"]["backend"], "n": len(seg), "mIoU_skin": seg.iou_skin.mean(),
                       "mIoU_lips": seg.iou_lips.mean()}]).to_csv(res / "segmentation.csv", index=False)
    summary["undertone"] = _undertone(meta, vecs)
    if latency:
        summary["latency"] = _latency(cfg, list(estimates))
    summary["storage_bytes_per_image"] = {"baseline": 3 * 4, **({"proposed": int(vecs["dino"].shape[1] * 4)} if "dino" in vecs else {})}
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _plots(by_view, by_mst, res)
    print(overall.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))
    return summary


def _undertone(meta, vecs) -> dict | None:
    m = meta.undertone_label.notna() & (meta.view == "orig") & meta.found
    tr, te = (m & (meta.split == "train")).to_numpy(), (m & (meta.split == "test")).to_numpy()
    y_tr, y_te = meta.undertone_label[tr], meta.undertone_label[te]
    if tr.sum() < 6 or te.sum() < 2 or y_tr.nunique() < 2:
        return None
    out = {"majority_class_acc": float((y_te == y_tr.mode()[0]).mean())}
    for name in [n for n in ("lab_corr", "dino") if n in vecs]:
        pred = LogisticRegression(max_iter=2000).fit(vecs[name][tr], y_tr).predict(vecs[name][te])
        out[f"{name}_acc"] = float((pred == y_te).mean())
        out[f"{name}_macro_f1"] = float(f1_score(y_te, pred, average="macro"))
    return out


def _latency(cfg: dict, methods: list[str]) -> dict:
    from .pipeline import ShadeSense

    df = load_manifest(cfg)
    df = df[df.split == "test"].head(cfg["evaluate"]["latency_images"])
    out = {}
    for method in (m for m in methods if m != "raw"):
        ss = ShadeSense(cfg, method=method)
        imgs = [(load_image(r.path, cfg["dataset"]["image_size"]), r.image_id) for r in df.itertuples()]
        ss.analyze(imgs[0][0], image_id=imgs[0][1])          # warm-up
        t = []
        for img, iid in imgs:
            t0 = time.perf_counter(); ss.analyze(img, image_id=iid); t.append((time.perf_counter() - t0) * 1000)
        out[method] = {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "n": len(t)}
    return out


def _plots(by_view: pd.DataFrame, by_mst: pd.DataFrame, res: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for data, key, fname, title in ((by_view, "view", "top3_by_view.png", "Top-3 shade accuracy by lighting / corruption"),
                                    (by_mst, "mst_group", "top3_by_mst_group.png", "Top-3 accuracy under lighting changes, by skin-tone group")):
        if data.empty:
            continue
        piv = data.pivot_table(index=key, columns="method", values="top3")
        piv = piv[[m for m in METHOD_COLOURS if m in piv]]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        w = 0.8 / piv.shape[1]
        for i, m in enumerate(piv.columns):
            ax.bar(np.arange(len(piv)) + i * w, piv[m] * 100, w - 0.02, color=METHOD_COLOURS[m], label=m,
                   edgecolor="white", linewidth=1.5)
        ax.set_xticks(np.arange(len(piv)) + w * (piv.shape[1] - 1) / 2, piv.index, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Top-3 accuracy (%)", color="#52514e")
        ax.set_ylim(0, 105)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(res / fname, dpi=180); plt.close(fig)
