"""Module 8: offline experiments, always reporting baseline (LAB) and proposed (DINOv2) side by side.

  extract   segment every image, plus synthetic re-lit / corrupted copies of each reference image,
            and store LAB + DINOv2 features             -> data/store/<ds>/features.parquet
  train     fit the DINOv2 -> LAB colour head (and an undertone classifier if labels exist), train split only
  evaluate  shade accuracy under lighting changes, per Monk Skin Tone group, segmentation IoU,
            undertone accuracy, latency                  -> results/<ds>/*.csv, summary.json, *.png

Ground truth (`evaluate.truth`):
  card  each person's reference photo (`dataset.reference_condition`) has a white card next to the face.
        The photo is rescaled so the card is neutral white, and the median skin LAB of that calibrated
        photo is the true colour. The card is independent of every method tested, so the evaluation is
        not circular. People whose card is not found fall back to `raw` (counted in summary.json).
  raw   the median skin LAB of the uncorrected reference photo (older behaviour).
The question is whether the system returns the same shade as the truth when the light changes.

Scopes reported: reference photo, real lighting (the person's other photos), synthetic lighting (re-lit
copies of one photo), blur / JPEG.
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

from .colour import delta_e, lab_to_hex, mst_from_lab
from .config import paths, seed_everything
from .features import DinoFeaturizer, make_colour_head, skin_crop, skin_lab
from .ingest import load_image, load_manifest
from .lighting import card_calibrate, corrupt, correct, find_card, relight
from .match import ShadeMatcher
from .segment import OracleSegmenter, iou, load_segmenter
from .store import load_catalogue, read_features, write_features

METHOD_COLOURS = {"raw": "#898781", "baseline": "#2a78d6", "proposed": "#eb6834"}


# --------------------------------------------------------------------------- extract
def render_view(rgb: np.ndarray, view: str, ev: dict) -> np.ndarray:
    """Recreate a stored view ('orig', 'relight:<name>', 'corrupt:<name>') from the original photo."""
    kind, _, name = view.partition(":")
    if kind == "relight":
        spec = ev["relight"][name]
        return relight(rgb, spec["gains"], spec["brightness"])
    if kind == "corrupt":
        return corrupt(rgb, ev["corruptions"][name])
    return rgb


def _views(rgb, ev, synthetic: bool):
    yield "orig", rgb
    if synthetic:
        for name in ev["relight"]:
            yield f"relight:{name}", render_view(rgb, f"relight:{name}", ev)
        for name in ev["corruptions"]:
            yield f"corrupt:{name}", render_view(rgb, f"corrupt:{name}", ev)


def _synthetic_sources(df: pd.DataFrame, cfg: dict) -> set[str]:
    """image_ids that get synthetic lighting changes: each person's `relight_condition` photo, or their
    reference photo if they have none. The default (daylight, no card) keeps the card out of the
    re-lit copies, so white balance cannot simply read the light off the card."""
    cond = cfg["evaluate"].get("relight_condition") or cfg["dataset"]["reference_condition"]
    out = set()
    for _, g in df.groupby("identity"):
        pick = g[g.condition == cond]
        pick = pick if len(pick) else g[g.is_reference]
        out.update(pick.image_id.head(1))
    return out


def load_card_boxes(cfg: dict) -> dict[str, tuple]:
    """Optional hand-marked card positions: CSV image_id,x0,y0,x1,y1 as fractions of width and height."""
    path = cfg["dataset"].get("card_boxes")
    if not path or not Path(path).exists():
        return {}
    b = pd.read_csv(path)
    return {r.image_id: (r.x0, r.y0, r.x1, r.y1) for r in b.itertuples()}


def calibrated_lab(img: np.ndarray, masks, cfg: dict, box=None):
    """(skin LAB after white-card calibration, card mask) or (None, None) if no card is found."""
    card = find_card(img, masks.background & ~masks.skin & ~masks.lips, box=box)
    if card is None:
        return None, None
    cal = card_calibrate(img, card, cfg["evaluate"].get("card_white", 0.8))
    return skin_lab(cal, masks.skin, cfg["features"]["trim_percentiles"]), card


def extract(cfg: dict, use_dino: bool = True) -> pd.DataFrame:
    seed_everything(cfg["seed"])
    p, ds, f = paths(cfg), cfg["dataset"], cfg["features"]
    df = load_manifest(cfg)
    seg = load_segmenter(cfg)
    gt = OracleSegmenter(ds["masks_dir"]) if ds.get("masks_dir") and cfg["segment"]["backend"] != "oracle" else None
    dino = DinoFeaturizer(f["dino_hf_id"], f["dino_size"]) if use_dino else None
    synth = _synthetic_sources(df, cfg)
    boxes = load_card_boxes(cfg)

    rows, crops = [], []
    for row in df.itertuples():
        rgb = load_image(row.path, ds["image_size"])
        for view, img in _views(rgb, cfg["evaluate"], row.image_id in synth):
            t0 = time.perf_counter()
            masks = seg(img, row.image_id)
            seg_ms = (time.perf_counter() - t0) * 1000
            rec = {"image_id": row.image_id, "identity": row.identity, "condition": row.condition,
                   "is_reference": row.is_reference, "split": row.split, "view": view,
                   "mst_label": getattr(row, "mst", np.nan), "undertone_label": getattr(row, "undertone", None),
                   "seg_ms": seg_ms, "found": masks is not None and masks.skin.sum() >= 50}
            if not rec["found"]:
                rows.append({**rec, "lab_raw": np.zeros(3), "lab_corr": np.zeros(3), "lab_ref": np.zeros(3),
                             "card_found": False})
                crops.append(None)
                continue
            rec["skin_px"] = int(masks.skin.sum())
            rec["lab_raw"] = skin_lab(img, masks.skin, f["trim_percentiles"])
            corrected = correct(img, cfg["lighting"]["method"], cfg["lighting"]["p_norm"], mask=masks.background,
                                 max_gain=cfg["lighting"]["max_gain"])
            rec["lab_corr"] = skin_lab(corrected, masks.skin, f["trim_percentiles"])
            ref_lab = None
            if row.is_reference and view == "orig":
                ref_lab, _ = calibrated_lab(img, masks, cfg, boxes.get(row.image_id))
            rec["card_found"] = ref_lab is not None
            rec["lab_ref"] = ref_lab if ref_lab is not None else np.zeros(3)
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
    n_ref = int((out.is_reference & (out.view == "orig")).sum())
    print(f"[extract] {len(out)} views from {len(df)} images, {int(out.found.sum())} with a face, "
          f"white card found in {int(out.card_found.sum())}/{n_ref} reference photos -> {p['features']}")
    return out


# --------------------------------------------------------------------------- truth + train
def _truth(meta: pd.DataFrame, vecs: dict, cfg: dict | None = None) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """True skin LAB per identity and where it came from ('card' or 'raw')."""
    mode = (cfg or {}).get("evaluate", {}).get("truth", "card")
    ref = meta.index[(meta.view == "orig") & meta.is_reference & meta.found]
    truth, source = {}, {}
    for i in ref:
        use_card = mode == "card" and "lab_ref" in vecs and bool(meta.get("card_found", pd.Series(False, meta.index))[i])
        truth[meta.identity[i]] = vecs["lab_ref"][i] if use_card else vecs["lab_raw"][i]
        source[meta.identity[i]] = "card" if use_card else "raw"
    return truth, source


def train(cfg: dict) -> None:
    p = paths(cfg)
    meta, vecs = read_features(p["features"])
    truth, _ = _truth(meta, vecs, cfg)
    m = ((meta.split == "train") & meta.found & meta.identity.isin(truth)).to_numpy()
    if "dino" in vecs and m.sum() == 0:
        print("[train] no training identities with a reference photo: skipping the colour head")
    if cfg.get("models_from") and cfg["models_from"] != cfg["dataset"]["name"]:
        print(f"[train] using models trained on {cfg['models_from']}: nothing to train")
        return
    if "dino" in vecs and m.sum() > 0:
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
LIGHTING_SCOPES = ("real lighting", "synthetic lighting")


def _scope(view: str, is_reference: bool) -> str:
    if view.startswith("relight"):
        return "synthetic lighting"
    if view.startswith("corrupt"):
        return "blur / JPEG"
    return "reference photo" if is_reference else "real lighting"


def correction_effect(per: pd.DataFrame) -> pd.DataFrame:
    """Does white balance help or hurt, per skin-tone group? Pairs 'raw' and 'baseline' on the same views."""
    cols = ["image_id", "view", "scope", "mst_group"]
    raw = per[per.method == "raw"][cols + ["de_skin", "top3"]]
    base = per[per.method == "baseline"][cols + ["de_skin", "top3"]]
    both = raw.merge(base, on=cols, suffixes=("_raw", "_corrected"))
    if both.empty:
        return pd.DataFrame()
    rows = []
    for (scope, grp), g in both.groupby(["scope", "mst_group"]):
        rows.append({"scope": scope, "mst_group": grp, "n": len(g),
                     "mean_de_raw": g.de_skin_raw.mean(), "mean_de_corrected": g.de_skin_corrected.mean(),
                     "de_change": g.de_skin_corrected.mean() - g.de_skin_raw.mean(),
                     "share_improved": float((g.de_skin_corrected < g.de_skin_raw).mean()),
                     "top3_raw": g.top3_raw.mean(), "top3_corrected": g.top3_corrected.mean()})
    return pd.DataFrame(rows)


def mst_prediction(meta: pd.DataFrame, idx: np.ndarray, estimates: dict) -> dict | None:
    """Predicted Monk Skin Tone (nearest MST colour to the estimated skin LAB) vs the hand label."""
    labels = pd.to_numeric(meta.mst_label.iloc[idx], errors="coerce").to_numpy() if "mst_label" in meta else None
    if labels is None or np.isnan(labels).all():
        return None
    ok = ~np.isnan(labels)
    out = {}
    for method, est in estimates.items():
        pred = np.array([mst_from_lab(e) for e in est[ok]])
        err = np.abs(pred - labels[ok])
        out[method] = {"n": int(ok.sum()), "exact": float((err == 0).mean()), "within_1": float((err <= 1).mean()),
                       "mae": float(err.mean())}
    return out


PER_COLS = ["image_id", "identity", "condition", "view", "scope", "method", "mst", "mst_group", "truth_source",
            "top1", "top3", "de_skin", "de_shade", "true_hex", "pred_hex", "true_shade", "pred_shade"]


def evaluate(cfg: dict, latency: bool = True) -> dict:
    p, ev = paths(cfg), cfg["evaluate"]
    res = p["results"]
    res.mkdir(parents=True, exist_ok=True)
    meta, vecs = read_features(p["features"])
    truth, truth_src = _truth(meta, vecs, cfg)
    csv = cfg["catalogue"]["csv"] if Path(cfg["catalogue"]["csv"]).exists() else cfg["catalogue"]["demo_csv"]
    matcher = ShadeMatcher(load_catalogue(csv), cfg["match"]["candidates"])
    k = cfg["match"]["k"]

    test_all = meta.index[(meta.split == "test") & meta.found].to_numpy()
    estimates = {"raw": vecs["lab_raw"][test_all], "baseline": vecs["lab_corr"][test_all]}
    if "dino" in vecs and p["head"].exists():
        estimates["proposed"] = joblib.load(p["head"]).predict(vecs["dino"][test_all])

    # Monk Skin Tone group per identity: hand label if present, otherwise estimated from the true colour.
    ref_rows = meta[(meta.view == "orig") & meta.is_reference]
    mst_lab = pd.to_numeric(ref_rows.set_index("identity").mst_label, errors="coerce").to_dict()
    mst_src = "label" if pd.notna(list(mst_lab.values())).any() else "estimated"
    mst = {i: (mst_lab.get(i) if pd.notna(mst_lab.get(i)) else mst_from_lab(t)) for i, t in truth.items()}

    rows = []
    for j, r in enumerate(test_all):
        ident = meta.identity[r]
        if ident not in truth:
            continue
        t = truth[ident]
        true_shade = matcher.top_k_ids(t, 1)[0]
        for method, est in estimates.items():
            ids = matcher.top_k_ids(est[j], k)
            rows.append({"image_id": meta.image_id[r], "identity": ident, "condition": meta.condition[r],
                         "view": meta.view[r] if meta.view[r] != "orig" else f"photo:{meta.condition[r]}",
                         "scope": _scope(meta.view[r], bool(meta.is_reference[r])), "method": method,
                         "mst": mst[ident], "mst_group": mst_group(mst[ident], ev["mst_groups"]),
                         "truth_source": truth_src[ident],
                         "top1": ids[0] == true_shade, "top3": true_shade in ids,
                         "de_skin": float(delta_e(est[j], t)),
                         "de_shade": float(delta_e(matcher.lab[ids[0]], t)),
                         "true_hex": lab_to_hex(t), "pred_hex": lab_to_hex(est[j]),
                         "true_shade": int(true_shade), "pred_shade": int(ids[0])})
    per = pd.DataFrame(rows, columns=PER_COLS)
    per.to_csv(res / "per_image.csv", index=False)

    summary = {"n_test_identities": int(per.identity.nunique()), "mst_source": mst_src,
               "truth": ev.get("truth", "card"),
               "truth_source_counts": pd.Series(truth_src).value_counts().to_dict(),
               "catalogue": csv, "n_shades": len(matcher.cat)}
    lit = per[per.scope.isin(LIGHTING_SCOPES)]
    by_view = by_mst = pd.DataFrame()
    if len(per):
        by_view = _summ(per, ["method", "scope", "view"], ev["bootstrap"])
        by_view.to_csv(res / "by_view.csv", index=False)
        by_mst = pd.concat([_summ(per, ["method", "scope", "mst_group"], ev["bootstrap"]),
                            _summ(lit.assign(scope="all lighting changes"), ["method", "scope", "mst_group"], ev["bootstrap"])
                            if len(lit) else pd.DataFrame()], ignore_index=True)
        by_mst.to_csv(res / "by_mst_group.csv", index=False)
        overall = _summ(per, ["method", "scope"], ev["bootstrap"])
        overall.to_csv(res / "overall.csv", index=False)
        print(overall.round(3).to_string(index=False))
        effect = correction_effect(per)
        if len(effect):
            effect.to_csv(res / "correction_effect_by_mst.csv", index=False)
            el = effect[effect.scope.isin(LIGHTING_SCOPES)]
            summary["lighting_correction_de_change_by_mst"] = (
                el.groupby("mst_group").apply(lambda g: float(np.average(g.de_change, weights=g.n)), include_groups=False).to_dict())
    else:
        print("[evaluate] no test identities with a reference photo: shade metrics skipped")
        effect = pd.DataFrame()

    if "proposed" in estimates and len(lit):
        a = lit[lit.method == "proposed"].sort_values(["image_id", "view"]).top3.to_numpy()
        b = lit[lit.method == "baseline"].sort_values(["image_id", "view"]).top3.to_numpy()
        summary["H1_mcnemar_top3_proposed_vs_baseline"] = mcnemar(a, b)
    if len(lit):
        g = _summ(lit, ["method", "mst_group"], ev["bootstrap"])
        summary["H2_top3_gap_between_mst_groups"] = g.groupby("method").top3.agg(lambda s: float(s.max() - s.min())).to_dict()
    summary["mst_prediction"] = mst_prediction(meta, test_all, estimates)

    if "iou_skin" in meta:
        seg = meta[meta.iou_skin.notna()]
        pd.DataFrame([{"backend": cfg["segment"]["backend"], "n": len(seg), "mIoU_skin": seg.iou_skin.mean(),
                       "mIoU_lips": seg.iou_lips.mean()}]).to_csv(res / "segmentation.csv", index=False)
    summary["undertone"] = _undertone(meta, vecs)
    if latency and len(per):
        summary["latency"] = _latency(cfg, list(estimates))
    summary["storage_bytes_per_image"] = {"baseline": 3 * 4, **({"proposed": int(vecs["dino"].shape[1] * 4)} if "dino" in vecs else {})}
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _plots(by_view, lit, effect, res, ev["bootstrap"], list(ev["mst_groups"]))
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


def _bars(ax, piv: pd.DataFrame, colours: dict, ylabel: str, ylim=None):
    w = 0.8 / piv.shape[1]
    for i, m in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w, piv[m], w - 0.02, color=colours.get(m, "#898781"), label=m,
               edgecolor="white", linewidth=1.5)
    ax.set_xticks(np.arange(len(piv)) + w * (piv.shape[1] - 1) / 2, piv.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, color="#52514e")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def _plots(by_view: pd.DataFrame, lit: pd.DataFrame, effect: pd.DataFrame, res: Path, n_boot: int,
           group_order: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = []
    if len(by_view):
        piv = by_view.pivot_table(index="view", columns="method", values="top3") * 100
        charts.append((piv, METHOD_COLOURS, "Top-3 accuracy (%)", (0, 105), "top3_by_view.png",
                       "Top-3 shade accuracy by photo / lighting / corruption"))
    if len(lit):
        g = _summ(lit, ["method", "mst_group"], n_boot)
        piv = g.pivot_table(index="mst_group", columns="method", values="top3") * 100
        charts.append((piv, METHOD_COLOURS, "Top-3 accuracy (%)", (0, 105), "top3_by_mst_group.png",
                       "Top-3 accuracy under lighting changes, by skin-tone group"))
    if len(effect):
        e = effect[effect.scope.isin(LIGHTING_SCOPES)]
        if len(e):
            piv = e.groupby("mst_group")[["mean_de_raw", "mean_de_corrected"]].mean()
            piv.columns = ["no correction", "white balance"]
            charts.append((piv, {"no correction": METHOD_COLOURS["raw"], "white balance": METHOD_COLOURS["baseline"]},
                           "Mean skin ΔE2000 (lower is better)", None, "correction_effect_by_mst.png",
                           "Does white balance help? Skin colour error by skin-tone group"))
    order = list(group_order or []) + ["unknown"]
    for piv, colours, ylabel, ylim, fname, title in charts:
        piv = piv[[m for m in colours if m in piv]]
        if piv.index.name == "mst_group":
            piv = piv.reindex([g for g in order if g in piv.index])
        fig, ax = plt.subplots(figsize=(8, 3.8))
        _bars(ax, piv, colours, ylabel, ylim)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        fig.tight_layout(); fig.savefig(res / fname, dpi=180); plt.close(fig)
