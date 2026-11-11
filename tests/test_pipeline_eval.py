"""End to end on tiny synthetic data, baseline only (no DINOv2 download): ingest -> extract -> evaluate -> report."""
import json

import pandas as pd

from shadesense.config import load_config
from shadesense.evaluate import evaluate, extract, train
from shadesense.ingest import ingest, make_demo_catalogue
from shadesense.report import check_cards, failures, find_shade, foundation_check


def _cfg(root, synth_root):
    cfg = load_config("configs/smoke.yaml")
    cfg["paths"].update({"manifests": str(root / "man"), "store": str(root / "store"), "results": str(root / "res")})
    cfg["dataset"].update({"name": "t", "images_dir": str(synth_root / "faces"), "masks_dir": str(synth_root / "masks"),
                           "mst_labels": str(synth_root / "faces_labels.csv"),
                           "undertone_labels": str(synth_root / "faces_labels.csv"), "test_fraction": 0.5})
    cfg["catalogue"].update({"csv": str(root / "cat.csv"), "demo_csv": str(root / "cat.csv")})
    cfg["evaluate"].update({"bootstrap": 20, "failures": 2})
    make_demo_catalogue(root / "cat.csv", n=60)
    return cfg


def test_end_to_end_baseline(tmp_path, synth):
    synth_root, _ = synth
    cfg = _cfg(tmp_path, synth_root)
    ingest(cfg, None)
    feats = extract(cfg, use_dino=False)
    assert feats.card_found.sum() == 6                       # every reference photo's card is found
    train(cfg)
    s = evaluate(cfg, latency=False)
    res = tmp_path / "res" / "t"
    assert s["truth_source_counts"] == {"card": 6}
    overall = pd.read_csv(res / "overall.csv")
    assert {"reference photo", "real lighting", "synthetic lighting", "blur / JPEG"} <= set(overall.scope)
    effect = pd.read_csv(res / "correction_effect_by_mst.csv")
    assert {"mean_de_raw", "mean_de_corrected", "share_improved"} <= set(effect.columns)
    assert s["mst_prediction"]["baseline"]["n"] > 0
    assert (res / "correction_effect_by_mst.png").exists()

    assert len(failures(cfg)) == 2 and (res / "failures.png").exists()
    assert check_cards(cfg).card_found.all()

    # a person "wearing" the catalogue shade closest to their true colour: that shade must rank first
    meta = pd.read_parquet(tmp_path / "man" / "t.parquet")
    per = pd.read_csv(res / "per_image.csv")
    cat = pd.read_csv(tmp_path / "cat.csv")
    row = per[per.scope == "reference photo"].iloc[0]
    shade = cat.iloc[int(row.true_shade)]
    pd.DataFrame([{"identity": row.identity, "foundation_brand": shade.brand, "foundation_shade": shade.shade}]).to_csv(
        tmp_path / "people.csv", index=False)
    cfg["dataset"]["foundation_labels"] = str(tmp_path / "people.csv")
    fc = foundation_check(cfg)
    assert fc["n_compared"] == 1 and fc["real_shade_in_top3"] == 1.0
    assert json.loads((res / "foundation_check.json").read_text())["n_compared"] == 1
    assert len(meta)


def test_find_shade_matching():
    cat = pd.DataFrame({"brand": ["Fenty Beauty", "Maybelline"], "product": ["Pro Filt'r", "Fit Me"],
                        "shade": ["240", "Natural Beige 220"]})
    assert find_shade(cat, "fenty", "240") == 0
    assert find_shade(cat, "Maybelline", "natural beige") == 1
    assert find_shade(cat, "", "") is None
