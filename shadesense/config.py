"""Load config.yaml with `--set key.path=value` overrides."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import yaml

DEFAULT = Path(__file__).resolve().parents[1] / "config.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    path = Path(path or DEFAULT)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("defaults", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def paths(cfg: dict) -> dict[str, Path]:
    p = cfg["paths"]
    name = cfg["dataset"]["name"]
    return {
        "manifest": Path(p["manifests"]) / f"{name}.parquet",
        "rejections": Path(p["manifests"]) / f"{name}_rejections.csv",
        "quality": Path(p["manifests"]) / f"{name}_quality.json",
        "features": Path(p["store"]) / name / "features.parquet",
        "head": Path(p["store"]) / name / "colour_head.joblib",
        "undertone_clf": Path(p["store"]) / name / "undertone_clf.joblib",
        "catalogue_index": Path(p["store"]) / "catalogue.faiss",
        "results": Path(p["results"]) / name,
        "models": Path(p["models"]),
    }
