"""The whole pipeline as one call: ShadeSense(cfg).analyze(image) -> result dict.

The web layer (Streamlit / FastAPI) only ever calls this, so the ML code never changes for the website.
Images are processed in memory and never written to disk.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np

from .colour import lab_to_hex, mst_from_lab, undertone_rule
from .config import load_config, paths
from .features import DinoFeaturizer, skin_crop, skin_lab
from .lighting import correct
from .match import ShadeMatcher, confidence, lipsticks_for
from .segment import load_segmenter
from .store import load_catalogue
from .tryon import apply_lipstick


class ShadeSense:
    def __init__(self, cfg: dict | None = None, method: str = "baseline", catalogue_csv: str | None = None,
                 segmenter=None):
        """method: 'baseline' (gray-world + median LAB) or 'proposed' (DINOv2 + trained colour head)."""
        self.cfg = cfg or load_config()
        self.method = method
        self.segment = segmenter or load_segmenter(self.cfg)
        csv = catalogue_csv or self.cfg["catalogue"]["csv"]
        if not Path(csv).exists():
            csv = self.cfg["catalogue"]["demo_csv"]
        self.matcher = ShadeMatcher(load_catalogue(csv), self.cfg["match"]["candidates"])
        self.dino = self.head = None
        if method == "proposed":
            f = self.cfg["features"]
            self.dino = DinoFeaturizer(f["dino_hf_id"], f["dino_size"])
            self.head = joblib.load(paths(self.cfg)["head"])
        clf_path = paths(self.cfg)["undertone_clf"]
        self.undertone_clf = joblib.load(clf_path) if clf_path.exists() else None

    def estimate_lab(self, rgb: np.ndarray, masks) -> np.ndarray:
        if self.method == "proposed":
            crop = skin_crop(rgb, masks.skin, self.cfg["features"]["dino_size"])
            return self.head.predict(self.dino([crop]))[0]
        corrected = correct(rgb, self.cfg["lighting"]["method"], self.cfg["lighting"]["p_norm"], mask=masks.background,
                            max_gain=self.cfg["lighting"]["max_gain"])
        return skin_lab(corrected, masks.skin, self.cfg["features"]["trim_percentiles"])

    def analyze(self, rgb: np.ndarray, lipstick_hex: str | None = None, image_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        masks = self.segment(rgb, image_id)
        if masks is None or masks.skin.sum() < 50:
            return {"found": False, "message": "No face found. Use a clear, front-facing photo in even light."}
        lab = self.estimate_lab(rgb, masks)
        u = self.cfg["undertone"]
        if self.undertone_clf is not None and self.method == "baseline":
            undertone = str(self.undertone_clf.predict(np.asarray(lab)[None])[0])
        else:
            undertone = undertone_rule(lab, u["cool_below"], u["warm_above"])
        shades = self.matcher.top_k(lab, self.cfg["match"]["k"])
        lips = lipsticks_for(undertone, lab)
        chosen = lipstick_hex or lips[0]["hex"]
        tryon = apply_lipstick(rgb, masks.lips, chosen, self.cfg["tryon"]["strength"], self.cfg["tryon"]["feather_px"])
        return {
            "found": True,
            "method": self.method,
            "skin_hex": lab_to_hex(lab),
            "skin_lab": [round(float(v), 2) for v in lab],
            "mst": mst_from_lab(lab),
            "undertone": undertone,
            "foundation": shades[["brand", "product", "shade", "hex", "delta_e"]].round({"delta_e": 2}).to_dict("records"),
            "lipsticks": lips,
            "tryon_lipstick": chosen,
            "tryon_image": tryon,
            "confidence": confidence(int(masks.skin.sum()), float(shades.delta_e.iloc[0])),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
