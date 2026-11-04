"""Module 6: match a skin colour to foundation shades and pick lipstick colours."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .colour import delta_e, hex_to_lab
from .store import build_catalogue_index

# Illustrative lipstick families per undertone (generic colour names, not products).
LIPSTICKS = {
    "warm": [("brick red", "#a4322b"), ("terracotta nude", "#b5654a"), ("coral", "#e0614f"), ("orange red", "#c83a24")],
    "cool": [("blue red", "#a1192f"), ("berry", "#8a2c4b"), ("mauve", "#a3687a"), ("fuchsia", "#c2347a")],
    "neutral": [("rosy nude", "#b8746e"), ("classic red", "#b0222e"), ("dusty rose", "#b26f78"), ("soft plum", "#7e3b4f")],
}


class ShadeMatcher:
    """FAISS shortlist by Euclidean LAB distance, then exact re-ranking by Delta E 2000."""

    def __init__(self, catalogue: pd.DataFrame, candidates: int = 50):
        self.cat = catalogue.reset_index(drop=True)
        self.lab = self.cat[["L", "a", "b"]].to_numpy()
        self.index = build_catalogue_index(self.cat)
        self.candidates = min(candidates, len(self.cat))

    def _ranked(self, lab, k: int) -> tuple[np.ndarray, np.ndarray]:
        _, idx = self.index.search(np.asarray(lab, dtype=np.float32)[None], self.candidates)
        idx = idx[0][idx[0] >= 0]
        de = delta_e(self.lab[idx], np.asarray(lab)[None])
        order = np.argsort(de)[:k]
        return idx[order], de[order]

    def top_k(self, lab, k: int = 3) -> pd.DataFrame:
        idx, de = self._ranked(lab, k)
        return self.cat.iloc[idx].assign(delta_e=de).reset_index(drop=True)

    def top_k_ids(self, lab, k: int = 3) -> list[int]:
        return self._ranked(lab, k)[0].tolist()


def lipsticks_for(undertone: str, skin_lab, n: int = 2) -> list[dict]:
    """Pick the family for the undertone; prefer shades with clear but not extreme contrast to the skin."""
    fam = LIPSTICKS.get(undertone, LIPSTICKS["neutral"])
    target_L = float(skin_lab[0]) - 25          # a bit deeper than the skin
    scored = sorted(fam, key=lambda x: abs(hex_to_lab(x[1])[0] - target_L))
    return [{"name": name, "hex": h} for name, h in scored[:n]]


def confidence(skin_pixels: int, best_delta_e: float) -> float:
    """Heuristic 0-1 score shown in the demo: low when the skin area is tiny or no shade is close."""
    area = min(1.0, skin_pixels / 5000)
    closeness = max(0.0, 1 - best_delta_e / 8)
    return round(float(area * closeness), 2)
