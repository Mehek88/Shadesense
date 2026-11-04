import numpy as np

from shadesense.colour import hex_to_lab
from shadesense.ingest import make_demo_catalogue
from shadesense.match import ShadeMatcher, confidence, lipsticks_for
from shadesense.store import load_catalogue


def test_top_k_sorted_and_exact(tmp_path):
    make_demo_catalogue(tmp_path / "c.csv", n=60)
    cat = load_catalogue(tmp_path / "c.csv")
    m = ShadeMatcher(cat, candidates=20)
    target = cat[["L", "a", "b"]].to_numpy()[17]
    top = m.top_k(target, 3)
    assert top.delta_e.is_monotonic_increasing and top.delta_e[0] < 1e-6
    assert m.top_k_ids(target, 1) == [17]


def test_lipsticks_and_confidence():
    assert len(lipsticks_for("warm", hex_to_lab("#a07e56"))) == 2
    assert confidence(10_000, 0.0) == 1.0 and confidence(0, 0.0) == 0.0
