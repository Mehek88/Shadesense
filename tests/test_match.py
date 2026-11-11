import numpy as np

from shadesense.colour import hex_to_lab
from shadesense.ingest import make_demo_catalogue
from shadesense.match import LIPSTICKS, ShadeMatcher, confidence, lipsticks_for
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
    assert len(lipsticks_for("warm", hex_to_lab("#a07e56"))) == 3
    assert sum(len(v) for v in LIPSTICKS.values()) >= 20
    light = lipsticks_for("cool", hex_to_lab("#f6ede4"), n=1)[0]["hex"]
    deep = lipsticks_for("cool", hex_to_lab("#292420"), n=1)[0]["hex"]
    assert hex_to_lab(light)[0] > hex_to_lab(deep)[0]          # lighter skin -> lighter lipstick
    assert confidence(10_000, 0.0) == 1.0 and confidence(0, 0.0) == 0.0
