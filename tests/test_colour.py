import numpy as np

from shadesense.colour import delta_e, hex_to_lab, lab_to_hex, mst_from_lab, undertone_rule


def test_hex_lab_roundtrip():
    for h in ("#a07e56", "#f6ede4", "#292420"):
        assert lab_to_hex(hex_to_lab(h)) == h


def test_delta_e_zero_and_symmetric():
    a, b = hex_to_lab("#a07e56"), hex_to_lab("#825c43")
    assert delta_e(a, a) < 1e-6
    assert np.isclose(delta_e(a, b), delta_e(b, a))


def test_mst_extremes():
    assert mst_from_lab(hex_to_lab("#f6ede4")) == 1
    assert mst_from_lab(hex_to_lab("#292420")) == 10


def test_undertone_rule():
    assert undertone_rule([60, 20, 5]) == "cool"      # pinkish
    assert undertone_rule([60, 5, 25]) == "warm"      # yellowish
