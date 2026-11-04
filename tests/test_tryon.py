import numpy as np

from shadesense.colour import delta_e, hex_to_lab, rgb_to_lab
from shadesense.tryon import apply_lipstick


def test_lipstick_moves_lips_only():
    rgb = np.full((40, 40, 3), [200, 160, 140], np.uint8)
    lips = np.zeros((40, 40), bool); lips[18:24, 12:28] = True
    out = apply_lipstick(rgb, lips, "#a1192f", strength=1.0, feather_px=0)
    lab = rgb_to_lab(out)
    assert delta_e(lab[lips].mean(0), hex_to_lab("#a1192f")) < delta_e(rgb_to_lab(rgb)[lips].mean(0), hex_to_lab("#a1192f"))
    assert np.array_equal(out[0, 0], rgb[0, 0])


def test_empty_mask_is_noop():
    rgb = np.zeros((8, 8, 3), np.uint8)
    assert np.array_equal(apply_lipstick(rgb, np.zeros((8, 8), bool), "#ff0000"), rgb)
