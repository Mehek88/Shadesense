import numpy as np

from shadesense.colour import delta_e, hex_to_lab
from shadesense.evaluate import calibrated_lab
from shadesense.features import skin_lab
from shadesense.ingest import load_image
from shadesense.lighting import card_calibrate, find_card, relight
from shadesense.segment import OracleSegmenter


def _scene():
    img = np.full((100, 100, 3), 120, np.uint8)
    img[30:70, 35:65] = [200, 150, 120]                 # "skin"
    img[40:60, 2:12] = 231                              # white card, linear 0.8
    skin = np.zeros((100, 100), bool); skin[30:70, 35:65] = True
    bg = np.ones((100, 100), bool); bg[20:80, 25:75] = False
    return img, skin, bg


def test_find_card_locates_card_and_box_override():
    img, _, bg = _scene()
    card = find_card(relight(img, (1.2, 1.0, 0.8), 0.9), bg)
    assert card is not None and card[50, 6] and not card[50, 50] and not card[10, 90]
    boxed = find_card(img, bg, box=(0.0, 0.4, 0.12, 0.6))
    assert boxed[50, 5] and not boxed[10, 5]
    assert find_card(np.full((100, 100, 3), 120, np.uint8), bg) is None     # plain wall: no card


def test_card_calibration_undoes_cast_and_exposure():
    img, skin, bg = _scene()
    true = skin_lab(img, skin)
    cast = relight(img, (1.15, 1.0, 0.8), 0.85)
    fixed = card_calibrate(cast, find_card(cast, bg))
    assert delta_e(skin_lab(fixed, skin), true) < 1.5 < delta_e(skin_lab(cast, skin), true)


def test_synthetic_reference_has_card(synth):
    root, labels = synth
    rgb = load_image(root / "faces" / "person002/neutral.png")
    masks = OracleSegmenter(root / "masks").load("person002/neutral")
    cfg = {"evaluate": {"card_white": 0.8}, "features": {"trim_percentiles": [10, 90]}}
    lab, card = calibrated_lab(rgb, masks, cfg)
    true = hex_to_lab(labels.set_index("image_id").true_hex["person002/neutral"])
    assert card is not None and delta_e(lab, true) < 3.5
