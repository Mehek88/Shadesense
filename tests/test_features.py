import numpy as np
import pandas as pd

from shadesense.colour import delta_e, hex_to_lab
from shadesense.features import skin_crop, skin_lab
from shadesense.ingest import load_image
from shadesense.segment import OracleSegmenter


def test_skin_lab_recovers_true_colour(synth):
    """Photos have a random camera cast and exposure; after white-card calibration the median skin LAB
    must be close to the colour the face was drawn with."""
    from shadesense.evaluate import calibrated_lab

    root, labels = synth
    seg = OracleSegmenter(root / "masks")
    cfg = {"evaluate": {"card_white": 0.8}, "features": {"trim_percentiles": [10, 90]}}
    for row in labels[labels.image_id.str.endswith("/neutral")].itertuples():
        rgb = load_image(root / "faces" / f"{row.image_id}.png")
        masks = seg.load(row.image_id)
        lab, _ = calibrated_lab(rgb, masks, cfg)
        assert delta_e(lab, hex_to_lab(row.true_hex)) < 4      # side shading + noise, so not exact
        assert skin_lab(rgb, masks.skin).shape == (3,)


def test_skin_crop_shape(synth):
    root, _ = synth
    rgb = load_image(root / "faces" / "person001/neutral.png")
    crop = skin_crop(rgb, OracleSegmenter(root / "masks").load("person001/neutral").skin, 224)
    assert crop.shape == (224, 224, 3)
