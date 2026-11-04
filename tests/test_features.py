import numpy as np
import pandas as pd

from shadesense.colour import delta_e, hex_to_lab
from shadesense.features import skin_crop, skin_lab
from shadesense.ingest import load_image
from shadesense.segment import OracleSegmenter


def test_skin_lab_recovers_true_colour(synth):
    root, labels = synth
    seg = OracleSegmenter(root / "masks")
    for row in labels.itertuples():
        rgb = load_image(root / "faces" / f"{row.image_id}.png")
        lab = skin_lab(rgb, seg.load(row.image_id).skin)
        assert delta_e(lab, hex_to_lab(row.true_hex)) < 6   # side shading + noise, so not exact


def test_skin_crop_shape(synth):
    root, _ = synth
    rgb = load_image(root / "faces" / "person001/neutral.png")
    crop = skin_crop(rgb, OracleSegmenter(root / "masks").load("person001/neutral").skin, 224)
    assert crop.shape == (224, 224, 3)
