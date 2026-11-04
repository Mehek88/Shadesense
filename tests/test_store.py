import numpy as np
import pandas as pd

from shadesense.store import load_catalogue, read_features, write_features


def test_features_roundtrip(tmp_path):
    df = pd.DataFrame({"image_id": ["a", "b"], "split": ["train", "test"],
                       "lab_raw": [np.ones(3), np.zeros(3)], "lab_corr": [np.ones(3), np.zeros(3)],
                       "dino": [np.arange(4.0), np.arange(4.0)]})
    write_features(tmp_path / "f.parquet", df)
    meta, vecs = read_features(tmp_path / "f.parquet")
    assert meta.image_id.tolist() == ["a", "b"] and vecs["dino"].shape == (2, 4)


def test_catalogue_column_detection(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"Brand": ["X", "X", "Y"], "Product": ["F", "F", "G"], "Hex": ["a07e56", "#bad", "zzz"]}).to_csv(p, index=False)
    cat = load_catalogue(p)
    assert len(cat) == 1 and cat.hex[0] == "#a07e56"     # "#bad" (3-digit) and "zzz" rejected
    assert {"L", "a", "b"} <= set(cat)
