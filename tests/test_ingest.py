import pandas as pd
import pytest

from shadesense.ingest import assert_identity_disjoint, quality, scan, split_by_identity


def test_scan_and_split(synth):
    root, labels = synth
    df = scan(root / "faces", "neutral")
    assert len(df) == 6 and df.is_reference.all()
    df = split_by_identity(df, 0.3, seed=0)
    assert set(df.split) == {"train", "test"}
    assert_identity_disjoint(df)


def test_leakage_detected():
    df = pd.DataFrame({"identity": ["a", "a"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_identity_disjoint(df)


def test_quality_flags_blur(rgb):
    import numpy as np
    flat = np.full_like(rgb, 120)
    assert quality(flat)["blur_var"] < quality(rgb)["blur_var"]
