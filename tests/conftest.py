import numpy as np
import pytest

from shadesense.ingest import make_synthetic


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """Six synthetic faces with ground-truth masks and labels."""
    root = tmp_path_factory.mktemp("synth")
    labels = make_synthetic(root / "faces", root / "masks", n_identities=6, size=128, seed=1)
    return root, labels


@pytest.fixture
def rgb():
    return np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
