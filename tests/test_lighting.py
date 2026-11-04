import numpy as np

from shadesense.lighting import corrupt, gray_world, relight


def test_gray_world_undoes_colour_cast(rgb):
    cast = relight(rgb, gains=(1.3, 1.0, 0.7))
    fixed = gray_world(cast)
    spread = lambda x: np.ptp(x.reshape(-1, 3).mean(0))
    assert spread(fixed) < spread(cast)


def test_relight_identity_and_brightness(rgb):
    assert np.abs(relight(rgb).astype(int) - rgb).max() <= 1
    assert relight(rgb, brightness=0.5).mean() < rgb.mean()


def test_corruptions_keep_shape(rgb):
    for spec in ({"kind": "jpeg", "quality": 20}, {"kind": "blur", "sigma": 1.5}):
        assert corrupt(rgb, spec).shape == rgb.shape
