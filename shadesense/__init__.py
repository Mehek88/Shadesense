"""ShadeSense: foundation and lipstick shade matching from one selfie.

    from shadesense import analyze
    result = analyze("selfie.jpg")      # uses config.yaml in the repo root
"""
from __future__ import annotations

__version__ = "0.1.0"

_default = None


def analyze(image, method: str = "baseline", **kwargs) -> dict:
    """image: file path or RGB uint8 array. Loads the pipeline once and reuses it."""
    global _default
    from .ingest import load_image
    from .pipeline import ShadeSense

    if _default is None or _default.method != method:
        _default = ShadeSense(method=method)
    rgb = load_image(image, _default.cfg["dataset"]["image_size"]) if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__") else image
    return _default.analyze(rgb, **kwargs)
