"""Build a folder ready to push to a Hugging Face Space (Gradio SDK).

    python scripts/build_space.py                 # -> build/space/
    cd build/space && git init && git remote add origin https://huggingface.co/spaces/<you>/shadesense
    git add . && git commit -m "ShadeSense demo" && git push -u origin main

Copies only code, config, the shade catalogue and trained model heads. It never copies photos: data/raw
(selfies) and results/ stay on your laptop. Check the listing it prints before pushing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shadesense.config import load_config  # noqa: E402
REQUIREMENTS = """numpy
pandas
pyarrow
scipy
scikit-image
scikit-learn
faiss-cpu
pillow
pyyaml
joblib
torch
transformers
mediapipe
gradio
"""
README = """---
title: ShadeSense
emoji: 💄
colorFrom: pink
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# ShadeSense

One selfie → Monk Skin Tone, undertone, the 3 closest foundation shades and a lipstick try-on.
CS 5130 course project. Photos are processed in memory and never stored. Suggestions only.
"""
APP = '''import os

os.environ.setdefault("SHADESENSE_CONFIG", "config.yaml")
from app.gradio_app import build  # noqa: E402

build().launch(server_name="0.0.0.0")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="build/space")
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "app").mkdir(parents=True)
    shutil.copytree(ROOT / "shadesense", out / "shadesense", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "app" / "gradio_app.py", out / "app" / "gradio_app.py")
    (out / "app" / "__init__.py").write_text("")

    cfg = load_config(ROOT / a.config)
    name = cfg["dataset"]["name"]
    store = Path(cfg["paths"]["store"]) / name
    cat_src = ROOT / cfg["catalogue"]["csv"]
    if not cat_src.exists():
        cat_src = ROOT / cfg["catalogue"]["demo_csv"]
        print(f"note: real catalogue missing, using the demo catalogue {cat_src}")
    (out / "assets").mkdir()
    shutil.copy(cat_src, out / "assets" / "catalogue.csv")
    for f in ("colour_head.joblib", "undertone_clf.joblib"):
        if (ROOT / store / f).exists():
            (out / "assets" / name).mkdir(exist_ok=True)
            shutil.copy(ROOT / store / f, out / "assets" / name / f)
    cfg["paths"].update({"store": "assets", "models": "models"})
    cfg["catalogue"].update({"csv": "assets/catalogue.csv", "demo_csv": "assets/catalogue.csv"})
    cfg["segment"]["backend"] = "landmarks"
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "README.md").write_text(README)
    (out / "app.py").write_text(APP)

    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    bad = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}]
    assert not bad, f"image files would be published: {bad}"
    print(f"Space ready in {out} ({len(files)} files):")
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
