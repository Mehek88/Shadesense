"""Command line. Run from the repo root:

  python -m shadesense make-demo   [--config configs/smoke.yaml]   synthetic faces + demo shade catalogue
  python -m shadesense ingest      validate images, split by identity      (module 1)
  python -m shadesense extract     segment + features for every image      (modules 2-5)
  python -m shadesense train       fit DINOv2 -> LAB head, undertone classifier
  python -m shadesense evaluate    all metrics, baseline vs proposed       (module 8)
  python -m shadesense pipeline    ingest -> extract -> train -> evaluate
  python -m shadesense analyze --image selfie.jpg [--method proposed] [--out tryon.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shadesense")
    ap.add_argument("command", choices=["make-demo", "ingest", "extract", "train", "evaluate", "pipeline", "analyze"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-dino", action="store_true", help="baseline only: skip DINOv2 features")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--method", default="baseline", choices=["baseline", "proposed"])
    ap.add_argument("--out", default="tryon.png")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)

    if a.command == "make-demo":
        from .ingest import make_demo_catalogue, make_synthetic

        ds = cfg["dataset"]
        make_synthetic(ds["images_dir"], ds["masks_dir"], seed=cfg["seed"])
        make_demo_catalogue(cfg["catalogue"]["demo_csv"], seed=cfg["seed"])
        print(f"[make-demo] faces in {ds['images_dir']}, masks in {ds['masks_dir']}, catalogue {cfg['catalogue']['demo_csv']}")
    if a.command in ("ingest", "pipeline"):
        from .ingest import ingest
        from .segment import load_segmenter

        ingest(cfg, load_segmenter(cfg) if cfg["ingest"]["require_face"] else None)
    if a.command in ("extract", "pipeline"):
        from .evaluate import extract

        extract(cfg, use_dino=not a.no_dino)
    if a.command in ("train", "pipeline"):
        from .evaluate import train

        train(cfg)
    if a.command in ("evaluate", "pipeline"):
        from .evaluate import evaluate

        evaluate(cfg, latency=not a.no_latency)
    if a.command == "analyze":
        from PIL import Image

        from .ingest import load_image
        from .pipeline import ShadeSense

        img_path = Path(a.image).resolve()
        root = Path(cfg["dataset"]["images_dir"]).resolve()
        image_id = str(img_path.relative_to(root).with_suffix("")) if img_path.is_relative_to(root) else img_path.stem
        res = ShadeSense(cfg, method=a.method).analyze(load_image(img_path, cfg["dataset"]["image_size"]),
                                                       image_id=image_id)
        if res.get("found"):
            Image.fromarray(res.pop("tryon_image")).save(a.out)
            res["tryon_saved_to"] = a.out
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
