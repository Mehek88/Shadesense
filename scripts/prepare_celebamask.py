"""Convert CelebAMask-HQ into the layout ShadeSense expects, for segmentation mIoU (module 2).

    python scripts/prepare_celebamask.py --src data/raw/CelebAMask-HQ --n 2000

Reads  <src>/CelebA-HQ-img/<i>.jpg  and  <src>/CelebAMask-HQ-mask-anno/<folder>/<i:05d>_<part>.png
Writes data/raw/celebamask/<i>/neutral.jpg              (resized to 512, matching the masks)
       data/raw/celebamask_masks/<i>/neutral_skin.png   (skin + nose, as both our segmenters define skin)
       data/raw/celebamask_masks/<i>/neutral_lips.png   (upper + lower lip)
Check the folder names against your download; the dataset README describes the layout.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/CelebAMask-HQ")
    ap.add_argument("--out", default="data/raw/celebamask")
    ap.add_argument("--masks-out", default="data/raw/celebamask_masks")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    src = Path(a.src)
    parts = {p.name: p for p in (src / "CelebAMask-HQ-mask-anno").rglob("*.png")}
    done = 0
    for i in range(30000):
        img_path = src / "CelebA-HQ-img" / f"{i}.jpg"
        skin_f = parts.get(f"{i:05d}_skin.png")
        if not img_path.exists() or skin_f is None:
            continue
        load = lambda name: (np.asarray(Image.open(parts[name]).convert("L")) > 127) if name in parts else None
        skin = load(f"{i:05d}_skin.png")
        for extra in ("nose",):
            m = load(f"{i:05d}_{extra}.png")
            skin = skin | m if m is not None else skin
        lips = np.zeros_like(skin)
        for part in ("u_lip", "l_lip"):
            m = load(f"{i:05d}_{part}.png")
            lips = lips | m if m is not None else lips
        skin &= ~lips
        (Path(a.out) / str(i)).mkdir(parents=True, exist_ok=True)
        (Path(a.masks_out) / str(i)).mkdir(parents=True, exist_ok=True)
        Image.open(img_path).convert("RGB").resize(skin.shape[::-1], Image.LANCZOS).save(Path(a.out) / str(i) / "neutral.jpg", quality=95)
        Image.fromarray(skin.astype(np.uint8) * 255).save(Path(a.masks_out) / str(i) / "neutral_skin.png")
        Image.fromarray(lips.astype(np.uint8) * 255).save(Path(a.masks_out) / str(i) / "neutral_lips.png")
        done += 1
        if done >= a.n:
            break
    print(f"wrote {done} images to {a.out} and masks to {a.masks_out}")


if __name__ == "__main__":
    main()
