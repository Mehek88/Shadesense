"""Web API for the fuller website: POST /analyze with an image file -> JSON (+ base64 try-on PNG).

    uvicorn app.api:app --reload
    curl -F "file=@selfie.jpg" http://127.0.0.1:8000/analyze

Models load once at startup; images stay in memory and are never stored.
"""
import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadesense.config import load_config  # noqa: E402
from shadesense.pipeline import ShadeSense  # noqa: E402

cfg = load_config(os.environ.get("SHADESENSE_CONFIG", "config.yaml"))
pipe = ShadeSense(cfg, method=os.environ.get("SHADESENSE_METHOD", "baseline"))
app = FastAPI(title="ShadeSense")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), lipstick: str | None = None):
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not read image")
    img.thumbnail((cfg["dataset"]["image_size"],) * 2)
    res = pipe.analyze(np.asarray(img), lipstick_hex=lipstick, image_id=Path(file.filename or "upload").stem)
    if res.get("found"):
        buf = io.BytesIO()
        Image.fromarray(res.pop("tryon_image")).save(buf, format="PNG")
        res["tryon_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    return res
