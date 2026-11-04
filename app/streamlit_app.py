"""ShadeSense demo: upload or take a selfie -> skin tone, undertone, 3 foundation shades, lipstick try-on.

    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --config configs/smoke.yaml     # synthetic demo data

The selfie is processed in memory and never saved.
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadesense.config import load_config  # noqa: E402
from shadesense.pipeline import ShadeSense  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
cfg = load_config(ap.parse_args().config)

st.set_page_config(page_title="ShadeSense", page_icon="💄", layout="wide")
st.title("ShadeSense")
st.caption("Foundation and lipstick shades from one selfie. Suggestions only — always test in store.")

method = st.sidebar.radio("Skin colour method", ["baseline", "proposed"],
                          help="baseline: white balance + median skin colour. proposed: DINOv2 + trained colour head.")


@st.cache_resource
def pipeline(method: str):
    return ShadeSense(cfg, method=method)


try:
    ss = pipeline(method)
except FileNotFoundError:
    st.error("The proposed method needs a trained colour head. Run `python -m shadesense pipeline` first.")
    st.stop()

src = st.radio("Photo", ["Upload", "Camera"], horizontal=True)
file = st.file_uploader("Front-facing selfie, even light", type=["jpg", "jpeg", "png"]) if src == "Upload" else st.camera_input("Selfie")
if not file:
    st.stop()

img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.getvalue()))).convert("RGB")
img.thumbnail((cfg["dataset"]["image_size"],) * 2)
rgb = np.asarray(img)
res = ss.analyze(rgb, image_id=Path(getattr(file, "name", "camera")).stem)
if not res["found"]:
    st.warning(res["message"])
    st.stop()


def swatch(hex_colour: str, label: str) -> str:
    return (f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
            f"<div style='width:36px;height:36px;border-radius:8px;background:{hex_colour};border:1px solid #ccc'></div>"
            f"<div>{label}</div></div>")


left, right = st.columns(2)
with left:
    st.subheader("Your skin")
    st.markdown(swatch(res["skin_hex"], f"Monk Skin Tone <b>MST {res['mst']}</b> · undertone <b>{res['undertone']}</b>"),
                unsafe_allow_html=True)
    st.caption(f"Confidence {res['confidence']:.2f} · {res['latency_ms']:.0f} ms")
    st.subheader("Foundation, closest 3")
    for s in res["foundation"]:
        st.markdown(swatch(s["hex"], f"<b>{s['brand']}</b> {s['product']} {s['shade']} · ΔE {s['delta_e']:.1f}"),
                    unsafe_allow_html=True)
    st.subheader("Lipstick")
    names = [l["name"] for l in res["lipsticks"]]
    pick = st.radio("Try on", names, horizontal=True)
    chosen = next(l["hex"] for l in res["lipsticks"] if l["name"] == pick)
with right:
    if chosen != res["tryon_lipstick"]:
        res = ss.analyze(rgb, lipstick_hex=chosen, image_id=Path(getattr(file, "name", "camera")).stem)
    st.image([rgb, res["tryon_image"]], caption=["Original", f"With {pick}"], width=300)
