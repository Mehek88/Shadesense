"""ShadeSense demo in Gradio (runs locally and on Hugging Face Spaces).

    python app/gradio_app.py                                   # uses config.yaml
    SHADESENSE_CONFIG=configs/smoke.yaml python app/gradio_app.py

The selfie is processed in memory and never saved.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shadesense.config import load_config  # noqa: E402
from shadesense.pipeline import ShadeSense  # noqa: E402

CFG = load_config(os.environ.get("SHADESENSE_CONFIG", "config.yaml"))
NOTE = "Suggestions only: shades vary by brand and lighting, so always test in store."


@lru_cache(maxsize=2)
def pipeline(method: str) -> ShadeSense:
    return ShadeSense(CFG, method=method)


def _get(method: str) -> tuple[ShadeSense, str]:
    try:
        return pipeline(method), ""
    except FileNotFoundError:
        return pipeline("baseline"), "The DINOv2 method has no trained colour head here yet, so the baseline was used. "


def _prep(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img).convert("RGB")
    pil.thumbnail((CFG["dataset"]["image_size"],) * 2)
    return np.asarray(pil)


def _swatch(hex_colour: str, text: str) -> str:
    return (f"<div style='display:flex;align-items:center;gap:10px;margin:6px 0'>"
            f"<div style='width:34px;height:34px;border-radius:8px;background:{hex_colour};border:1px solid #9994'></div>"
            f"<div>{text}</div></div>")


def _html(res: dict, note: str) -> str:
    out = [f"<h3>Your skin</h3>",
           _swatch(res["skin_hex"], f"Monk Skin Tone <b>MST {res['mst']}</b> · undertone <b>{res['undertone']}</b>"),
           f"<p style='opacity:.7'>Confidence {res['confidence']:.2f} · {res['latency_ms']:.0f} ms · method {res['method']}</p>",
           "<h3>Foundation, closest 3</h3>"]
    out += [_swatch(s["hex"], f"<b>{s['brand']}</b> {s['product']} {s['shade']} · ΔE {s['delta_e']:.1f}") for s in res["foundation"]]
    out += ["<h3>Lipstick suggestions</h3>"]
    out += [_swatch(l["hex"], f"{l['name']} ({l['family']})") for l in res["lipsticks"]]
    out += [f"<p style='opacity:.7'>{note}{NOTE}</p>"]
    return "".join(out)


def analyze(img, method: str):
    if img is None:
        return "<p>Upload or take a front-facing selfie in even light.</p>", None, gr.update(choices=[], value=None), None
    ss, note = _get(method)
    rgb = _prep(img)
    res = ss.analyze(rgb)
    if not res["found"]:
        return f"<p>{res['message']}</p>", None, gr.update(choices=[], value=None), None
    names = [l["name"] for l in res["lipsticks"]]
    state = {"lipsticks": {l["name"]: l["hex"] for l in res["lipsticks"]}, "method": method}
    return _html(res, note), res["tryon_image"], gr.update(choices=names, value=names[0]), state


def tryon(img, name: str, state):
    if img is None or not name or not state:
        return None
    ss, _ = _get(state["method"])
    res = ss.analyze(_prep(img), lipstick_hex=state["lipsticks"][name])
    return res["tryon_image"] if res["found"] else None


def build() -> gr.Blocks:
    with gr.Blocks(title="ShadeSense") as demo:
        gr.Markdown("# ShadeSense\nOne selfie → skin tone, undertone, 3 foundation shades and a lipstick try-on. "
                    "Your photo is processed in memory and never stored.")
        state = gr.State()
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="numpy", sources=["upload", "webcam"], label="Selfie")
                method = gr.Radio(["baseline", "proposed"], value="baseline", label="Skin colour method",
                                  info="baseline: white balance + median skin colour · proposed: DINOv2 + colour head")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column():
                result = gr.HTML()
                lip = gr.Radio([], label="Try on lipstick")
                out = gr.Image(label="Try-on", interactive=False)
        go.click(analyze, [img, method], [result, out, lip, state])
        lip.change(tryon, [img, lip, state], out)
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
