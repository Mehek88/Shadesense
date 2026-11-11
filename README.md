# ShadeSense

One selfie in, and out come the skin tone (Monk Skin Tone 1–10), the undertone, the 3 closest foundation
shades, lipstick suggestions and a virtual try-on. Built for CS 5130, Theme 2 (Representation Learning
& Data Pipelines).

The research question is: does a frozen **DINOv2** embedding of the skin estimate colour more reliably
than the hand-crafted **median CIELAB** colour when the lighting changes? Both methods end in a LAB
colour and feed the same shade matcher (ΔE2000), so any difference comes from the representation.

![Try-on: original, MediaPipe lips, SegFormer lips](docs/tryon_example.png)

## Pipeline

| # | Module | File | Owner |
|---|---|---|---|
| 1 | Ingest & validate: reject blurry, badly exposed or no-face images; split **by identity** | `shadesense/ingest.py` | Member 1 (Data) |
| 2 | Face segmentation: MediaPipe landmarks (default) or SegFormer face parser → skin and lip masks | `shadesense/segment.py` | Member 2 (Vision) |
| 3 | Lighting: shades-of-gray white balance estimated from pixels around the face; **white-card calibration** of reference photos | `shadesense/lighting.py` | Member 2 (Vision) |
| 4 | Features: median LAB (baseline) vs DINOv2-S + ridge head → LAB (proposed) | `shadesense/features.py` | Member 3 (Features/Match) |
| 5 | Store: Parquet features, product catalogue with a FAISS index | `shadesense/store.py` | Member 3 (Features/Match) |
| 6 | Match: FAISS shortlist re-ranked by ΔE2000 → top 3; undertone → lipstick family (21 shades) | `shadesense/match.py` | Member 3 (Features/Match) |
| 7 | Try-on: recolour the lip mask in LAB, keeping texture | `shadesense/tryon.py` | Member 4 (Eval/App) |
| 8 | Evaluation: shade accuracy vs calibrated truth, lighting robustness, per skin-tone group, MST accuracy, latency | `shadesense/evaluate.py` | Member 4 (Eval/App) |
| 8b | Report material: failure cases, real-foundation check, white-card previews | `shadesense/report.py` | Member 4 (Eval/App) |
| – | Demo: Gradio (Hugging Face Spaces) and Streamlit; FastAPI web API | `app/` | Member 4 (Eval/App) |

`shadesense/pipeline.py` wraps everything as `ShadeSense(cfg).analyze(image)`, and
`shadesense/colour.py` holds the shared colour maths (ΔE2000, MST, hue angle). One `config.yaml`
drives every stage.

**How DINOv2 is compared with LAB.** DINOv2 gives 384 numbers, not a colour. A ridge regression head
(`features.make_colour_head`), trained on the training people only, maps those 384 numbers to the person's
true skin colour (L, a, b). From there both methods are identical: the same FAISS + ΔE2000 matcher picks
the foundation shades, and the same rule maps the colour to MST (the nearest of the 10 Monk Skin Tone
reference colours by ΔE2000).

## Ground truth (why the evaluation is not circular)

Each person has one **reference photo**: daylight, no makeup, with a plain **white card or paper held
next to the cheek**. The card has a known colour, so rescaling the photo until the card is neutral white
removes the light colour, the camera's white balance and the exposure. The median skin LAB of that
calibrated photo is the person's true colour. It does not depend on any method being tested.

Every other photo of the person (no card) is a test case: a good method returns the same shade as the
truth. People whose card is not found fall back to the uncalibrated reference; `summary.json` counts them.
The collection protocol is in [docs/data_collection.md](docs/data_collection.md).

## Setup

MediaPipe needs **Python 3.9–3.12**, so create a separate environment:

```bash
conda create -n shadesense python=3.11 -y
```

```bash
conda activate shadesense
```

```bash
make install
```

## Check it works (no downloads except DINOv2-small, about 90 MB)

```bash
make test
```

```bash
make smoke
```

`make smoke` generates 40 cartoon people across all 10 MST tones. Each has a reference photo with a white
card and a slightly wrong camera white balance, plus daylight, warm-lamp and cool-LED photos without the
card. It also makes a fake 120-shade catalogue ("DemoBrand"). It then runs ingest → extract → train →
evaluate → report with ground-truth masks; results land in `results/synthetic/`. This only tests the
plumbing: synthetic faces say nothing about real ones.

Analyse one image:

```bash
python -m shadesense analyze --image path/to/selfie.jpg --out tryon.png
```

## Real data

| Data | Where to put it | Used for |
|---|---|---|
| **Team selfies** (15–20 people, written consent) | `data/raw/team_selfies/<p01>/<condition>.jpg`: `reference.jpg` (daylight + white card), `daylight.jpg`, `warm_lamp.jpg`, `cool_led.jpg`, `dim.jpg` | Shade accuracy and lighting robustness against the calibrated truth |
| Hand labels | `docs/templates/people.csv` → `python scripts/expand_labels.py` → `data/labels/` | MST groups, undertone classifier, real-foundation check |
| **Makeup Shades** (The Pudding, also on Kaggle) | `data/raw/makeup_shades/shades.csv`; brand / product / hex columns are detected automatically | The real shade catalogue (the demo catalogue is used until this exists) |
| **MST-E** (Google Monk Skin Tone Examples) | download, then `python scripts/prepare_mst_e.py --images … --labels …` | Fairness: MST prediction accuracy per skin-tone group, `configs/mst_e.yaml` |
| **CelebAMask-HQ** | `data/raw/CelebAMask-HQ/`, then `python scripts/prepare_celebamask.py` | Segmentation mIoU, `configs/celebamask.yaml` |

Before the full run, check that the white card is found in every reference photo:

```bash
python -m shadesense check-cards --config config.yaml
```

Previews with the card outlined in green land in `results/team_selfies/cards/`. If a card is missed, add
its position to a CSV (`image_id,x0,y0,x1,y1` as fractions of the image) and set `dataset.card_boxes`.

Then run everything:

```bash
python -m shadesense pipeline --config config.yaml
```

**Privacy.** Selfies are collected with written consent and never committed: `data/`, `results/` and
`build/` are git-ignored. Results and the report use anonymous IDs (`p01`, …).

## Outputs (results/<dataset>/)

| File | Answers |
|---|---|
| `overall.csv` | Top-1 / top-3 shade accuracy and mean ΔE for raw, baseline and proposed, per scope: reference photo, real lighting (the person's other photos), synthetic lighting (re-lit copies), blur / JPEG |
| `by_view.csv`, `top3_by_view.png` | The same, per photo condition and per synthetic light |
| `by_mst_group.csv`, `top3_by_mst_group.png` | Fairness: accuracy for MST 1–3, 4–7 and 8–10, per scope |
| `correction_effect_by_mst.csv`, `.png` | Does white balance help or hurt, per skin-tone group? Raw vs corrected ΔE on the same photos |
| `summary.json` | H1 McNemar test (proposed vs baseline), H2 gap between skin-tone groups, MST prediction accuracy (exact / within 1), undertone accuracy and F1, truth source counts, latency p50/p95, storage per image |
| `failures.csv`, `failures.png` | Worst predictions per method with true vs predicted colour; fill in the `reason` column by hand |
| `foundation_check.csv`, `.json` | People who know their real foundation: catalogue colour vs measured skin, rank of their real shade, and whether the error is a constant offset (catalogue limitation) |
| `segmentation.csv` | Skin and lip mIoU against ground-truth masks |
| `per_image.csv` | Every prediction, for failure analysis |

Methods compared: **raw** is the median LAB with no lighting fix, **baseline** is the median LAB after
the lighting fix, and **proposed** is DINOv2 + colour head.

## Demo and web API

Gradio (the version for Hugging Face Spaces):

```bash
make gradio
```

Streamlit:

```bash
make demo
```

Web API (JSON plus a base64 try-on PNG):

```bash
make api
```

```bash
curl -F "file=@selfie.jpg" http://127.0.0.1:8000/analyze
```

Deploy to Hugging Face Spaces: `make space` builds `build/space/` with the code, config, catalogue and
trained heads (never photos), then push that folder to a new Gradio Space; the steps are at the top of
`scripts/build_space.py`. Images are processed in memory and never stored.

## What has been verified, and what we found

- All 30 unit tests pass, including an end-to-end baseline run on synthetic data; `make smoke` runs end to
  end on Python 3.11.
- **White-card calibration works on synthetic data**: the calibrated truth is within ΔE 1.7 of the colour
  the faces were drawn with on average, against ΔE 3.9 for the uncalibrated reference photo.
- The real-photo path works: MediaPipe and SegFormer both segment a real face, and `POST /analyze`
  returns a result in about 170 ms on a laptop CPU.
- **White balance can hurt.** On a photo with a coloured background (reddish hair, a flag), every
  correction tested made the skin colour *worse* than no correction. Gray-world was off by ΔE 19;
  capped shades-of-gray, now the default, by ΔE 8. On gray walls, correction cuts the error from
  about 4 to 1.5. `correction_effect_by_mst.csv` answers this per skin-tone group on the team selfies.
- **Dim light is the hardest case** for the baseline in the smoke run (see `failures.png`): white balance
  fixes colour, not exposure.
- **The SegFormer lip mask spills onto the chin** on some faces. Teeth, surrounding skin and specks
  are filtered out, but a mislabelled chin patch remains. MediaPipe landmarks give cleaner lips and
  are the default.
- **The DINOv2 head is weak on the smoke set** (28 training cartoon faces). This is expected; judge it on
  real data.
- **Limitations to report:** the catalogue hex values come from website swatch photos (about 2018), so a
  shade's listed colour is itself approximate; `foundation_check` measures this. The MST reference
  colours in `colour.py` should be checked against the official Monk Skin Tone page; the 21 lipstick
  colours in `match.py` are generic colours, not products.

## Repository rules

`main` is protected. Work on a branch per module, open a PR, and merge after 1 review and passing CI
(`.github/workflows/tests.yml`). Data, results, build output and model files are git-ignored.
