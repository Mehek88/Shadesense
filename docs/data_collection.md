# Selfie collection guide (Member 1, Data)

Goal: **15–20 people**, each photographed once as a calibrated reference and 4–5 times under everyday
lighting. Same face, different light: a good method should return the same foundation shade every time.

## 1. Before anyone takes a photo

- Each person signs the consent form (section 6). Keep the signed forms off GitHub.
- Give each person an anonymous ID: `p01`, `p02`, … Keep the name ↔ ID list private, never in the repo.
- Aim for a spread of skin tones: try to get people from light (MST 1–3), medium (4–7) **and** deep (8–10).
  The fairness results are meaningless if one group is missing.

## 2. Photos per person

| File name | How to take it | Purpose |
|---|---|---|
| `reference.jpg` | **Daylight** near a window (no direct sun), **plain white paper or card held next to the cheek**, no makeup, no filter | Ground truth colour (calibrated by the white card) |
| `daylight.jpg` | Same window light, **no card** | Easy test case |
| `warm_lamp.jpg` | Indoors under a yellow/warm bulb | Lighting test |
| `cool_led.jpg` | Under white/blue LED or office light | Lighting test |
| `dim.jpg` | Dim room, no flash | Lighting test |
| `phone_front.jpg` *(optional)* | Normal front-camera selfie, any light | Real-world case |

Rules for every photo:
- Face the camera straight on, whole face visible, hair off the cheeks, no glasses if possible.
- **Turn off beauty mode, filters, portrait mode and HDR.** These change skin colour.
- No makeup on the face (lip balm is fine). Same phone for all of one person's photos.
- Plain background if possible (a wall). Avoid strongly coloured backgrounds.

## 3. Folder layout (on your laptop / shared drive only — never on GitHub)

```
data/raw/team_selfies/
  p01/reference.jpg
  p01/daylight.jpg
  p01/warm_lamp.jpg
  ...
  p02/reference.jpg
  ...
```

`data/` is git-ignored, so the photos cannot be committed by accident. Share them with the team through a
private Google Drive / OneDrive folder.

`config.yaml` already expects `reference.jpg` as each person's reference photo (`dataset.reference_condition`).

## 4. Labels (hand-made, by people — not by the code)

Two people label each person's `reference.jpg` independently, then agree on a final value:

- **MST 1–10:** compare the cheek with the official Monk Skin Tone scale (skintone.google).
- **Undertone:** warm / cool / neutral (veins green = warm, blue/purple = cool, mixed = neutral;
  gold vs silver jewellery as a second check).
- **Real foundation shade** if the person wears one (brand + shade name). Used to check the catalogue.

Fill in `docs/templates/people.csv` (one row per person). The code needs one row **per photo**, so run:

```bash
python scripts/expand_labels.py
```

This writes `data/labels/mst.csv` and `data/labels/undertone.csv`. Then set in `config.yaml`:

```yaml
dataset:
  mst_labels: data/labels/mst.csv
  undertone_labels: data/labels/undertone.csv
  foundation_labels: docs/templates/people.csv   # real foundation shades, for the catalogue check
```

## 5. Check the data

```bash
python -m shadesense ingest --config config.yaml
```

Open `data/manifests/` and check the quality report: number of people, photos rejected (blurry, too
dark, no face) and anyone missing a reference photo. Re-take rejected photos.

Then check that the white card is found in every reference photo:

```bash
python -m shadesense check-cards --config config.yaml
```

Look through `results/team_selfies/cards/`: the card should be outlined in green. If it is missing or the
wrong area is outlined, either re-take the photo (card fully in shot, not overexposed, plain background) or
add a row to `data/labels/card_boxes.csv` (`image_id,x0,y0,x1,y1`, fractions of width and height, e.g.
`p03/reference,0.02,0.40,0.15,0.60`) and set `dataset.card_boxes: data/labels/card_boxes.csv`.

## 6. Consent form (copy, print, sign)

> I agree that my photos may be used by the CS 5130 ShadeSense project team (Northeastern University)
> to test skin-tone and shade-matching methods. I understand that:
> - my photos are stored privately, shared only within the team, and never posted publicly or on GitHub;
> - results are reported only with an anonymous ID, and my photos are not shown in the report or demo
>   unless I tick the box below;
> - I can withdraw at any time, and my photos will be deleted;
> - all photos will be deleted after the course ends (December 2026).
>
> ☐ My photos may appear in the report/presentation.
>
> Name: ____________  Signature: ____________  Date: ________

## 7. Other Member 1 tasks

- **MST-E dataset:** download it (accept its licence), then run `python scripts/prepare_mst_e.py --images <folder>
  --labels <labels csv>`; it prints the number of images per MST value. Run it with `configs/mst_e.yaml`.
- **Verify sizes** of every dataset used (Makeup Shades catalogue, CelebAMask-HQ, MST-E) and write the
  exact counts into the README before the Nov 18 progress report.
- **First milestone:** ~10 people (≈50 photos) by Nov 16, so the baseline can run on real data.
