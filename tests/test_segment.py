import numpy as np

from shadesense.segment import FACE_OVAL, LIPS_OUTER, OracleSegmenter, iou, masks_from_landmarks


def test_oracle_loads_masks(synth):
    root, _ = synth
    m = OracleSegmenter(root / "masks").load("person000/neutral")
    assert m is not None and m.skin.sum() > 100 and m.lips.sum() > 10
    assert not (m.skin & m.lips).any()


def test_masks_from_landmarks_geometry():
    pts = np.zeros((478, 2)) + 50.0
    t = np.linspace(0, 2 * np.pi, len(FACE_OVAL), endpoint=False)
    pts[FACE_OVAL] = np.c_[50 + 40 * np.cos(t), 50 + 45 * np.sin(t)]
    t = np.linspace(0, 2 * np.pi, len(LIPS_OUTER), endpoint=False)
    pts[LIPS_OUTER] = np.c_[50 + 12 * np.cos(t), 75 + 5 * np.sin(t)]
    m = masks_from_landmarks((100, 100, 3), pts, erode_px=1)
    assert m.skin.sum() > 1000 and m.lips.sum() > 50
    assert not (m.skin & m.lips).any()


def test_iou():
    a = np.zeros((4, 4), bool); a[:2] = True
    assert iou(a, a) == 1.0 and iou(a, ~a) == 0.0
