import importlib.util
from pathlib import Path

import pandas as pd
from PIL import Image


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_prepare_mst_e(tmp_path):
    src = tmp_path / "dl"
    for subj, n in (("s1", 2), ("s2", 1)):
        (src / subj).mkdir(parents=True)
        for i in range(n):
            Image.new("RGB", (8, 8)).save(src / subj / f"{subj}_{i}.jpg")
    pd.DataFrame({"filename": ["s1_0.jpg", "s1_1.jpg", "s2_0.jpg"], "subject_id": ["s1", "s1", "s2"],
                  "MST": ["MST 3", "3", "8"]}).to_csv(tmp_path / "labels.csv", index=False)
    lab = _load("prepare_mst_e").prepare(src, tmp_path / "labels.csv", tmp_path / "out", tmp_path / "mst.csv")
    assert sorted(lab.mst) == [3, 3, 8] and (tmp_path / "out" / "s2" / "s2_0.jpg").exists()


def test_expand_labels(tmp_path):
    for p in ("p01/reference", "p01/warm_lamp", "p02/reference"):
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(tmp_path / f"{p}.jpg")
    people = pd.DataFrame({"identity": ["p01", "p02"], "mst": [4, 7], "undertone": ["warm", "cool"]})
    df = _load("expand_labels").expand(people, tmp_path)
    assert set(df.image_id) == {"p01/reference", "p01/warm_lamp", "p02/reference"}
