"""Module 5: feature store (Parquet) and the product shade catalogue (FAISS index over CIELAB)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .colour import hex_to_lab

VECTOR_COLS = ("lab_raw", "lab_corr", "lab_ref", "dino")


def write_features(path: str | Path, df: pd.DataFrame) -> None:
    """Scalar columns as-is; vector columns (lists of floats) as fixed-size float32 lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scalars = df.drop(columns=[c for c in VECTOR_COLS if c in df])
    table = pa.Table.from_pandas(scalars.reset_index(drop=True), preserve_index=False)
    for c in VECTOR_COLS:
        if c in df and df[c].notna().all():
            arr = np.stack(df[c].to_numpy()).astype(np.float32)
            table = table.append_column(c, pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel()), arr.shape[1]))
    pq.write_table(table, path, compression="zstd")


def read_features(path: str | Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    table = pq.read_table(path)
    vecs = {c: np.asarray(table[c].combine_chunks().flatten(), dtype=np.float32).reshape(table.num_rows, -1)
            for c in VECTOR_COLS if c in table.column_names}
    meta = table.drop([c for c in vecs]).to_pandas()
    return meta, vecs


# --------------------------------------------------------------------------- catalogue
def _find(cols: list[str], *names: str) -> str | None:
    low = {c.lower(): c for c in cols}
    return next((low[n] for n in names if n in low), None)


def load_catalogue(csv: str | Path) -> pd.DataFrame:
    """Accepts any CSV with a hex column plus brand/product/shade-name columns (names detected, case-insensitive)."""
    df = pd.read_csv(csv)
    hex_col = _find(list(df), "hex", "hex_code", "hexcode", "color", "colour")
    if hex_col is None:
        raise ValueError(f"{csv}: need a hex colour column, got {list(df)}")
    out = pd.DataFrame({
        "brand": df[_find(list(df), "brand", "brand_name")] if _find(list(df), "brand", "brand_name") else "",
        "product": df[_find(list(df), "product", "product_name", "name")] if _find(list(df), "product", "product_name", "name") else "",
        "shade": df[_find(list(df), "shade", "shade_name", "description", "imgalt")] if _find(list(df), "shade", "shade_name", "description", "imgalt") else "",
        "hex": df[hex_col].astype(str).str.strip(),
    })
    out = out[out.hex.str.fullmatch(r"#?[0-9a-fA-F]{6}")].drop_duplicates(["brand", "product", "shade", "hex"])
    out["hex"] = "#" + out.hex.str.lstrip("#").str.lower()
    lab = np.stack([hex_to_lab(h) for h in out.hex])
    out[["L", "a", "b"]] = lab
    return out.reset_index(drop=True)


def build_catalogue_index(cat: pd.DataFrame):
    import faiss

    index = faiss.IndexFlatL2(3)
    index.add(np.ascontiguousarray(cat[["L", "a", "b"]].to_numpy(), dtype=np.float32))
    return index
