from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from lib.constants import COLUMN_VARIANTS, GALLONS_PER_BBL, REQUIRED_COLUMNS


@dataclass
class NormalizeResult:
    df: pd.DataFrame
    missing_required: List[str]
    column_map: Dict[str, str]
    volume_note: str


def _norm(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).replace("\ufeff", "").strip().lower())
    return s


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    used = set()
    out_cols = []
    for c in df.columns:
        key = _norm(c)
        canon = COLUMN_VARIANTS.get(key, str(c).strip())
        candidate = canon
        i = 2
        while candidate in used:
            candidate = f"{canon} ({i})"
            i += 1
        used.add(candidate)
        out_cols.append(candidate)
        mapping[str(c)] = candidate
    out = df.copy()
    out.columns = out_cols
    return out, mapping


def _to_numeric(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def derive_volumes(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    for col in ["Daily Volume (gal)", "Monthly Volume (gal 30 day)", "Daily Volume (bbl)", "Monthly Volume (bbl 30 day)"]:
        _to_numeric(out, col)

    notes = []
    if "Daily Volume (bbl)" not in out.columns and "Daily Volume (gal)" in out.columns:
        out["Daily Volume (bbl)"] = out["Daily Volume (gal)"] / GALLONS_PER_BBL
        notes.append("Derived Daily Volume (bbl) from Daily Volume (gal)")
    if "Daily Volume (gal)" not in out.columns and "Daily Volume (bbl)" in out.columns:
        out["Daily Volume (gal)"] = out["Daily Volume (bbl)"] * GALLONS_PER_BBL
        notes.append("Derived Daily Volume (gal) from Daily Volume (bbl)")
    if "Monthly Volume (bbl 30 day)" not in out.columns and "Monthly Volume (gal 30 day)" in out.columns:
        out["Monthly Volume (bbl 30 day)"] = out["Monthly Volume (gal 30 day)"] / GALLONS_PER_BBL
        notes.append("Derived Monthly Volume (bbl 30 day) from Monthly Volume (gal 30 day)")
    if "Monthly Volume (gal 30 day)" not in out.columns and "Monthly Volume (bbl 30 day)" in out.columns:
        out["Monthly Volume (gal 30 day)"] = out["Monthly Volume (bbl 30 day)"] * GALLONS_PER_BBL
        notes.append("Derived Monthly Volume (gal 30 day) from Monthly Volume (bbl 30 day)")

    if "Monthly Volume (gal 30 day)" not in out.columns and "Daily Volume (gal)" in out.columns:
        out["Monthly Volume (gal 30 day)"] = out["Daily Volume (gal)"] * 30
        notes.append("Derived Monthly Volume (gal 30 day) from Daily Volume (gal) × 30")
    if "Monthly Volume (bbl 30 day)" not in out.columns and "Daily Volume (bbl)" in out.columns:
        out["Monthly Volume (bbl 30 day)"] = out["Daily Volume (bbl)"] * 30
        notes.append("Derived Monthly Volume (bbl 30 day) from Daily Volume (bbl) × 30")

    return out, ("; ".join(notes) if notes else "Using provided volume columns")


def normalize_dataset(df_raw: pd.DataFrame) -> NormalizeResult:
    df, column_map = normalize_columns(df_raw)
    df, volume_note = derive_volumes(df)
    for col in ["Freight Cost (30 days)", "Product Cost (30 days)", "Total Cost (30 days)", "Latitude", "Longitude"]:
        _to_numeric(df, col)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return NormalizeResult(df=df, missing_required=missing, column_map=column_map, volume_note=volume_note)
