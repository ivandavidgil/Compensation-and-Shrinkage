#!/usr/bin/env python3
"""roll_shrinkage_grouping.py  ─ v2025‑04‑22‑f

Enhancements:
1️⃣  Exclude materials whose pretreatment width is missing **or ≤ 0** (affects CSV & stats).
2️⃣  Bar‑chart: average‑rolls line and right‑hand y‑axis coloured **orange**.
3️⃣  Write a third output CSV **`subgroup_stats.csv`** containing the stats underlying the chart.

Bug‑fix: previous cut‑off introduced a SyntaxError. Script now ends correctly.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_BETA = -0.7
MAX_SHRINK = 0.02

FILENAMES = {
    "item_additions": "item_additions.csv",
    "rolls": "rolls.csv",
    "inspection": "material_inspection_data.csv",
    "settings": "locked_substation_settings.csv",
    "materials": "materials.csv",
}

OUTPUT_COLUMNS = [
    "roll_name",
    "material_code",
    "mill_lot",
    "shipment_arrive_date",
    "width_avg_mm",
    "sequence_number",
]

# ──────────────────────────── helpers ────────────────────────────────

def compute_width_band(wt: float, w: float, beta: float = DEFAULT_BETA,
                        max_shrinkage_range: float = MAX_SHRINK) -> float:
    """Allowable width range ΔW for ±max_shrinkage_range."""
    if abs(beta) < 1e-9:
        return float("inf")
    return (max_shrinkage_range * w * w) / (abs(beta) * wt)


def _clean_numeric_field(val):
    """Extract mean of all numbers from a string; return NaN if none."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", str(val))
    if not nums:
        return np.nan
    return float(np.mean([float(n) for n in nums]))


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def _find_pretreat_col(settings: pd.DataFrame) -> Optional[str]:
    """Detect the Pretreatment Width 3 column regardless of extra spaces/case."""
    for col in settings.columns:
        cname = col.lower().replace(" ", "")
        if "pretreatment" in cname and "width" in cname and "3" in cname:
            return col
    return None

# ─────────────────────────── data IO ──────────────────────────────────

def load_data(input_dir: pathlib.Path) -> Tuple[pd.DataFrame, ...]:
    def _read(name: str, parse_dates: List[str] | None = None):
        path = input_dir / FILENAMES[name]
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, dtype=str, parse_dates=parse_dates)
        return _strip_cols(df)

    return (
        _read("item_additions", ["Shipment Arrive Date"]),
        _read("rolls"),
        _read("inspection"),
        _read("settings"),
        _read("materials"),
    )

# ─────────────────────────── merging ─────────────────────────────────

def merge_tables(item_additions, rolls, inspection, settings, materials):
    woven_codes = materials.loc[
        materials["Fabric Type"].str.lower() == "woven", "Material Code"
    ].unique().tolist()

    # width averages per roll
    insp = inspection[["Rolls", "Width Average (mm)"]].rename(
        columns={"Rolls": "roll_name", "Width Average (mm)": "width_raw"})
    insp["width_avg_mm"] = insp["width_raw"].apply(_clean_numeric_field)
    width_avg = insp.groupby("roll_name", as_index=False)["width_avg_mm"].mean()

    # rolls ready
    rolls_ready = rolls.rename(columns={"Name": "roll_name"})
    if "State" in rolls_ready.columns:
        rolls_ready = rolls_ready.loc[rolls_ready["State"] == "Ready to print state"].copy()
    rolls_ready["roll_id"] = rolls_ready["roll_name"].str.extract(r"^(R\d{3,6})")

    item_additions = item_additions.rename(columns={"Name": "add_name"})
    item_additions["roll_id"] = item_additions["add_name"].str.extract(r"^(R\d{3,6})")

    pret_col = _find_pretreat_col(settings)
    if pret_col is None:
        raise KeyError("Pretreatment width column not found in settings CSV.")

    merged = (
        rolls_ready
        .merge(item_additions[["roll_id", "Material Code", "Order.Shipments", "Shipment Arrive Date"]],
                on="roll_id", how="left")
        .merge(width_avg, on="roll_name", how="left")
        .merge(settings[["Name", pret_col]], left_on="Material Code", right_on="Name", how="left")
    )

    merged = merged.loc[merged["Material Code"].isin(woven_codes)].copy()

    merged = merged.rename(columns={
        "Order.Shipments": "mill_lot",
        "Shipment Arrive Date": "shipment_arrive_date",
        pret_col: "wt_mm",
        "Material Code": "material_code",
    })

    merged["width_avg_mm"] = merged["width_avg_mm"].astype(float)
    merged["wt_mm"] = merged["wt_mm"].apply(_clean_numeric_field).astype(float)

    # filter invalid pretreatment width
    merged = merged.loc[merged["wt_mm"].notna() & (merged["wt_mm"] > 0)].copy()

    return merged

# ─────────────────────── subgrouping & sequences ─────────────────────

def subgroup_by_width(df_lot: pd.DataFrame, wt: float, beta: float):
    widths = df_lot["width_avg_mm"].astype(float)
    order = widths.sort_values(ascending=False).index
    subgroup_id = {}
    current = 0
    central = None
    dw = None
    for idx in order:
        w = widths.loc[idx]
        if central is None:
            central = w
            dw = compute_width_band(wt, central, beta)
            subgroup_id[idx] = current
            continue
        if abs(w - central) <= dw / 2:
            subgroup_id[idx] = current
        else:
            current += 1
            central = w
            dw = compute_width_band(wt, central, beta)
            subgroup_id[idx] = current
    return pd.Series(subgroup_id, name="subgroup")


def assign_sequences(df: pd.DataFrame, beta: float):
    """Assign subgroup labels and **per‑material subgroup sequence numbers**.

    All rolls in the same (mill_lot, subgroup) inside a material share one
    *sequence_number* (1, 1, 1 … then 2, 2, 2 …) following the hierarchy:
        shipment_arrive_date  →  mill_lot  →  subgroup  →  width desc.
    """
    out = df.copy()
    out["shipment_arrive_date"] = pd.to_datetime(out["shipment_arrive_date"], errors="coerce")

    # Build subgroup labels within each (mill_lot, material)
    labels = []
    for (lot, mat), chunk in out.groupby(["mill_lot", "material_code"], sort=False):
        wt = chunk["wt_mm"].iloc[0]
        labels.append(subgroup_by_width(chunk, wt, beta))
    out["subgroup"] = pd.concat(labels).reindex(out.index)

    # Sort using the required hierarchy, keeping materials separated first
    out = out.sort_values([
        "material_code",
        "shipment_arrive_date",
        "mill_lot",
        "subgroup",
        "width_avg_mm"  # widest first
    ], ascending=[True, True, True, True, False])

    # Create a key that identifies each *group* of rolls which should share a number
    out["_grp_key"] = (
        out["shipment_arrive_date"].dt.strftime("%Y-%m-%d") + "|" +
        out["mill_lot"].fillna("") + "|" + out["subgroup"].astype(str)
    )

    # Map unique keys to 1..n within each material in order of appearance
    def _seq_map(series):
        unique_keys = pd.Series(series.unique(), index=series.unique()).index
        return {k: i + 1 for i, k in enumerate(unique_keys)}

    seq_number = (
        out.groupby("material_code")["_grp_key"]
            .transform(lambda s: s.map(_seq_map(s)))
    )
    out["sequence_number"] = seq_number
    out = out.drop(columns="_grp_key")
    return out

# ─────────────────────── stats & plotting ────────────────────────────

def compute_stats(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Per‑material statistics including average allowable width band (ΔW).

    For each (material, mill_lot, subgroup) we:
        • take the *central* width (widest roll) and wt_mm.
        • compute ΔW via `compute_width_band`.
    We then average those ΔW values per material, so *avg_band_mm* reflects the
    typical subgroup width range that keeps shrinkage variance low.
    """
    # base stats
    stats = (
        df.groupby(["material_code", "subgroup"], as_index=False)["roll_name"].count()
          .rename(columns={"roll_name": "rolls"})
    )

    # ΔW per subgroup: pick first row of each (material,mill_lot,subgroup)
    first_rows = df.sort_values(["material_code","mill_lot","subgroup","width_avg_mm"],ascending=[True,True,True,False]) \
                   .groupby(["material_code","mill_lot","subgroup"], as_index=False).first()
    first_rows["band_mm"] = first_rows.apply(
        lambda r: compute_width_band(r["wt_mm"], r["width_avg_mm"], beta), axis=1
    )
    band_avg = first_rows.groupby("material_code")["band_mm"].mean().rename("avg_band_mm")

    mat = (
        stats.groupby("material_code")
             .agg(total_subgroups=("subgroup", "nunique"),
                  avg_rolls_per_subgroup=("rolls", "mean"))
             .join(band_avg)
             .sort_values("total_subgroups", ascending=False)
    )
    return mat


def make_bar_chart(mat: pd.DataFrame, out_path: pathlib.Path):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(mat))
    ax1.bar(x, mat["total_subgroups"], label="Total sub‑groups")
    ax1.set_ylabel("Total Sub‑groups")
    ax1.set_xlabel("Material code")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mat.index, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, mat["avg_rolls_per_subgroup"], color="orange", marker="o", linestyle="--", label="Avg rolls / subgroup")
    ax2.set_ylabel("Average rolls / sub‑group", color="orange")
    ax2.tick_params(axis="y", colors="orange")

    plt.title("Sub‑group stats (woven materials — pretreat width > 0)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# ───────────────────────────── main ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Group woven rolls to curb shrinkage variance")
    ap.add_argument("--input-dir", required=True, type=pathlib.Path)
    ap.add_argument("--output-dir", type=pathlib.Path, default="output")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--max-shrink", type=float, default=MAX_SHRINK)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    item_additions, rolls, inspection, settings, materials = load_data(args.input_dir)
    merged = merge_tables(item_additions, rolls, inspection, settings, materials)
    if merged.empty:
        raise SystemExit("No woven rolls with valid pretreatment width found — check inputs.")

    seq_df = assign_sequences(merged, args.beta)
    mat_stats = compute_stats(seq_df, args.beta)

    csv1 = args.output_dir / "rolls_with_sequence.csv"
    seq_df[OUTPUT_COLUMNS].to_csv(csv1, index=False)
    print(f"💾 Rolls CSV     → {csv1.resolve()}")

    csv2 = args.output_dir / "subgroup_stats.csv"
    mat_stats.to_csv(csv2)
    print(f"💾 Stats CSV     → {csv2.resolve()}")

    chart = args.output_dir / "subgroup_stats.png"
    make_bar_chart(mat_stats, chart)
    print(f"📊 Chart saved   → {chart.resolve()}")


if __name__ == "__main__":
    main()
