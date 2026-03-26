"""
analysis.py
-----------
Analytical computations on the mobile phone dataset.
Keeps all business logic separate from I/O and visualisation.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# 1. Price vs RAM
# ──────────────────────────────────────────────

def price_vs_ram(df: pd.DataFrame) -> dict:
    """
    Compute correlation and per-RAM-tier summary statistics.

    Returns
    -------
    dict with keys:
        'correlation'   – Pearson r between ram and price_usd
        'ram_summary'   – DataFrame: mean/median/count per RAM value
        'data'          – filtered DataFrame used for plotting
    """
    data = df[["ram", "price_usd"]].dropna()

    correlation = data["ram"].corr(data["price_usd"])

    ram_summary = (
        data.groupby("ram")["price_usd"]
        .agg(mean_price="mean", median_price="median", count="count")
        .reset_index()
        .sort_values("ram")
    )

    print(f"[analysis] Price vs RAM  |  Pearson r = {correlation:.4f}")
    return {
        "correlation": correlation,
        "ram_summary": ram_summary,
        "data": data,
    }


# ──────────────────────────────────────────────
# 2. Brand Comparison
# ──────────────────────────────────────────────

def brand_comparison(df: pd.DataFrame) -> dict:
    """
    Compute average price per brand, sorted descending.

    Returns
    -------
    dict with keys:
        'brand_avg'  – DataFrame: brand | mean_price | count
    """
    brand_avg = (
        df.groupby("brand")["price_usd"]
        .agg(mean_price="mean", count="count")
        .reset_index()
        .sort_values("mean_price", ascending=False)
    )

    print("[analysis] Brand comparison:")
    for _, row in brand_avg.iterrows():
        print(f"  {row['brand']:<12}  avg ${row['mean_price']:>7.2f}  (n={row['count']})")

    return {"brand_avg": brand_avg}


# ──────────────────────────────────────────────
# 3. Battery Capacity Trends
# ──────────────────────────────────────────────

def battery_trends(df: pd.DataFrame) -> dict:
    """
    Compute distribution counts and average price per battery tier.

    Returns
    -------
    dict with keys:
        'battery_dist'       – value_counts of battery_capacity
        'battery_price_avg'  – DataFrame: battery_capacity | mean_price
        'data'               – column slice used for plotting
    """
    data = df[["battery_capacity", "price_usd"]].dropna()

    battery_dist = (
        data["battery_capacity"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "battery_capacity", "count": "count"})
        .sort_values("battery_capacity")
    )
    # pandas ≥ 2.0 value_counts already names column correctly
    if "battery_capacity" not in battery_dist.columns:
        battery_dist.columns = ["battery_capacity", "count"]

    battery_price_avg = (
        data.groupby("battery_capacity")["price_usd"]
        .mean()
        .reset_index()
        .rename(columns={"price_usd": "mean_price"})
        .sort_values("battery_capacity")
    )

    print(f"[analysis] Battery tiers found: {sorted(data['battery_capacity'].unique())}")
    return {
        "battery_dist": battery_dist,
        "battery_price_avg": battery_price_avg,
        "data": data,
    }


# ──────────────────────────────────────────────
# 4. Camera Megapixels vs Price
# ──────────────────────────────────────────────

def camera_vs_price(df: pd.DataFrame) -> dict:
    """
    Compute correlation between primary camera MP and price,
    plus a binned summary.

    Returns
    -------
    dict with keys:
        'correlation'  – Pearson r
        'binned'       – DataFrame: camera_bin | mean_price | count
        'data'         – filtered DataFrame used for plotting
    """
    data = df[["primary_camera_mp", "price_usd", "price_range"]].dropna()
    data = data[data["primary_camera_mp"] > 0]

    correlation = data["primary_camera_mp"].corr(data["price_usd"])

    # Bin camera MP into labelled ranges
    bins   = [0, 12, 24, 48, 64, 108, 500]
    labels = ["≤12 MP", "13–24 MP", "25–48 MP", "49–64 MP", "65–108 MP", ">108 MP"]
    data = data.copy()
    data["camera_bin"] = pd.cut(
        data["primary_camera_mp"],
        bins=bins,
        labels=labels,
        right=True,
    )

    binned = (
        data.groupby("camera_bin", observed=True)["price_usd"]
        .agg(mean_price="mean", count="count")
        .reset_index()
    )

    print(f"[analysis] Camera vs Price  |  Pearson r = {correlation:.4f}")
    return {
        "correlation": correlation,
        "binned": binned,
        "data": data,
    }


# ──────────────────────────────────────────────
# Convenience wrapper
# ──────────────────────────────────────────────

def run_all_analyses(df: pd.DataFrame) -> dict:
    """
    Run every analysis and return all results in one dict.

    Parameters
    ----------
    df : pd.DataFrame  – cleaned mobile phone dataset

    Returns
    -------
    dict
        Keys: 'price_ram', 'brand', 'battery', 'camera'
    """
    return {
        "price_ram": price_vs_ram(df),
        "brand":     brand_comparison(df),
        "battery":   battery_trends(df),
        "camera":    camera_vs_price(df),
    }
