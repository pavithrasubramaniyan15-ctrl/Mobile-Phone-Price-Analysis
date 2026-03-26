"""
main.py
-------
Entry point for the Mobile Phone Price Analysis EDA project.

Usage
-----
    python main.py

All four charts will be saved to the ``outputs/`` folder.
"""

import os
import sys
import time

# Add the project root to sys.path so sub-module imports work regardless
# of where the user runs the script from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader   import load_dataset, clean_dataset, get_data_summary
from src.analysis      import run_all_analyses
from src.visualization import generate_all_charts

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "mobile_phone_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def print_banner() -> None:
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║         📱  MOBILE PHONE PRICE ANALYSIS  –  EDA             ║
║              Senior Data Scientist Project                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_summary(summary: dict) -> None:
    print("\n── Dataset Summary ──────────────────────────────────────────")
    print(f"  Rows × Columns  : {summary['shape'][0]:,} × {summary['shape'][1]}")
    print(f"  Brands          : {', '.join(summary['brands'])}")
    print(f"  RAM values (GB) : {summary['ram_values']}")
    null_total = sum(summary["null_counts"].values())
    print(f"  Total nulls     : {null_total}")
    print("─────────────────────────────────────────────────────────────\n")


def main() -> None:
    start = time.time()
    print_banner()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("Step 1/4  →  Loading dataset …")
    df = load_dataset(DATA_PATH)

    # ── 2. Summarise & clean ─────────────────────────────────────────────────
    print("\nStep 2/4  →  Inspecting & cleaning …")
    summary = get_data_summary(df)
    print_summary(summary)
    df = clean_dataset(df)

    # ── 3. Analyse ───────────────────────────────────────────────────────────
    print("\nStep 3/4  →  Running analyses …")
    analyses = run_all_analyses(df)

    # Print key findings
    corr_ram    = analyses["price_ram"]["correlation"]
    corr_camera = analyses["camera"]["correlation"]
    print(f"\n  ✔ Price ↔ RAM correlation      :  r = {corr_ram:.4f}")
    print(f"  ✔ Price ↔ Camera MP correlation:  r = {corr_camera:.4f}")

    top_brand = analyses["brand"]["brand_avg"].iloc[0]
    print(f"  ✔ Most expensive brand on avg  :  {top_brand['brand']}  "
          f"(${top_brand['mean_price']:,.2f})")

    low_brand = analyses["brand"]["brand_avg"].iloc[-1]
    print(f"  ✔ Most affordable brand on avg :  {low_brand['brand']}  "
          f"(${low_brand['mean_price']:,.2f})")

    # ── 4. Visualise ─────────────────────────────────────────────────────────
    print(f"\nStep 4/4  →  Generating charts → {OUTPUT_DIR}")
    saved_files = generate_all_charts(analyses, OUTPUT_DIR)

    elapsed = time.time() - start
    print(f"\n✅  All done in {elapsed:.1f}s.  {len(saved_files)} charts saved:\n")
    for path in saved_files:
        print(f"   📊  {path}")
    print()


if __name__ == "__main__":
    main()
