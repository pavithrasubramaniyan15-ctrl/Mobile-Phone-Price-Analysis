"""
visualization.py
----------------
Generates and saves all EDA charts for the Mobile Phone Price Analysis project.

Chart palette follows a consistent dark-accent theme so every PNG looks
cohesive when viewed side-by-side.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Global style ────────────────────────────────────────────────────────────
PALETTE      = "viridis"
ACCENT       = "#5B8EFF"       # single-series highlight colour
BG_COLOR     = "#0F1117"
GRID_COLOR   = "#2A2D3A"
TEXT_COLOR   = "#E8ECF4"
SPINE_COLOR  = "#3A3D4A"
FONT_FAMILY  = "DejaVu Sans"

def _apply_dark_theme() -> None:
    """Apply a consistent dark theme to all subsequent Matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":  BG_COLOR,
        "axes.facecolor":    "#161B27",
        "axes.edgecolor":    SPINE_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "axes.titlecolor":   TEXT_COLOR,
        "axes.grid":         True,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.6,
        "grid.linestyle":    "--",
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "legend.facecolor":  "#1C2130",
        "legend.edgecolor":  SPINE_COLOR,
        "font.family":       FONT_FAMILY,
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    12,
    })

_apply_dark_theme()

# ── Helper ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved → {path}")
    return path


def _add_watermark(ax: plt.Axes, text: str = "Mobile Phone Price Analysis") -> None:
    ax.text(
        0.99, 0.02, text,
        transform=ax.transAxes,
        fontsize=8, color="#555B6E",
        ha="right", va="bottom", style="italic",
    )


# ── Chart 1: Price vs RAM ────────────────────────────────────────────────────

def plot_price_vs_ram(result: dict, output_dir: str) -> str:
    """
    Scatter plot of RAM vs Price with a regression line and
    per-RAM mean overlay.
    """
    data        = result["data"]
    ram_summary = result["ram_summary"]
    corr        = result["correlation"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)
    fig.suptitle("Price vs RAM Analysis", fontsize=16, fontweight="bold",
                 color=TEXT_COLOR, y=1.01)

    # ── Left: scatter + regression ──
    ax = axes[0]
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    sc = ax.scatter(
        data["ram"] + jitter, data["price_usd"],
        c=data["price_usd"], cmap=PALETTE,
        alpha=0.55, s=40, edgecolors="none",
    )
    # Regression line
    m, b = np.polyfit(data["ram"], data["price_usd"], 1)
    x_line = np.linspace(data["ram"].min(), data["ram"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="#FF6B6B", lw=2.0,
            label=f"Regression  (r = {corr:.2f})")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Price (USD)", color=TEXT_COLOR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    ax.set_xlabel("RAM (GB)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Scatter: RAM vs Price")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(data["ram"].unique())))
    _add_watermark(ax)

    # ── Right: bar of mean prices per RAM tier ──
    ax2 = axes[1]
    colors = sns.color_palette(PALETTE, len(ram_summary))
    bars = ax2.bar(
        ram_summary["ram"].astype(str) + " GB",
        ram_summary["mean_price"],
        color=colors, edgecolor=SPINE_COLOR, linewidth=0.8,
        width=0.55,
    )
    # Annotate bars
    for bar, val in zip(bars, ram_summary["mean_price"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"${val:.0f}", ha="center", va="bottom",
            fontsize=9, color=TEXT_COLOR,
        )

    ax2.set_xlabel("RAM Tier")
    ax2.set_ylabel("Average Price (USD)")
    ax2.set_title("Mean Price per RAM Tier")
    ax2.set_ylim(0, ram_summary["mean_price"].max() * 1.18)
    _add_watermark(ax2)

    fig.tight_layout()
    return _save(fig, output_dir, "price_vs_ram.png")


# ── Chart 2: Brand Comparison ────────────────────────────────────────────────

def plot_brand_comparison(result: dict, output_dir: str) -> str:
    """
    Horizontal bar chart of average price per brand with count labels.
    """
    brand_avg = result["brand_avg"].sort_values("mean_price")

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_COLOR)

    colors = sns.color_palette(PALETTE, len(brand_avg))
    bars = ax.barh(
        brand_avg["brand"],
        brand_avg["mean_price"],
        color=colors, edgecolor=SPINE_COLOR, linewidth=0.8,
        height=0.6,
    )

    # Value labels at end of each bar
    for bar, val, cnt in zip(bars, brand_avg["mean_price"], brand_avg["count"]):
        ax.text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}  (n={cnt})",
            va="center", fontsize=10, color=TEXT_COLOR,
        )

    ax.set_xlabel("Average Price (USD)")
    ax.set_title("Average Phone Price by Brand", pad=12)
    ax.set_xlim(0, brand_avg["mean_price"].max() * 1.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _add_watermark(ax)

    fig.tight_layout()
    return _save(fig, output_dir, "brand_comparison.png")


# ── Chart 3: Battery Capacity Trends ────────────────────────────────────────

def plot_battery_trend(result: dict, output_dir: str) -> str:
    """
    Dual-panel: (left) battery distribution bar chart,
                (right) mean price line plot per battery tier.
    """
    battery_dist      = result["battery_dist"]
    battery_price_avg = result["battery_price_avg"]

    # Normalise column names defensively
    if battery_dist.columns[0] != "battery_capacity":
        battery_dist.columns = ["battery_capacity", "count"]

    battery_dist      = battery_dist.sort_values("battery_capacity")
    battery_price_avg = battery_price_avg.sort_values("battery_capacity")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)
    fig.suptitle("Battery Capacity Trends", fontsize=16, fontweight="bold",
                 color=TEXT_COLOR, y=1.01)

    # ── Left: distribution ──
    ax = axes[0]
    colors = sns.color_palette(PALETTE, len(battery_dist))
    bars = ax.bar(
        battery_dist["battery_capacity"].astype(str),
        battery_dist["count"],
        color=colors, edgecolor=SPINE_COLOR, linewidth=0.8,
        width=0.65,
    )
    for bar, cnt in zip(bars, battery_dist["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(cnt), ha="center", va="bottom",
            fontsize=9, color=TEXT_COLOR,
        )
    ax.set_xlabel("Battery Capacity (mAh)")
    ax.set_ylabel("Number of Phones")
    ax.set_title("Distribution of Battery Capacities")
    ax.tick_params(axis="x", rotation=30)
    _add_watermark(ax)

    # ── Right: price trend line ──
    ax2 = axes[1]
    x_vals = battery_price_avg["battery_capacity"].astype(int)
    y_vals = battery_price_avg["mean_price"]

    ax2.plot(x_vals, y_vals, color=ACCENT, lw=2.5, marker="o",
             markersize=8, markerfacecolor="#FF6B6B", markeredgecolor="white",
             markeredgewidth=1.2, label="Mean Price")
    ax2.fill_between(x_vals, y_vals, alpha=0.15, color=ACCENT)

    for x, y in zip(x_vals, y_vals):
        ax2.text(x, y + 8, f"${y:.0f}", ha="center", fontsize=8.5,
                 color=TEXT_COLOR)

    ax2.set_xlabel("Battery Capacity (mAh)")
    ax2.set_ylabel("Average Price (USD)")
    ax2.set_title("Average Price vs Battery Capacity")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)
    _add_watermark(ax2)

    fig.tight_layout()
    return _save(fig, output_dir, "battery_trend.png")


# ── Chart 4: Camera MP vs Price ──────────────────────────────────────────────

def plot_camera_vs_price(result: dict, output_dir: str) -> str:
    """
    Scatter coloured by price_range + binned mean-price bar chart.
    """
    data   = result["data"]
    binned = result["binned"]
    corr   = result["correlation"]

    price_range_labels = {
        0: "Budget",
        1: "Mid-range",
        2: "Premium",
        3: "Flagship",
    }
    range_colors = {
        0: "#4CAF82",
        1: "#5B8EFF",
        2: "#FF9F43",
        3: "#FF6B6B",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)
    fig.suptitle("Camera Megapixels vs Price Analysis", fontsize=16,
                 fontweight="bold", color=TEXT_COLOR, y=1.01)

    # ── Left: scatter by price range ──
    ax = axes[0]
    for pr, grp in data.groupby("price_range"):
        label = price_range_labels.get(pr, str(pr))
        ax.scatter(
            grp["primary_camera_mp"], grp["price_usd"],
            c=range_colors.get(pr, ACCENT),
            label=label, alpha=0.65, s=45, edgecolors="none",
        )

    # Regression line
    m, b = np.polyfit(data["primary_camera_mp"], data["price_usd"], 1)
    x_line = np.linspace(data["primary_camera_mp"].min(),
                         data["primary_camera_mp"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="white", lw=1.8,
            linestyle="--", label=f"Regression  (r = {corr:.2f})")

    ax.set_xlabel("Primary Camera (MP)")
    ax.set_ylabel("Price (USD)")
    ax.set_title("Scatter: Camera MP vs Price")
    ax.legend(fontsize=9)
    _add_watermark(ax)

    # ── Right: mean price per camera bin ──
    ax2 = axes[1]
    colors = sns.color_palette(PALETTE, len(binned))
    bars = ax2.bar(
        binned["camera_bin"].astype(str),
        binned["mean_price"],
        color=colors, edgecolor=SPINE_COLOR, linewidth=0.8,
        width=0.6,
    )
    for bar, val, cnt in zip(bars, binned["mean_price"], binned["count"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"${val:.0f}\n(n={cnt})",
            ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR,
        )

    ax2.set_xlabel("Camera Resolution Tier")
    ax2.set_ylabel("Average Price (USD)")
    ax2.set_title("Mean Price per Camera Tier")
    ax2.set_ylim(0, binned["mean_price"].max() * 1.22)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.tick_params(axis="x", rotation=15)
    _add_watermark(ax2)

    fig.tight_layout()
    return _save(fig, output_dir, "camera_vs_price.png")


# ── Master runner ─────────────────────────────────────────────────────────────

def generate_all_charts(analyses: dict, output_dir: str) -> list[str]:
    """
    Generate and save all four EDA charts.

    Parameters
    ----------
    analyses : dict
        Output of ``analysis.run_all_analyses(df)``.
    output_dir : str
        Directory where PNG files will be written.

    Returns
    -------
    list[str]
        Sorted list of saved file paths.
    """
    saved = []
    saved.append(plot_price_vs_ram(analyses["price_ram"], output_dir))
    saved.append(plot_brand_comparison(analyses["brand"],   output_dir))
    saved.append(plot_battery_trend(analyses["battery"],   output_dir))
    saved.append(plot_camera_vs_price(analyses["camera"],  output_dir))
    return saved
