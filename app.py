"""
app.py  –  Streamlit web dashboard for Mobile Phone Price Analysis
-------------------------------------------------------------------
Entry point for Render deployment.
Run locally :  streamlit run app.py
"""

import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

# ── make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader   import load_dataset, clean_dataset
from src.analysis      import run_all_analyses
from src.visualization import (
    plot_price_vs_ram,
    plot_brand_comparison,
    plot_battery_trend,
    plot_camera_vs_price,
    _apply_dark_theme,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Mobile Phone Price Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0F1117; color: #E8ECF4; }
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B27; }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1C2130;
        border: 1px solid #2A2D3A;
        border-radius: 10px;
        padding: 12px 18px;
    }
    /* Headers */
    h1, h2, h3 { color: #E8ECF4 !important; }
    /* Divider */
    hr { border-color: #2A2D3A; }
    /* DataFrame */
    .stDataFrame { background-color: #161B27; }
</style>
""", unsafe_allow_html=True)

_apply_dark_theme()

# ── Helper: render matplotlib figure in Streamlit ────────────────────────────

def show_fig(fig: plt.Figure):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ── Load & cache data ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def get_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "mobile_phone_dataset.csv")
    df = load_dataset(data_path)
    df = clean_dataset(df)
    return df

@st.cache_data(show_spinner="Running analyses …")
def get_analyses(_df):          # leading underscore → skip hashing the df arg
    return run_all_analyses(_df)


df        = get_data()
analyses  = get_analyses(df)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📱 Mobile Phone\nPrice Analysis")
    st.markdown("---")

    # Brand filter
    all_brands = sorted(df["brand"].unique())
    sel_brands = st.multiselect(
        "Filter by Brand",
        options=all_brands,
        default=all_brands,
    )

    # RAM filter
    all_ram = sorted(df["ram"].unique())
    sel_ram = st.multiselect(
        "Filter by RAM (GB)",
        options=all_ram,
        default=all_ram,
    )

    # Price range slider
    min_p, max_p = int(df["price_usd"].min()), int(df["price_usd"].max())
    price_range_sel = st.slider(
        "Price Range (USD)",
        min_value=min_p,
        max_value=max_p,
        value=(min_p, max_p),
        step=50,
    )

    st.markdown("---")
    st.markdown(
        "**Dataset:** Mobile Phone Specifications  \n"
        "**Source:** Kaggle  \n"
        "**Records:** {:,}".format(len(df))
    )
    st.markdown("---")
    st.markdown("Built with 🐍 Python · Streamlit · Seaborn")


# Apply sidebar filters to a working copy
mask = (
    df["brand"].isin(sel_brands) &
    df["ram"].isin(sel_ram) &
    df["price_usd"].between(*price_range_sel)
)
df_filtered = df[mask].reset_index(drop=True)

# Re-run analyses on filtered data (cached separately)
analyses_filtered = run_all_analyses(df_filtered) if len(df_filtered) > 5 else analyses


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center;'>📱 Mobile Phone Price Analysis</h1>"
    "<p style='text-align:center; color:#8891A5;'>Exploratory Data Analysis Dashboard</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# KPI METRICS
# ═══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("📦 Total Phones",     f"{len(df_filtered):,}")
c2.metric("💰 Avg Price",        f"${df_filtered['price_usd'].mean():,.0f}")
c3.metric("💡 Median Price",     f"${df_filtered['price_usd'].median():,.0f}")
c4.metric("🔋 Avg Battery",      f"{df_filtered['battery_capacity'].mean():,.0f} mAh")
c5.metric("📸 Avg Camera",       f"{df_filtered['primary_camera_mp'].mean():.0f} MP")

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💾 Price vs RAM",
    "🏷️ Brand Comparison",
    "🔋 Battery Trends",
    "📸 Camera vs Price",
    "📋 Raw Data",
])

# ── Tab 1: Price vs RAM ───────────────────────────────────────────────────────
with tab1:
    st.subheader("Price vs RAM Analysis")
    corr = analyses_filtered["price_ram"]["correlation"]
    st.info(f"**Pearson Correlation (RAM ↔ Price):** r = `{corr:.4f}` — "
            f"{'Strong' if abs(corr)>0.7 else 'Moderate' if abs(corr)>0.4 else 'Weak'} "
            f"{'positive' if corr>0 else 'negative'} correlation")

    # Render chart into a temp buffer
    TMPDIR = "/tmp/eda_outputs"
    os.makedirs(TMPDIR, exist_ok=True)
    fig = plot_price_vs_ram.__wrapped__(analyses_filtered["price_ram"], TMPDIR) \
          if hasattr(plot_price_vs_ram, "__wrapped__") else None

    # Direct plot approach (avoids file I/O dependency)
    data        = analyses_filtered["price_ram"]["data"]
    ram_summary = analyses_filtered["price_ram"]["ram_summary"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0F1117")
    fig.suptitle("Price vs RAM Analysis", fontsize=15, fontweight="bold",
                 color="#E8ECF4", y=1.01)

    ax = axes[0]
    jitter = np.random.uniform(-0.15, 0.15, size=len(data))
    sc = ax.scatter(data["ram"] + jitter, data["price_usd"],
                    c=data["price_usd"], cmap="viridis", alpha=0.5, s=35, edgecolors="none")
    m, b = np.polyfit(data["ram"], data["price_usd"], 1)
    x_line = np.linspace(data["ram"].min(), data["ram"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="#FF6B6B", lw=2, label=f"r = {corr:.2f}")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Price (USD)", color="#E8ECF4")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#E8ECF4")
    ax.set_xlabel("RAM (GB)"); ax.set_ylabel("Price (USD)")
    ax.set_title("Scatter: RAM vs Price"); ax.legend()
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(data["ram"].unique())))

    ax2 = axes[1]
    colors = sns.color_palette("viridis", len(ram_summary))
    bars = ax2.bar(ram_summary["ram"].astype(str) + " GB", ram_summary["mean_price"],
                   color=colors, edgecolor="#3A3D4A", width=0.55)
    for bar, val in zip(bars, ram_summary["mean_price"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+8,
                 f"${val:.0f}", ha="center", fontsize=9, color="#E8ECF4")
    ax2.set_xlabel("RAM Tier"); ax2.set_ylabel("Average Price (USD)")
    ax2.set_title("Mean Price per RAM Tier")
    ax2.set_ylim(0, ram_summary["mean_price"].max() * 1.18)
    fig.tight_layout()
    show_fig(fig)

    with st.expander("📊 RAM Summary Table"):
        st.dataframe(ram_summary.rename(columns={
            "ram": "RAM (GB)", "mean_price": "Avg Price ($)",
            "median_price": "Median Price ($)", "count": "Count"
        }).set_index("RAM (GB)").style.format("${:.2f}", subset=["Avg Price ($)", "Median Price ($)"]))


# ── Tab 2: Brand Comparison ───────────────────────────────────────────────────
with tab2:
    st.subheader("Brand Comparison — Average Price")
    brand_avg = analyses_filtered["brand"]["brand_avg"]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0F1117")
    brand_sorted = brand_avg.sort_values("mean_price")
    colors = sns.color_palette("viridis", len(brand_sorted))
    bars = ax.barh(brand_sorted["brand"], brand_sorted["mean_price"],
                   color=colors, edgecolor="#3A3D4A", height=0.6)
    for bar, val, cnt in zip(bars, brand_sorted["mean_price"], brand_sorted["count"]):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
                f"${val:,.0f}  (n={cnt})", va="center", fontsize=10, color="#E8ECF4")
    ax.set_xlabel("Average Price (USD)"); ax.set_title("Average Phone Price by Brand", pad=12)
    ax.set_xlim(0, brand_sorted["mean_price"].max() * 1.28)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    show_fig(fig)

    with st.expander("📊 Brand Summary Table"):
        st.dataframe(
            brand_avg.rename(columns={"brand": "Brand", "mean_price": "Avg Price ($)", "count": "Count"})
            .set_index("Brand")
            .sort_values("Avg Price ($)", ascending=False)
            .style.format("${:.2f}", subset=["Avg Price ($)"])
            .bar(subset=["Avg Price ($)"], color="#5B8EFF")
        )


# ── Tab 3: Battery Trends ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Battery Capacity Trends")
    battery_dist      = analyses_filtered["battery"]["battery_dist"]
    battery_price_avg = analyses_filtered["battery"]["battery_price_avg"]

    if battery_dist.columns[0] != "battery_capacity":
        battery_dist.columns = ["battery_capacity", "count"]
    battery_dist      = battery_dist.sort_values("battery_capacity")
    battery_price_avg = battery_price_avg.sort_values("battery_capacity")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0F1117")
    fig.suptitle("Battery Capacity Trends", fontsize=15, fontweight="bold", color="#E8ECF4", y=1.01)

    ax = axes[0]
    colors = sns.color_palette("viridis", len(battery_dist))
    bars = ax.bar(battery_dist["battery_capacity"].astype(str),
                  battery_dist["count"], color=colors, edgecolor="#3A3D4A", width=0.65)
    for bar, cnt in zip(bars, battery_dist["count"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(cnt), ha="center", fontsize=9, color="#E8ECF4")
    ax.set_xlabel("Battery (mAh)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Battery Capacities")
    ax.tick_params(axis="x", rotation=30)

    ax2 = axes[1]
    x_vals = battery_price_avg["battery_capacity"].astype(int)
    y_vals = battery_price_avg["mean_price"]
    ax2.plot(x_vals, y_vals, color="#5B8EFF", lw=2.5, marker="o",
             markersize=8, markerfacecolor="#FF6B6B", markeredgecolor="white",
             markeredgewidth=1.2, label="Mean Price")
    ax2.fill_between(x_vals, y_vals, alpha=0.12, color="#5B8EFF")
    for x, y in zip(x_vals, y_vals):
        ax2.text(x, y + 8, f"${y:.0f}", ha="center", fontsize=8.5, color="#E8ECF4")
    ax2.set_xlabel("Battery (mAh)"); ax2.set_ylabel("Average Price (USD)")
    ax2.set_title("Average Price vs Battery Capacity")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.legend(); ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    show_fig(fig)


# ── Tab 4: Camera vs Price ────────────────────────────────────────────────────
with tab4:
    st.subheader("Camera Megapixels vs Price")
    corr_cam = analyses_filtered["camera"]["correlation"]
    st.info(f"**Pearson Correlation (Camera MP ↔ Price):** r = `{corr_cam:.4f}`")

    data_cam = analyses_filtered["camera"]["data"]
    binned   = analyses_filtered["camera"]["binned"]

    price_range_labels = {0: "Budget", 1: "Mid-range", 2: "Premium", 3: "Flagship"}
    range_colors       = {0: "#4CAF82", 1: "#5B8EFF", 2: "#FF9F43", 3: "#FF6B6B"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0F1117")
    fig.suptitle("Camera MP vs Price", fontsize=15, fontweight="bold", color="#E8ECF4", y=1.01)

    ax = axes[0]
    for pr, grp in data_cam.groupby("price_range"):
        ax.scatter(grp["primary_camera_mp"], grp["price_usd"],
                   c=range_colors.get(pr, "#5B8EFF"),
                   label=price_range_labels.get(pr, str(pr)),
                   alpha=0.6, s=40, edgecolors="none")
    m, b = np.polyfit(data_cam["primary_camera_mp"], data_cam["price_usd"], 1)
    x_line = np.linspace(data_cam["primary_camera_mp"].min(),
                         data_cam["primary_camera_mp"].max(), 200)
    ax.plot(x_line, m * x_line + b, color="white", lw=1.8, linestyle="--",
            label=f"Regression (r={corr_cam:.2f})")
    ax.set_xlabel("Camera (MP)"); ax.set_ylabel("Price (USD)")
    ax.set_title("Scatter: Camera MP vs Price"); ax.legend(fontsize=9)

    ax2 = axes[1]
    colors = sns.color_palette("viridis", len(binned))
    bars = ax2.bar(binned["camera_bin"].astype(str), binned["mean_price"],
                   color=colors, edgecolor="#3A3D4A", width=0.6)
    for bar, val, cnt in zip(bars, binned["mean_price"], binned["count"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 f"${val:.0f}\n(n={cnt})", ha="center", fontsize=8.5, color="#E8ECF4")
    ax2.set_xlabel("Camera Tier"); ax2.set_ylabel("Avg Price (USD)")
    ax2.set_title("Mean Price per Camera Tier")
    ax2.set_ylim(0, binned["mean_price"].max() * 1.22)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    show_fig(fig)

    with st.expander("📊 Camera Bin Table"):
        st.dataframe(binned.rename(columns={
            "camera_bin": "Camera Tier", "mean_price": "Avg Price ($)", "count": "Count"
        }).set_index("Camera Tier").style.format("${:.2f}", subset=["Avg Price ($)"]))


# ── Tab 5: Raw Data ───────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Raw Dataset — {len(df_filtered):,} rows (filtered)")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        search_brand = st.text_input("🔍 Search brand", "")
    with col_b:
        sort_col = st.selectbox("Sort by", ["price_usd", "ram", "battery_capacity",
                                             "primary_camera_mp"], index=0)

    disp = df_filtered.copy()
    if search_brand:
        disp = disp[disp["brand"].str.contains(search_brand, case=False)]
    disp = disp.sort_values(sort_col, ascending=False)

    st.dataframe(
        disp.style.format({"price_usd": "${:.2f}"}),
        use_container_width=True,
        height=420,
    )

    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download filtered CSV",
        data=csv,
        file_name="mobile_phones_filtered.csv",
        mime="text/csv",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555B6E; font-size:13px;'>"
    "📱 Mobile Phone Price Analysis · Built with Python, Streamlit & Seaborn"
    "</p>",
    unsafe_allow_html=True,
)
