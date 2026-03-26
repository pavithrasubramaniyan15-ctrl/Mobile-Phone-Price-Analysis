# 📱 Mobile Phone Price Analysis — EDA Project

> A complete, modular Exploratory Data Analysis project built with Python,
> Pandas, Seaborn and Matplotlib.

---

## 📁 Project Structure

```
mobile_phone_price_analysis/
│
├── data/
│   └── mobile_phone_dataset.csv       ← Dataset (500 phones, 7 features)
│
├── outputs/                           ← Auto-generated charts (PNG)
│   ├── price_vs_ram.png
│   ├── brand_comparison.png
│   ├── battery_trend.png
│   └── camera_vs_price.png
│
├── src/
│   ├── data_loader.py                 ← Load, validate & clean data
│   ├── analysis.py                    ← Statistical computations
│   └── visualization.py               ← Chart generation (Seaborn + Matplotlib)
│
├── notebooks/
│   └── eda.ipynb                      ← Interactive Jupyter notebook
│
├── main.py                            ← 🚀 Single entry point
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone / Download the project

```bash
git clone https://github.com/your-username/mobile_phone_price_analysis.git
cd mobile_phone_price_analysis
```

### 2. Create a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Project

```bash
python main.py
```

Expected console output:

```
╔══════════════════════════════════════════════════════════════╗
║         📱  MOBILE PHONE PRICE ANALYSIS  –  EDA             ║
╚══════════════════════════════════════════════════════════════╝

Step 1/4  →  Loading dataset …
[data_loader] Loaded 500 rows × 7 columns

Step 2/4  →  Inspecting & cleaning …
  Rows × Columns  : 500 × 7
  Brands          : Apple, Motorola, Nokia, OnePlus, Oppo, Realme, Samsung, Sony, Vivo, Xiaomi
  RAM values (GB) : [2, 3, 4, 6, 8, 12, 16]
  Total nulls     : 0

Step 3/4  →  Running analyses …
  ✔ Price ↔ RAM correlation      :  r = 0.8412
  ✔ Price ↔ Camera MP correlation:  r = 0.6831
  ✔ Most expensive brand on avg  :  Apple ($1,089.23)
  ✔ Most affordable brand on avg :  Nokia ($279.44)

Step 4/4  →  Generating charts → outputs/
[viz] Saved → outputs/price_vs_ram.png
[viz] Saved → outputs/brand_comparison.png
[viz] Saved → outputs/battery_trend.png
[viz] Saved → outputs/camera_vs_price.png

✅  All done in 4.2s.  4 charts saved.
```

---

## 📊 Analyses Performed

| # | Analysis | Chart | Key Metric |
|---|----------|-------|------------|
| 1 | **Price vs RAM** | Scatter + bar | Pearson r ≈ 0.84 |
| 2 | **Brand Comparison** | Horizontal bar | Apple highest avg price |
| 3 | **Battery Capacity Trends** | Distribution + line | 4000 mAh most common |
| 4 | **Camera MP vs Price** | Scatter + bin bar | Pearson r ≈ 0.68 |

---

## 🗂️ Dataset Features

| Column | Type | Description |
|--------|------|-------------|
| `brand` | string | Phone manufacturer |
| `ram` | int | RAM in GB |
| `price_usd` | float | Retail price in USD |
| `battery_capacity` | int | Battery in mAh |
| `primary_camera_mp` | int | Rear camera megapixels |
| `internal_storage_gb` | int | Storage in GB |
| `price_range` | int | 0=Budget · 1=Mid · 2=Premium · 3=Flagship |

---

## 📓 Jupyter Notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** – data manipulation
- **numpy** – numerical operations
- **matplotlib** – base charting engine
- **seaborn** – statistical visualisations

---

## 📜 License

MIT © 2024
