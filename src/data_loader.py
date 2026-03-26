"""
data_loader.py
--------------
Handles loading and initial validation of the mobile phone dataset.
"""

import pandas as pd
import os


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the mobile phone dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded and lightly validated DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If required columns are missing from the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    # --- Column validation ---
    required_columns = [
        "brand",
        "ram",
        "price_usd",
        "battery_capacity",
        "primary_camera_mp",
        "internal_storage_gb",
        "price_range",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    print(f"[data_loader] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a concise summary of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Dictionary with shape, dtypes, null counts, and basic statistics.
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict(),
        "brands": sorted(df["brand"].unique().tolist()),
        "ram_values": sorted(df["ram"].unique().tolist()),
        "price_range_labels": {
            0: "Budget (<$300)",
            1: "Mid-range ($300–$599)",
            2: "Premium ($600–$899)",
            3: "Flagship (≥$900)",
        },
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal cleaning:
    - Drop rows with null values in key numeric columns
    - Clamp negative prices / camera MP to 0
    - Strip whitespace from string columns

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    original_len = len(df)

    # Strip string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Drop rows missing critical numeric data
    numeric_key_cols = ["price_usd", "ram", "battery_capacity", "primary_camera_mp"]
    df = df.dropna(subset=numeric_key_cols)

    # Clamp unrealistic values
    df = df[df["price_usd"] > 0]
    df = df[df["primary_camera_mp"] > 0]
    df = df[df["battery_capacity"] > 0]

    dropped = original_len - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} invalid/null rows during cleaning.")

    df = df.reset_index(drop=True)
    print(f"[data_loader] Clean dataset ready: {len(df):,} rows")
    return df
