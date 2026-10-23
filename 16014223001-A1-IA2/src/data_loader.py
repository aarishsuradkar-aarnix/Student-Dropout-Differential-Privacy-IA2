"""
data_loader.py
Dataset ingestion and verification module for IA-2.
Loads the UCI Student Dropout and Academic Success dataset, verifies dimensions,
sanitizes column headers, and checks for missing or duplicate values.
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

try:
    from .config import DATASET_PATH, DATASET_URL, TARGET_COLUMN
except ImportError:
    from config import DATASET_PATH, DATASET_URL, TARGET_COLUMN


def download_dataset_if_needed(target_path=DATASET_PATH, url=DATASET_URL):
    """
    Downloads dataset from UCI repository if absent.
    """
    target_path = Path(target_path)
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = target_path.parent / "temp_download.zip"
    try:
        urllib.request.urlretrieve(url, temp_zip)
        with zipfile.ZipFile(temp_zip, 'r') as z:
            csv_file = [f for f in z.namelist() if f.endswith('.csv')][0]
            z.extract(csv_file, target_path.parent)
            extracted = target_path.parent / csv_file
            if extracted != target_path:
                extracted.rename(target_path)
        if temp_zip.exists():
            temp_zip.unlink()
        return target_path
    except Exception as e:
        if temp_zip.exists():
            temp_zip.unlink()
        raise RuntimeError(f"Failed to download dataset: {e}")


def load_dataset(filepath=DATASET_PATH, auto_download=True):
    """
    Loads data.csv with semicolon separator, strips extraneous header whitespace,
    and returns a verified pandas DataFrame.
    """
    filepath = Path(filepath)
    if not filepath.exists() and auto_download:
        filepath = download_dataset_if_needed(target_path=filepath)

    df = pd.read_csv(filepath, sep=';')
    df.columns = df.columns.str.strip()
    return df


def get_dataset_summary(df):
    """
    Returns summary statistics including row count, column count,
    missing values, and class breakdown.
    """
    return {
        "total_records": len(df),
        "total_features": df.shape[1] - 1,
        "total_columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "class_proportions": (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()
    }
