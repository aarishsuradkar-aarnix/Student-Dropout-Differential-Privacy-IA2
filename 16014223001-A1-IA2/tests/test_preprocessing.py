"""
test_preprocessing.py
Unit tests verifying data loading, sanitization, stratified splitting, and zero-leakage scaling.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure src can be imported
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_dataset, get_dataset_summary
from preprocessing import prepare_features_and_target, split_and_scale_data, get_preprocessed_pipeline


def test_dataset_integrity():
    df = load_dataset()
    summary = get_dataset_summary(df)
    assert summary["total_records"] == 4424, f"Expected 4424 records, got {summary['total_records']}"
    assert summary["total_features"] == 36, f"Expected 36 features, got {summary['total_features']}"
    assert summary["missing_values"] == 0, f"Expected 0 missing values, got {summary['missing_values']}"
    assert summary["duplicates"] == 0, f"Expected 0 duplicates, got {summary['duplicates']}"


def test_stratified_split_and_dimensions():
    pipeline = get_preprocessed_pipeline()
    assert pipeline["X_train"].shape == (3539, 36)
    assert pipeline["X_test"].shape == (885, 36)
    assert len(pipeline["y_train"]) == 3539
    assert len(pipeline["y_test"]) == 885
    assert pipeline["input_dim"] == 36
    assert pipeline["num_classes"] == 3


def test_zero_data_leakage():
    pipeline = get_preprocessed_pipeline()
    scaler = pipeline["scaler"]
    # Check that scaler was fitted on 3539 training samples
    assert scaler.n_samples_seen_ == 3539, f"Scaler saw {scaler.n_samples_seen_} samples instead of 3539"
    # Ensure train data has mean approx 0 and std approx 1
    assert np.allclose(np.mean(pipeline["X_train"], axis=0), 0.0, atol=1e-2)
    assert np.allclose(np.std(pipeline["X_train"], axis=0), 1.0, atol=1e-2)


if __name__ == "__main__":
    test_dataset_integrity()
    test_stratified_split_and_dimensions()
    test_zero_data_leakage()
    print("All preprocessing tests passed successfully!")
