"""
preprocessing.py
Prepares and transforms the Student Dropout dataset for baseline and differentially private ML.
Implements:
  - Feature-target separation
  - Label encoding for multi-class target (Dropout: 0, Enrolled: 1, Graduate: 2)
  - Stratified 80/20 train-test split to preserve class proportions
  - Rigorous data leakage prevention (StandardScaler fitted strictly on training data)
  - Conversion to PyTorch TensorDatasets and DataLoaders
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .config import (
        TARGET_COLUMN, CLASS_MAPPING, TEST_SIZE, RANDOM_SEED, BATCH_SIZE
    )
except ImportError:
    from config import (
        TARGET_COLUMN, CLASS_MAPPING, TEST_SIZE, RANDOM_SEED, BATCH_SIZE
    )


def prepare_features_and_target(df, target_col=TARGET_COLUMN, class_mapping=CLASS_MAPPING):
    """
    Separates predictor features from the target column and maps categorical class labels to integers.
    """
    df_clean = df.copy()
    if target_col not in df_clean.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe columns: {df_clean.columns.tolist()}")

    X = df_clean.drop(columns=[target_col])
    feature_names = list(X.columns)

    # Encode target labels
    y_raw = df_clean[target_col]
    y = y_raw.map(class_mapping).values

    if np.isnan(y).any():
        unmapped = y_raw[np.isnan(y)].unique()
        raise ValueError(f"Unmapped target labels found: {unmapped}")

    return X, y, feature_names


def split_and_scale_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Performs a stratified train-test split and standardizes continuous/numerical features.
    StandardScaler is fitted EXCLUSIVELY on training data to prevent statistical leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_pytorch_loaders(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE):
    """
    Wraps preprocessed numpy arrays into PyTorch TensorDatasets and DataLoaders.
    """
    tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)

    tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
    tensor_y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    test_dataset = TensorDataset(tensor_X_test, tensor_y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader, train_dataset, test_dataset


def get_preprocessed_pipeline(df=None, batch_size=BATCH_SIZE, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    End-to-end preprocessing pipeline returning PyTorch DataLoaders and metadata.
    """
    if df is None:
        try:
            from .data_loader import load_dataset
        except ImportError:
            from data_loader import load_dataset
        df = load_dataset()

    X, y, feature_names = prepare_features_and_target(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(
        X, y, test_size=test_size, random_state=random_state
    )

    train_loader, test_loader, train_ds, test_ds = create_pytorch_loaders(
        X_train_scaled, y_train, X_test_scaled, y_test, batch_size=batch_size
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "class_mapping": CLASS_MAPPING,
        "input_dim": X_train_scaled.shape[1],
        "num_classes": len(CLASS_MAPPING)
    }
