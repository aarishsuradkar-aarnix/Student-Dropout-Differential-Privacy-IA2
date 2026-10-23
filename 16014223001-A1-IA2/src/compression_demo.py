"""
compression_demo.py
Module 5 Demonstration: Compressive Privacy and Dimensionality Reduction Trade-off.
Implements:
  - Principal Component Analysis (PCA) feature compression fitted strictly on training data
  - Systematic sweep across latent dimensions k in [2, 5, 10, 15, 20, 25, 30, 36]
  - Classification accuracy evaluation vs compression ratio
  - Critical analytical distinction between ordinary PCA and formal differentially private compression.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def run_pca_compression_sweep(X_train, y_train, X_test, y_test, components_list=[2, 5, 10, 15, 20, 25, 30, 36]):
    """
    Evaluates predictive utility as a function of PCA latent dimensions.
    PCA is fitted EXCLUSIVELY on X_train to prevent leakage.
    """
    results = []

    for k in components_list:
        pca = PCA(n_components=k, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        explained_var = float(np.sum(pca.explained_variance_ratio_))

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)

        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "components_k": int(k),
            "compression_ratio": round(k / X_train.shape[1], 4),
            "cumulative_variance_explained": round(explained_var * 100, 2),
            "test_accuracy": round(acc * 100, 2),
            "macro_f1": round(f1_mac * 100, 2)
        })

    return pd.DataFrame(results)
