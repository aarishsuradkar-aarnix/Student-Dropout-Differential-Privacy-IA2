"""
privacy_preserving_data_mining.py
Module 4 Demonstration: Privacy-Preserving Data Mining via k-Anonymity and Homogeneity Attack Analysis.
Implements:
  - Identification of Quasi-Identifiers (QIs)
  - Generalization (binning age into ordinal age brackets)
  - Calculation of Equivalence Classes (ECs) for k-anonymity (k=5, k=10)
  - Suppression of non-compliant records
  - Demonstration of a Homogeneity Attack showing why k-anonymity collapses, necessitating Differential Privacy.
"""

import pandas as pd
import numpy as np


def apply_generalization(df):
    """
    Applies domain generalization to quasi-identifiers:
      - Bins 'Age at enrollment' into age brackets ['<20', '20-24', '25-29', '>=30']
    """
    df_gen = df.copy()
    bins = [0, 19, 24, 29, 100]
    labels = ["<20", "20-24", "25-29", ">=30"]
    df_gen["Age_Bracket"] = pd.cut(df_gen["Age at enrollment"], bins=bins, labels=labels, right=True)
    return df_gen


def evaluate_k_anonymity(df, quasi_identifiers=["Age_Bracket", "Gender", "Marital status", "Course"], k_thresholds=[5, 10]):
    """
    Computes equivalence classes for a set of quasi-identifiers and evaluates k-anonymity compliance.
    """
    df_gen = apply_generalization(df)
    grouped = df_gen.groupby(quasi_identifiers).size().reset_index(name="Equivalence_Class_Size")

    results = {}
    for k in k_thresholds:
        violating_ecs = grouped[grouped["Equivalence_Class_Size"] < k]
        violating_records = violating_ecs["Equivalence_Class_Size"].sum()
        total_records = len(df_gen)
        compliance_pct = round((total_records - violating_records) / total_records * 100, 2)

        results[f"k={k}"] = {
            "total_equivalence_classes": len(grouped),
            "violating_classes_count": len(violating_ecs),
            "violating_records_count": int(violating_records),
            "compliance_percentage": compliance_pct
        }

    # Identify a representative Homogeneity Attack vulnerability
    # An equivalence class where k >= 5 but 100% of students share the same Target outcome
    full_grouped = df_gen.groupby(quasi_identifiers + ["Target"]).size().unstack(fill_value=0)
    full_grouped["Total"] = full_grouped.sum(axis=1)

    homogeneity_examples = []
    for idx, row in full_grouped.iterrows():
        if row["Total"] >= 5:
            for cls in ["Dropout", "Graduate", "Enrolled"]:
                if cls in row and row[cls] == row["Total"]:
                    homogeneity_examples.append({
                        "quasi_identifier_values": dict(zip(quasi_identifiers, idx)),
                        "class_size_k": int(row["Total"]),
                        "leaked_sensitive_target": cls,
                        "leakage_probability": 1.0
                    })
                    break

    return {
        "k_anonymity_metrics": results,
        "homogeneity_attack_examples": homogeneity_examples[:3]  # top 3 illustrative examples
    }
