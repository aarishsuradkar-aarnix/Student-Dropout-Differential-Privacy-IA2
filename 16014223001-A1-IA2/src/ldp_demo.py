"""
ldp_demo.py
Module 3 Demonstration: Local Differential Privacy (LDP) via Warner's Randomized Response.
Applies randomized response to a sensitive binary student attribute ('Debtor' or 'Scholarship holder')
demonstrating client-side perturbation, unbiased reconstruction, and the local privacy-utility trade-off.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path


def simulate_randomized_response(data_series, epsilon_ldp, seed=42):
    """
    Simulates Warner's Randomized Response mechanism on a binary array (0 or 1).
    
    Perturbation probability:
      p = exp(eps) / (exp(eps) + 1)
      
    Truthful reporting with probability p; flipped report with probability (1 - p).
    Unbiased estimator:
      pi_hat = (q_reported - (1 - p)) / (2 * p - 1)
    """
    np.random.seed(seed)
    n = len(data_series)
    true_values = np.array(data_series, dtype=int)
    true_proportion = np.mean(true_values)

    p = np.exp(epsilon_ldp) / (np.exp(epsilon_ldp) + 1.0)
    flip_mask = np.random.rand(n) > p

    # Perturbed responses reported by students
    reported_values = np.where(flip_mask, 1 - true_values, true_values)
    reported_proportion = np.mean(reported_values)

    # Unbiased reconstruction of true population proportion
    if (2 * p - 1) != 0:
        estimated_proportion = (reported_proportion - (1.0 - p)) / (2.0 * p - 1.0)
    else:
        estimated_proportion = 0.5

    # Clip to valid probability bounds [0, 1]
    estimated_proportion_clipped = np.clip(estimated_proportion, 0.0, 1.0)
    absolute_error = abs(estimated_proportion - true_proportion)

    return {
        "epsilon_ldp": float(epsilon_ldp),
        "perturbation_p": round(float(p), 4),
        "true_proportion": round(float(true_proportion), 4),
        "reported_proportion": round(float(reported_proportion), 4),
        "estimated_proportion": round(float(estimated_proportion), 4),
        "estimated_proportion_clipped": round(float(estimated_proportion_clipped), 4),
        "absolute_error": round(float(absolute_error), 4),
        "total_respondents": int(n)
    }


def run_ldp_sweep(df, attribute="Debtor", epsilons=[0.2, 0.5, 1.0, 1.5, 2.0, 3.0]):
    """
    Executes an LDP budget sweep across multiple epsilon values.
    """
    if attribute not in df.columns:
        raise KeyError(f"Attribute '{attribute}' not found in DataFrame.")

    results = []
    for eps in epsilons:
        res = simulate_randomized_response(df[attribute], epsilon_ldp=eps)
        results.append(res)

    df_results = pd.DataFrame(results)
    return df_results
