"""
run_ia2_experiments.py
Master execution pipeline for IA-2:
  - Cryptographic provenance and SHA-256 verification
  - Baseline model training and evaluation
  - Differentially Private DP-SGD model sweep across eps in {1.0, 3.0, 5.0, 10.0}
  - Module 1: Membership Inference Attack (MIA) evaluation
  - Module 3: Local Differential Privacy (LDP) Randomized Response sweep
  - Module 4: Privacy-Preserving Data Mining (k-Anonymity & Homogeneity Attack)
  - Module 5: Compressive Privacy (PCA dimension sweep)
  - Full figure generation (Figures 01 through 10)
"""

import os
import sys
import json
from pathlib import Path

# Safe console encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATASET_PATH, RESULTS_DIR, FIGURES_DIR, DOCS_DIR,
    CLASS_NAMES, TARGET_EPSILONS, TARGET_DELTA,
    EPOCHS, LEARNING_RATE_BASELINE, LEARNING_RATE_DP,
    RANDOM_SEED
)
from provenance import generate_provenance_document
from data_loader import load_dataset, get_dataset_summary
from preprocessing import get_preprocessed_pipeline
from baseline_model import StudentDropoutMLP, train_baseline_model, set_seed
from dp_training import train_dp_model_with_target_epsilon
from evaluation import evaluate_model, compile_comparison_table
from mia_evaluation import compare_models_mia
from ldp_demo import run_ldp_sweep
from privacy_preserving_data_mining import evaluate_k_anonymity
from compression_demo import run_pca_compression_sweep
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_accuracy_comparison,
    plot_f1_comparison,
    plot_privacy_vs_accuracy,
    plot_training_losses,
    plot_mia_vulnerability,
    plot_ldp_tradeoff,
    plot_compression_tradeoff
)


def run_all_ia2_experiments():
    print("=" * 80)
    print("IA-2: STUDENT DROPOUT PREDICTION USING DIFFERENTIAL PRIVACY")
    print("Student: Aarish Suradkar (Roll: 16014223001)")
    print("=" * 80)

    # 1. Dataset Provenance & SHA-256 Verification
    print("\n[Step 1/7] Running Cryptographic Provenance Audit...")
    hashes = generate_provenance_document()
    print(f"Verified SHA-256: {hashes['sha256']}")
    print(f"Verified MD5:     {hashes['md5']}")

    # 2. Ingest Dataset & Plot Class Distribution
    print("\n[Step 2/7] Ingesting dataset and verifying integrity...")
    df = load_dataset(filepath=DATASET_PATH)
    summary = get_dataset_summary(df)
    print(f"Cohort size: {summary['total_records']} rows, {summary['total_features']} features")
    print(f"Missing values: {summary['missing_values']} | Duplicates: {summary['duplicates']}")
    plot_class_distribution(df, save_path=FIGURES_DIR / "01_class_distribution.png")

    # 3. Preprocessing & Data Leakage Prevention
    print("\n[Step 3/7] Performing stratified 80/20 train/test split and scaling...")
    pipeline = get_preprocessed_pipeline(df=df)
    train_loader = pipeline["train_loader"]
    test_loader = pipeline["test_loader"]
    X_train = pipeline["X_train"]
    X_test = pipeline["X_test"]
    y_train = pipeline["y_train"]
    y_test = pipeline["y_test"]
    print(f"Training partition: {len(y_train)} samples | Testing partition: {len(y_test)} samples")

    # 4. Train Baseline (Non-Private) Model
    print("\n[Step 4/7] Training Baseline (Non-Private) Model...")
    set_seed(RANDOM_SEED)
    baseline_net = StudentDropoutMLP()
    baseline_model, baseline_losses, baseline_time = train_baseline_model(
        baseline_net, train_loader, epochs=EPOCHS, lr=LEARNING_RATE_BASELINE, verbose=True
    )
    baseline_metrics = evaluate_model(baseline_model, test_loader)
    print(f"Baseline Accuracy: {baseline_metrics['accuracy']*100:.2f}% | Macro F1: {baseline_metrics['f1_macro']*100:.2f}%")

    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        title="Baseline (Non-Private) Confusion Matrix",
        save_path=FIGURES_DIR / "02_baseline_confusion_matrix.png"
    )

    experiments_log = {}
    experiments_log["Baseline (Non-Private)"] = {
        "metrics": baseline_metrics,
        "target_epsilon": "N/A (Non-private)",
        "achieved_epsilon": "Non-private (inf)",
        "achieved_delta": "0.0",
        "noise_multiplier": "0.0",
        "training_time": baseline_time
    }
    loss_curves = {"Baseline (Non-Private)": baseline_losses}
    trained_models_for_mia = {"Baseline (Non-Private)": baseline_model}

    # 5. Differentially Private Models (DP-SGD) Sweep
    print("\n[Step 5/7] Training Differentially Private Models across Epsilon Budgets...")
    dp_metrics_dict = {}

    for eps in TARGET_EPSILONS:
        print(f"\n--- Training DP-SGD Model for Target eps = {eps} ---")
        set_seed(RANDOM_SEED)
        dp_net = StudentDropoutMLP()
        trained_dp, ach_eps, ach_delta, dp_losses, dp_time, noise_m = train_dp_model_with_target_epsilon(
            model=dp_net,
            train_loader=train_loader,
            target_epsilon=eps,
            target_delta=TARGET_DELTA,
            epochs=EPOCHS,
            lr=LEARNING_RATE_DP,
            verbose=True
        )

        metrics = evaluate_model(trained_dp, test_loader)
        label = f"DP (eps={eps:.1f})"
        experiments_log[label] = {
            "metrics": metrics,
            "target_epsilon": eps,
            "achieved_epsilon": round(ach_eps, 2),
            "achieved_delta": ach_delta,
            "noise_multiplier": round(noise_m, 4),
            "training_time": dp_time
        }
        loss_curves[label] = dp_losses
        trained_models_for_mia[label] = trained_dp
        dp_metrics_dict[eps] = (metrics, ach_eps)

    # Figure 3: DP Confusion Matrix at representative eps = 3.0
    rep_metrics, rep_eps = dp_metrics_dict[3.0]
    plot_confusion_matrix(
        rep_metrics["confusion_matrix"],
        title=f"DP-SGD (eps={rep_eps:.2f}) Confusion Matrix",
        save_path=FIGURES_DIR / "03_dp_confusion_matrix_eps3.png"
    )

    # Compile Summary Comparison Table
    df_comparison = compile_comparison_table(experiments_log)
    print("\n" + "=" * 90)
    print("IA-2 EXPERIMENTAL COMPARISON TABLE")
    print("=" * 90)
    print(df_comparison.to_string(index=False))

    df_comparison.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    with open(RESULTS_DIR / "metrics_summary.md", "w", encoding="utf-8") as f:
        f.write("# IA-2 Model Performance and Privacy Trade-Off Summary\n\n")
        f.write(df_comparison.to_markdown(index=False))

    plot_accuracy_comparison(df_comparison, save_path=FIGURES_DIR / "04_accuracy_comparison.png")
    plot_f1_comparison(df_comparison, save_path=FIGURES_DIR / "05_f1_score_comparison.png")
    plot_privacy_vs_accuracy(df_comparison, save_path=FIGURES_DIR / "06_privacy_vs_accuracy.png")
    plot_training_losses(loss_curves, save_path=FIGURES_DIR / "07_training_loss_curves.png")

    # 6. Supplementary Modules Evaluation
    print("\n[Step 6/7] Executing Supplementary Syllabus Demonstrations...")

    # Module 1: Membership Inference Attack (MIA)
    print(" -> Running Membership Inference Attack Evaluation...")
    mia_df = compare_models_mia(trained_models_for_mia, train_loader, test_loader)
    mia_df.to_csv(RESULTS_DIR / "mia_results.csv", index=False)
    print("MIA Results:\n", mia_df.to_string(index=False))
    plot_mia_vulnerability(mia_df, save_path=FIGURES_DIR / "08_mia_vulnerability_comparison.png")

    # Module 3: Local Differential Privacy (Warner's Randomized Response)
    print(" -> Running Local Differential Privacy (LDP) Randomized Response Sweep...")
    ldp_df = run_ldp_sweep(df, attribute="Debtor")
    ldp_df.to_json(RESULTS_DIR / "ldp_results.json", indent=2, orient="records")
    print("LDP Sweep Results:\n", ldp_df.to_string(index=False))
    plot_ldp_tradeoff(ldp_df, save_path=FIGURES_DIR / "09_ldp_randomized_response.png")

    # Module 4: Privacy-Preserving Data Mining (k-Anonymity)
    print(" -> Running k-Anonymity and Homogeneity Attack Analysis...")
    k_anon_results = evaluate_k_anonymity(df)
    with open(RESULTS_DIR / "k_anonymity_results.json", "w", encoding="utf-8") as f:
        json.dump(k_anon_results, f, indent=2)
    print(f"k-Anonymity Metrics: {k_anon_results['k_anonymity_metrics']}")

    # Module 5: Compressive Privacy (PCA Dimensionality Reduction)
    print(" -> Running PCA Feature Compression Sweep...")
    comp_df = run_pca_compression_sweep(X_train, y_train, X_test, y_test)
    comp_df.to_csv(RESULTS_DIR / "compression_results.csv", index=False)
    print("PCA Compression Results:\n", comp_df.to_string(index=False))
    plot_compression_tradeoff(comp_df, save_path=FIGURES_DIR / "10_compression_vs_accuracy.png")

    print("\n" + "=" * 80)
    print("ALL IA-2 EXPERIMENTAL INVESTIGATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return df_comparison, mia_df, ldp_df, k_anon_results, comp_df


if __name__ == "__main__":
    run_all_ia2_experiments()
