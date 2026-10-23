"""
visualization.py
Visualization module for IA-2: Generates publication-ready figures for all experimental investigations.
Produces figures 01 through 10 covering baseline, DP-SGD, MIA, LDP, and Compressive Privacy.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from .config import CLASS_NAMES, FIGURES_DIR
except ImportError:
    from config import CLASS_NAMES, FIGURES_DIR

# Clean publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_class_distribution(df, save_path=None):
    target_counts = df["Target"].value_counts()
    labels = target_counts.index.tolist()
    counts = target_counts.values.tolist()
    total = sum(counts)
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.1, width=0.55)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.015,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    ax.set_title("UCI Dataset: Student Outcome Class Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Academic Outcome Category", labelpad=10)
    ax.set_ylabel("Number of Students", labelpad=10)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_sum * 100) if row_sum > 0 else 0
            text_color = "white" if val > thresh else "black"
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontweight="bold", fontsize=9
            )

    ax.set_ylim(cm.shape[0] - 0.5, -0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_accuracy_comparison(comparison_df, save_path=None):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    models = comparison_df["Model Name"].tolist()
    accuracies = comparison_df["Accuracy (%)"].tolist()

    colors = ["#2b5c8f"] + ["#e67e22" if "eps=1.0" in m else "#3498db" for m in models[1:]]
    bars = ax.bar(models, accuracies, color=colors, edgecolor="black", linewidth=1.1, width=0.55)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    ax.set_title("Test Accuracy: Baseline vs Differentially Private Models", pad=15, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", labelpad=10)
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_f1_comparison(comparison_df, save_path=None):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    models = comparison_df["Model Name"].tolist()
    macro_f1 = comparison_df["F1 Macro (%)"].tolist()
    weighted_f1 = comparison_df["F1 Weighted (%)"].tolist()

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, macro_f1, width, label="Macro F1-Score", color="#2980b9", edgecolor="black")
    rects2 = ax.bar(x + width/2, weighted_f1, width, label="Weighted F1-Score", color="#27ae60", edgecolor="black")

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("F1-Score Comparison: Macro vs Weighted", pad=15, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_privacy_vs_accuracy(comparison_df, save_path=None):
    dp_df = comparison_df[comparison_df["Target Epsilon"] != "N/A (Non-private)"].copy()
    if dp_df.empty:
        return None, None

    dp_eps = [float(x) for x in dp_df["Target Epsilon"].values]
    dp_acc = dp_df["Accuracy (%)"].astype(float).values

    base_df = comparison_df[comparison_df["Target Epsilon"] == "N/A (Non-private)"]
    baseline_acc = float(base_df["Accuracy (%)"].values[0]) if not base_df.empty else None

    fig, ax = plt.subplots(figsize=(7.8, 5))
    ax.plot(dp_eps, dp_acc, marker="o", color="#d35400", linewidth=2.5, markersize=8, label="DP-SGD Models")

    for x, y in zip(dp_eps, dp_acc):
        ax.annotate(f"{y:.2f}% (eps={x})", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", fontsize=9.5)

    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Baseline Non-Private ({baseline_acc:.2f}%)")

    ax.set_title("Privacy–Utility Trade-Off: Privacy Budget (eps) vs Accuracy", pad=15, fontweight="bold")
    ax.set_xlabel("Privacy Budget Epsilon (eps) [Strictness decreases ->]", labelpad=10)
    ax.set_ylabel("Test Accuracy (%)", labelpad=10)
    ax.set_ylim(min(dp_acc) - 8, (baseline_acc or max(dp_acc)) + 8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_training_losses(loss_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#2c3e50", "#e74c3c", "#f39c12", "#3498db", "#27ae60"]

    for i, (name, losses) in enumerate(loss_dict.items()):
        color = colors[i % len(colors)]
        epochs_arr = np.arange(1, len(losses) + 1)
        ax.plot(epochs_arr, losses, label=name, color=color, linewidth=2)

    ax.set_title("Training Loss Convergence Across Epochs", pad=15, fontweight="bold")
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel("Cross-Entropy Loss", labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_mia_vulnerability(mia_df, save_path=None):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    models = mia_df["Model"].tolist()
    auc_scores = mia_df["Attack AUC-ROC"].tolist()

    colors = ["#c0392b" if "Baseline" in m else "#2980b9" for m in models]
    bars = ax.bar(models, auc_scores, color=colors, edgecolor="black", width=0.5)

    for bar, score in zip(bars, auc_scores):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.01, f"{score:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.5, label="Random Guessing (AUC = 0.50)")
    ax.set_title("Membership Inference Attack (MIA) Vulnerability Across Models", pad=15, fontweight="bold")
    ax.set_ylabel("Attack AUC-ROC Score", labelpad=10)
    ax.set_ylim(0.40, max(auc_scores) + 0.10)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax


def plot_ldp_tradeoff(ldp_df, save_path=None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    eps_vals = ldp_df["epsilon_ldp"].tolist()
    errors = ldp_df["absolute_error"].tolist()
    p_vals = ldp_df["perturbation_p"].tolist()

    color1 = "#d35400"
    ax1.plot(eps_vals, errors, marker="s", color=color1, linewidth=2.5, label="Absolute Estimation Error")
    ax1.set_xlabel("Local Privacy Budget Epsilon (eps_LDP)", labelpad=10)
    ax1.set_ylabel("Absolute Estimation Error |pi_hat - pi_true|", color=color1, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#2980b9"
    ax2.plot(eps_vals, p_vals, marker="o", color=color2, linewidth=2, linestyle="--", label="Truthful Report Probability (p)")
    ax2.set_ylabel("Truthful Reporting Probability (p)", color=color2, labelpad=10)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Local Differential Privacy (Warner's RR): Privacy vs Utility", pad=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax1


def plot_compression_tradeoff(comp_df, save_path=None):
    fig, ax1 = plt.subplots(figsize=(8.2, 5))
    k_vals = comp_df["components_k"].tolist()
    accs = comp_df["test_accuracy"].tolist()
    var_exp = comp_df["cumulative_variance_explained"].tolist()

    color1 = "#27ae60"
    ax1.plot(k_vals, accs, marker="o", color=color1, linewidth=2.5, label="Test Accuracy (%)")
    ax1.set_xlabel("Number of PCA Components (k)", labelpad=10)
    ax1.set_ylabel("Test Accuracy (%)", color=color1, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#8e44ad"
    ax2.plot(k_vals, var_exp, marker="^", color=color2, linewidth=2, linestyle="--", label="Cumulative Explained Variance (%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color=color2, labelpad=10)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Compressive Feature Representation: Dimension vs Accuracy", pad=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig, ax1
