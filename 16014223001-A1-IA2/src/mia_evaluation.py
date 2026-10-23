"""
mia_evaluation.py
Module 1 Demonstration: Membership Inference Attack (MIA) Empirical Evaluation.
Evaluates the vulnerability of trained models (Baseline vs DP-SGD at eps in {10, 5, 3, 1})
against black-box loss-based Membership Inference Attacks (Yeom et al., 2018; Salem et al., 2019).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def extract_sample_losses(model, data_loader, device="cpu"):
    """
    Computes per-sample cross-entropy losses for all samples in a DataLoader.
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss_batch = criterion(outputs, y_batch)
            losses.extend(loss_batch.cpu().numpy().tolist())

    return np.array(losses)


def run_membership_inference_attack(model, train_loader, test_loader, device="cpu"):
    """
    Simulates a threshold-based Membership Inference Attack (MIA).
    Training samples have ground-truth membership = 1.
    Test samples have ground-truth membership = 0.
    
    The adversary uses sample loss as the distinguishing score:
    Lower loss indicates higher likelihood of membership.
    Score = -loss (higher score => more likely to be a training member).
    """
    train_losses = extract_sample_losses(model, train_loader, device=device)
    test_losses = extract_sample_losses(model, test_loader, device=device)

    # Subsample training losses to match test set size for a 50/50 balanced attack evaluation
    np.random.seed(42)
    n_test = len(test_losses)
    train_idx = np.random.choice(len(train_losses), size=n_test, replace=False)
    sampled_train_losses = train_losses[train_idx]

    # Combine losses and true membership labels
    losses = np.concatenate([sampled_train_losses, test_losses])
    y_member = np.concatenate([np.ones(n_test, dtype=int), np.zeros(n_test, dtype=int)])

    # Attack score: negative loss (lower loss => higher member probability)
    scores = -losses

    # Compute Attack ROC-AUC
    attack_auc = roc_auc_score(y_member, scores)

    # Threshold-based decision: threshold at the median score
    threshold = np.median(scores)
    y_pred_member = (scores >= threshold).astype(int)

    attack_acc = accuracy_score(y_member, y_pred_member)
    attack_prec = precision_score(y_member, y_pred_member, zero_division=0)
    attack_rec = recall_score(y_member, y_pred_member, zero_division=0)

    mean_train_loss = float(np.mean(sampled_train_losses))
    mean_test_loss = float(np.mean(test_losses))
    loss_gap = float(mean_test_loss - mean_train_loss)

    return {
        "attack_auc": round(float(attack_auc), 4),
        "attack_accuracy": round(float(attack_acc * 100), 2),
        "attack_precision": round(float(attack_prec * 100), 2),
        "attack_recall": round(float(attack_rec * 100), 2),
        "mean_train_loss": round(mean_train_loss, 4),
        "mean_test_loss": round(mean_test_loss, 4),
        "generalization_loss_gap": round(loss_gap, 4)
    }


def compare_models_mia(models_dict, train_loader, test_loader, device="cpu"):
    """
    Evaluates and compares MIA vulnerability across all models:
    Non-private Baseline vs DP-SGD models.
    """
    results = []
    for model_name, model_obj in models_dict.items():
        metrics = run_membership_inference_attack(model_obj, train_loader, test_loader, device=device)
        record = {
            "Model": model_name,
            "Attack AUC-ROC": metrics["attack_auc"],
            "Attack Accuracy (%)": metrics["attack_accuracy"],
            "Train Loss": metrics["mean_train_loss"],
            "Test Loss": metrics["mean_test_loss"],
            "Loss Gap (Generalization Gap)": metrics["generalization_loss_gap"]
        }
        results.append(record)

    return pd.DataFrame(results)
