"""
dp_training.py
Differentially Private Stochastic Gradient Descent (DP-SGD) training pipeline via Opacus.
Implements:
  - Per-sample gradient clipping (norm bound C = 1.0)
  - Calibrated Gaussian noise addition (scale sigma)
  - Rényi Differential Privacy (RDP) privacy accounting
"""

import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine

# Suppress minor non-fatal Opacus warnings during experimentation
warnings.filterwarnings("ignore", category=UserWarning, module="opacus")

try:
    from .config import (
        EPOCHS, LEARNING_RATE_DP, MAX_GRAD_NORM, TARGET_DELTA,
        TARGET_EPSILONS, WEIGHT_DECAY, RANDOM_SEED
    )
    from .baseline_model import StudentDropoutMLP, set_seed
except ImportError:
    from config import (
        EPOCHS, LEARNING_RATE_DP, MAX_GRAD_NORM, TARGET_DELTA,
        TARGET_EPSILONS, WEIGHT_DECAY, RANDOM_SEED
    )
    from baseline_model import StudentDropoutMLP, set_seed


def train_dp_model_with_target_epsilon(
    model,
    train_loader,
    target_epsilon,
    target_delta=TARGET_DELTA,
    epochs=EPOCHS,
    lr=LEARNING_RATE_DP,
    max_grad_norm=MAX_GRAD_NORM,
    device="cpu",
    verbose=True
):
    """
    Trains a Differentially Private model using Opacus DP-SGD calibrated to reach
    a specified target epsilon (eps) at target delta after the specified epochs.
    """
    set_seed(RANDOM_SEED)
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    privacy_engine = PrivacyEngine()

    dp_model, dp_optimizer, dp_train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        loss_reduction="mean"
    )

    noise_multiplier = dp_optimizer.noise_multiplier
    loss_history = []
    start_time = time.perf_counter()

    if verbose:
        print("\n" + "="*60)
        print(f"Training DP-SGD Model (Target eps={target_epsilon}, delta={target_delta})")
        print(f"Calibrated Noise Multiplier sigma: {noise_multiplier:.4f} | Clipping Norm C: {max_grad_norm}")
        print("="*60)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in dp_train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            dp_optimizer.zero_grad()
            outputs = dp_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            dp_optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        loss_history.append(epoch_loss)

        current_eps = privacy_engine.get_epsilon(delta=target_delta)
        if verbose and (epoch % 5 == 0 or epoch == epochs or epoch == 1):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | Accumulated (eps={current_eps:.2f}, delta={target_delta})")

    elapsed_time = time.perf_counter() - start_time
    final_epsilon = privacy_engine.get_epsilon(delta=target_delta)

    if verbose:
        print(f"DP Training Complete: Achieved eps={final_epsilon:.2f}, delta={target_delta} in {elapsed_time:.2f}s")

    return dp_model, final_epsilon, target_delta, loss_history, elapsed_time, noise_multiplier
