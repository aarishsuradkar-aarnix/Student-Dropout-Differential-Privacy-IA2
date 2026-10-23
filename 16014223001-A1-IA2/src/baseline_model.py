"""
baseline_model.py
Standard (non-private) Multi-Layer Perceptron (MLP) architecture and training pipeline.
Serves as the empirical upper bound for predictive utility.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config import (
        INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT_RATE,
        EPOCHS, LEARNING_RATE_BASELINE, WEIGHT_DECAY, RANDOM_SEED
    )
except ImportError:
    from config import (
        INPUT_DIM, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT_RATE,
        EPOCHS, LEARNING_RATE_BASELINE, WEIGHT_DECAY, RANDOM_SEED
    )


class StudentDropoutMLP(nn.Module):
    """
    MLP architecture for Student Dropout Prediction.
    Architectural compatibility with Opacus:
      - Omission of Batch Normalization layers to avoid cross-sample gradient coupling.
      - Uses Linear + ReLU + Dropout layers natively compatible with per-sample gradient clipping.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim1=HIDDEN_DIM1,
                 hidden_dim2=HIDDEN_DIM2, output_dim=OUTPUT_DIM,
                 dropout_rate=DROPOUT_RATE):
        super(StudentDropoutMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed=RANDOM_SEED):
    """Sets random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_baseline_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE_BASELINE,
                         weight_decay=WEIGHT_DECAY, device="cpu", verbose=True):
    """
    Trains the non-private baseline neural network.
    """
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []
    start_time = time.perf_counter()

    if verbose:
        print("\n" + "="*50)
        print("Training Baseline (Non-Private) Model")
        print("="*50)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

        epoch_loss = total_loss / total_samples
        loss_history.append(epoch_loss)

        if verbose and (epoch % 5 == 0 or epoch == epochs or epoch == 1):
            print(f"Baseline Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

    elapsed_time = time.perf_counter() - start_time
    if verbose:
        print(f"Baseline Training Completed in {elapsed_time:.2f} seconds.")

    return model, loss_history, elapsed_time
