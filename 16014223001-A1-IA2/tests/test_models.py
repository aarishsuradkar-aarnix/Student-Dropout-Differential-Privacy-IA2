"""
test_models.py
Unit tests verifying neural network architecture forward pass, Opacus compatibility, and evaluation functions.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_model import StudentDropoutMLP
from evaluation import evaluate_model
from preprocessing import get_preprocessed_pipeline


def test_model_forward_pass():
    model = StudentDropoutMLP()
    dummy_input = torch.randn(16, 36)
    output = model(dummy_input)
    assert output.shape == (16, 3), f"Expected output shape (16, 3), got {output.shape}"


def test_opacus_compatibility():
    model = StudentDropoutMLP()
    # Ensure no BatchNorm layers exist
    for module in model.modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), \
            "BatchNorm detected! BatchNorm violates sample independence in Opacus DP-SGD."


def test_evaluation_metrics_structure():
    pipeline = get_preprocessed_pipeline()
    model = StudentDropoutMLP()
    metrics = evaluate_model(model, pipeline["test_loader"])
    required_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "loss", "confusion_matrix"]
    for k in required_keys:
        assert k in metrics, f"Missing metric key: {k}"


if __name__ == "__main__":
    test_model_forward_pass()
    test_opacus_compatibility()
    test_evaluation_metrics_structure()
    print("All model tests passed successfully!")
