"""
config.py
Configuration and global constants for IA-2: Student Dropout Prediction with Differential Privacy.
Student: Aarish Suradkar (Roll: 16014223001)
Course: Security and Privacy for Machine Learning
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Dataset Configuration
DATASET_FILENAME = "data.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
UCI_DOI = "10.24432/C5MC89"
EXPECTED_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"
EXPECTED_MD5 = "ee29c9ee9ea60654b5bb0a161c0a473d"

TARGET_COLUMN = "Target"
CLASS_MAPPING = {
    "Dropout": 0,
    "Enrolled": 1,
    "Graduate": 2
}
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
NUM_CLASSES = 3

# Reproducibility Seed
RANDOM_SEED = 42

# Train / Test Splitting
TEST_SIZE = 0.20
STRATIFY = True

# Neural Network Architecture
INPUT_DIM = 36          # 36 input predictor features
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 32
OUTPUT_DIM = 3          # 3 target classes
DROPOUT_RATE = 0.15     # Regularization dropout inside MLP

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE_BASELINE = 0.005
LEARNING_RATE_DP = 0.015     # DP-SGD requires slightly higher learning rate due to noise injection
WEIGHT_DECAY = 1e-4

# Differential Privacy Settings (Opacus DP-SGD)
MAX_GRAD_NORM = 1.0          # L2 gradient clipping bound C
TARGET_DELTA = 1e-5          # δ condition: δ < 1/N (N_train = 3539, so 1e-5 << 2.8e-4)
TARGET_EPSILONS = [1.0, 3.0, 5.0, 10.0]  # Academic privacy budget sweep
