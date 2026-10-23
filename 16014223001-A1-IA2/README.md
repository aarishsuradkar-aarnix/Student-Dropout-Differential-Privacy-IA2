# Student Dropout Prediction Using Differential Privacy (IA-2)

**Course:** Security and Privacy for Machine Learning  
**Assessment:** Continuous Internal Assessment — IA-2 (20 Marks)  
**Student Name:** Aarish Suradkar  
**Roll Number:** 16014223001  
**Degree Programme:** Third Year B.Tech — Artificial Intelligence and Data Science  
**GitHub Repository:** [https://github.com/aarishsuradkar-aarnix/Student-Dropout-Differential-Privacy](https://github.com/aarishsuradkar-aarnix/Student-Dropout-Differential-Privacy)  

---

## 1. Project Overview

Higher education administrative databases contain sensitive student indicators, including tuition fee debt status, scholarship reliance, parental occupations, and semester failure counts. Machine learning models trained on these records without privacy guarantees are vulnerable to **Membership Inference Attacks (MIA)**, where an adversary confirms whether an at-risk student was in the training cohort.

This **IA-2 upgrade** delivers a complete, verified, and defensible investigation of the **Privacy–Utility–Security Trade-Off**:
1. **Primary Contribution (Module 2):** PyTorch Multi-Layer Perceptron trained using Differentially Private SGD (DP-SGD) with Opacus across privacy budgets $\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$ at fixed $\delta = 10^{-5}$ ($C = 1.0$).
2. **Empirical Security Auditing (Module 1):** Implementation and evaluation of a black-box Membership Inference Attack (MIA), demonstrating that DP-SGD reduces attack AUC from 0.5823 (baseline) down to 0.5155 (approaching random guessing at $\epsilon = 1.0$).
3. **Multi-Module Demonstrations:** Working implementations of Local DP (Warner's Randomized Response on `Debtor`), Tabular Data Mining ($k$-anonymity with Homogeneity Attack evidence), and Compressive Privacy (PCA latent dimension sweep).
4. **Cryptographic Checksum:** Verified SHA-256 (`3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`) on the 4,424-record benchmark dataset.

---

## 2. Directory Structure

```
student_dropout_dp_ia2/
├── notebooks/
│   └── IA2_Student_Dropout_Differential_Privacy.ipynb  # Complete Google Colab notebook (CPU-friendly)
├── src/
│   ├── config.py             # Hyperparameters, random seed (42), file paths, constants
│   ├── data_loader.py        # UCI dataset ingestion, validation audits, column sanitization
│   ├── provenance.py         # Cryptographic SHA-256 & MD5 verification and documentation
│   ├── preprocessing.py      # Stratified 80:20 split, zero-leakage StandardScaler, DataLoader
│   ├── baseline_model.py     # Non-private PyTorch MLP architecture and training loop
│   ├── dp_training.py        # Opacus DP-SGD implementation, gradient clipping, RDP accounting
│   ├── evaluation.py         # Multi-class accuracy, precision, recall, macro/weighted F1, per-class F1
│   ├── mia_evaluation.py     # Module 1: Loss-based Membership Inference Attack empirical audit
│   ├── ldp_demo.py           # Module 3: Local Differential Privacy (Warner's Randomized Response)
│   ├── privacy_preserving_data_mining.py  # Module 4: k-Anonymity & Homogeneity Attack analysis
│   ├── compression_demo.py   # Module 5: Compressive Privacy (PCA dimension sweep)
│   ├── visualization.py      # Publication-quality plotting routines (10 figures)
│   └── run_ia2_experiments.py# Master execution pipeline
├── data/
│   ├── dataset_source.txt    # UCI repository attribution and citation details
│   └── data.csv              # Verified benchmark dataset (4,424 records, 36 features, 3 classes)
├── results/
│   ├── metrics_summary.csv   # Consolidated quantitative evaluation table across all models
│   ├── metrics_summary.md    # Markdown formatted summary comparison table
│   ├── mia_results.csv       # Empirical Membership Inference Attack evaluation results
│   ├── ldp_results.json      # Local DP sweep outputs (epsilon_ldp vs error)
│   ├── k_anonymity_results.json  # k-Anonymity compliance and homogeneity attack evidence
│   └── compression_results.csv   # PCA components vs accuracy and explained variance
├── figures/
│   ├── 01_class_distribution.png
│   ├── 02_baseline_confusion_matrix.png
│   ├── 03_dp_confusion_matrix_eps3.png
│   ├── 04_accuracy_comparison.png
│   ├── 05_f1_score_comparison.png
│   ├── 06_privacy_vs_accuracy.png
│   ├── 07_training_loss_curves.png
│   ├── 08_mia_vulnerability_comparison.png
│   ├── 09_ldp_randomized_response.png
│   └── 10_compression_vs_accuracy.png
├── docs/
│   ├── IA2_Final_Report.md   # Full 29-section academic report in Markdown
│   ├── IA2_Final_Report.docx # Formatted Microsoft Word report
│   ├── Dataset_Provenance.md # Cryptographic SHA-256 audit and provenance
│   ├── Module_Mapping.md     # Mapping across all 5 syllabus modules
│   ├── Workflow_Diagram.mmd  # Mermaid workflow diagram source
│   └── IA2_Submission_Form_Answers.md  # 24 ready-to-copy submission form answers
├── tests/
│   ├── test_preprocessing.py # Unit tests for data loading, split, and zero leakage
│   └── test_models.py        # Unit tests for architecture, forward pass, and Opacus compatibility
├── requirements.txt          # Python dependencies with compatible versions
└── README.md                 # Primary project overview and instructions
```

---

## 3. Verified Experimental Results Summary

### Model Performance & Privacy Accounting (Module 2)

| Model Description | Target Budget | Achieved $\epsilon$ | Failure Prob. $\delta$ | Noise Scale $\sigma$ | Test Accuracy (%) | Macro F1 (%) | Dropout F1 (%) | Enrolled F1 (%) | Graduate F1 (%) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Non-Private)** | N/A | Non-private ($\infty$) | $0.0$ | $0.0000$ | **73.22** | **67.01** | 76.95 | **42.01** | 82.07 | 0.7079 |
| **DP-SGD ($\epsilon = 10.0$)** | 10.0 | **9.99** | $10^{-5}$ | 0.6879 | **76.38** | **66.27** | 79.02 | **34.75** | 85.03 | 0.8248 |
| **DP-SGD ($\epsilon = 5.0$)** | 5.0 | **4.99** | $10^{-5}$ | 0.9094 | **76.50** | **65.83** | 79.23 | **33.04** | 85.21 | 0.8362 |
| **DP-SGD ($\epsilon = 3.0$)** | 3.0 | **2.99** | $10^{-5}$ | 1.1890 | **76.27** | **64.96** | 79.44 | **30.63** | 84.80 | 0.8510 |
| **DP-SGD ($\epsilon = 1.0$)** | 1.0 | **1.00** | $10^{-5}$ | 2.6562 | **73.45** | **59.63** | 76.63 | **18.78** | 83.49 | 1.0386 |

### Membership Inference Attack (MIA) Defense (Module 1)

| Model | Attack AUC-ROC | Attack Accuracy (%) | Generalization Loss Gap | Defense Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Non-Private)** | **0.5823** | **55.03%** | **0.3598** | **Vulnerable** (Training loss 0.348 vs test loss 0.708) |
| **DP-SGD ($\epsilon = 10.0$)** | 0.5253 | 52.54% | 0.0775 | Strong Defense (Loss gap reduced by 78%) |
| **DP-SGD ($\epsilon = 3.0$)** | 0.5337 | 53.33% | 0.0792 | Strong Defense |
| **DP-SGD ($\epsilon = 1.0$)** | **0.5155** | **52.66%** | **0.0813** | **Near-Optimal Defense** (Approaches 0.50 random guess) |

---

## 4. How to Run in Google Colab (Zero-Setup)

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File** $\to$ **Upload notebook** and upload:
   `notebooks/IA2_Student_Dropout_Differential_Privacy.ipynb`
3. Ensure runtime is set to CPU: **Runtime** $\to$ **Change runtime type** $\to$ **CPU**.
4. Select **Runtime** $\to$ **Run all**.
5. Execution takes ~2.5 minutes on CPU. All 10 plots and comparison tables are rendered inline.

---

## 5. How to Run Locally

```powershell
# 1. Clone repository
git clone https://github.com/aarishsuradkar-aarnix/Student-Dropout-Differential-Privacy.git
cd student_dropout_dp_ia2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests
python tests/test_preprocessing.py
python tests/test_models.py

# 4. Execute the complete experimental suite
python src/run_ia2_experiments.py
```

---

## 6. Academic Assessment Mapping

- **Criterion 1 (Problem Refinement):** Addressed in `docs/IA2_Final_Report.md` (Sections 4, 5, 6, 7).
- **Criterion 2 (Syllabus Integration):** Documented in `docs/Module_Mapping.md` and demonstrated via `src/mia_evaluation.py`, `src/dp_training.py`, `src/ldp_demo.py`, `src/privacy_preserving_data_mining.py`, and `src/compression_demo.py`.
- **Criterion 3 (Dataset & Provenance):** Verified in `docs/Dataset_Provenance.md` with SHA-256 checksum and CC BY 4.0 license.
- **Criterion 4 (Evaluation & Critical Analysis):** Documented in `docs/IA2_Final_Report.md` (Sections 18–23) with per-class F1, MIA defense, and minority class impact.
- **Criterion 5 (Final Documentation):** Provided via `docs/IA2_Final_Report.docx`, `docs/IA2_Final_Report.md`, 10 publication-ready figures, and `docs/IA2_Submission_Form_Answers.md`.
