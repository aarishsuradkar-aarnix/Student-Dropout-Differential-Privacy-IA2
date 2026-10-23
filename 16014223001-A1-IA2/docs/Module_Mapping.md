# Syllabus Module Mapping & Technical Integration (IA-2)

**Course:** Security and Privacy for Machine Learning  
**Project Title:** Student Dropout Prediction Using Differential Privacy  
**Student:** Aarish Suradkar (Roll: 16014223001)  
**Assessment:** IA-2 — Implementation, Evaluation and Technical Outcomes  

---

## 1. Overview and Taxonomy of Methods

In accordance with Criterion 2 of the IA-2 evaluation rubric, this project systematically integrates and operationalizes all five modules of the *Security and Privacy for Machine Learning* syllabus. The primary practical implementation is concentrated in **Module 2 (Differential Privacy for Machine Learning)**, while Modules 1, 3, 4, and 5 are integrated through controlled experimental demonstrations and analytical evaluations.

All concepts, experiments, and techniques are strictly classified according to their operational implementation status:

1. **Implemented:** Full, runnable production source code in PyTorch, Opacus, and scikit-learn.
2. **Demonstrated Experimentally:** Validated with empirical execution, measurable numbers, cryptographic logs, or visual plots.
3. **Compared:** Statistically evaluated across baseline and multiple privacy regimes or dimensional configurations.
4. **Discussed Conceptually:** Rigorously analyzed from mathematical principles, threat modeling, and regulatory frameworks.

---

## 2. Comprehensive Module-by-Module Breakdown

### Module 1: Privacy Considerations in Machine Learning
- **Syllabus Topics Covered:** Privacy complications in the AI era, threats and attacks against ML systems (Membership Inference, Model Inversion, Attribute Inference), Privacy-Preserving Machine Learning (PPML) vs Privacy-Preserving Data Mining (PPDM).
- **Application to Student Retention:** Student academic records contain highly sensitive, stigmatizing information—such as tuition fee default status (`Debtor`), scholarship receipt, parental educational level, and failing grades. A deployed prediction API that is vulnerable to membership inference allows an adversary to verify whether a particular student was part of the "at-risk dropout" training cohort.
- **IA-2 Experimental Contribution:** We implemented an empirical loss-based Membership Inference Attack (MIA) evaluation (`src/mia_evaluation.py`). We measured Attack AUC-ROC and Attack Accuracy across the non-private baseline and all differentially private models, experimentally proving that as $\epsilon \to 1.0$, the attack AUC drops toward 0.50 (random guess).
- **Status:** **Implemented, Demonstrated Experimentally & Compared.**

---

### Module 2: Differential Privacy for Machine Learning (PRIMARY MODULE)
- **Syllabus Topics Covered:** $(\epsilon, \delta)$-Differential Privacy definitions, axiomatic properties (Composability, Post-Processing Invariance), Differentially Private Stochastic Gradient Descent (DP-SGD), per-sample gradient clipping, calibrated Gaussian noise, Rényi Differential Privacy (RDP) accounting, DP supervised learning.
- **Application to Student Retention:** Formally bounding the maximum influence that any individual student's record can exert on the trained model's parameters, guaranteeing that output predictions do not leak training record membership.
- **IA-2 Experimental Contribution:** Full end-to-end PyTorch + Opacus pipeline training a Multi-Layer Perceptron across four privacy budgets ($\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$ at fixed $\delta = 10^{-5}$) with $C = 1.0$ gradient clipping.
- **Status:** **Implemented, Demonstrated Experimentally & Compared (Core Project Contribution).**

---

### Module 3: Local Differential Privacy for Machine Learning
- **Syllabus Topics Covered:** Local Differential Privacy (LDP) vs Central DP, LDP mechanisms (Warner's Randomized Response 1965, RAPPOR), Piecewise Mechanism, LDP with Naive Bayes classification.
- **Application to Student Retention:** Decentralized data collection where the central university server is untrusted. Students perturb their own responses on personal devices before submitting survey records regarding financial distress or mental health challenges.
- **IA-2 Experimental Contribution:** We implemented a controlled simulation of Warner's Randomized Response mechanism on the sensitive `Debtor` attribute (`src/ldp_demo.py`). We tested across local privacy budgets $\epsilon_{LDP} \in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]$, comparing the perturbed reports against the unbiased population estimator and plotting the local privacy–utility trade-off curve.
- **Status:** **Implemented & Demonstrated Experimentally.**

---

### Module 4: Privacy-Preserving Data Mining Techniques
- **Syllabus Topics Covered:** Privacy protection in data processing, Privacy-Preserving Data Publishing (PPDP), $k$-Anonymity, $l$-Diversity, $t$-Closeness, input/output modification, vulnerabilities beyond $k$-anonymity.
- **Application to Student Retention:** Traditional institutional data publishing where tabular records are released to academic researchers or government registries.
- **IA-2 Experimental Contribution:** We implemented domain generalization (age bracket binning) and equivalence class evaluation for $k$-anonymity ($k=5, 10$) on student quasi-identifiers (`Age`, `Gender`, `Marital status`, `Course`) in `src/privacy_preserving_data_mining.py`. We demonstrated a concrete Homogeneity Attack vulnerability where an equivalence class satisfies $k \ge 5$ but exhibits 100% target homogeneity (all students dropped out), proving why syntactic $k$-anonymity is insufficient and necessitates Differential Privacy.
- **Status:** **Implemented & Demonstrated Experimentally.**

---

### Module 5: Compressive Privacy for Machine Learning
- **Syllabus Topics Covered:** Compressive privacy foundations, information-theoretic privacy, mechanisms of compressive privacy, utility-task accuracy trade-offs.
- **Application to Student Retention:** Compressing student representation vectors to retain features predictive of graduation while compressing away sensitive demographic variations.
- **IA-2 Experimental Contribution:** We executed a controlled zero-leakage feature-compression sweep using Principal Component Analysis (PCA) across $k \in [2, 5, 10, 15, 20, 25, 30, 36]$ dimensions (`src/compression_demo.py`). We measured predictive accuracy vs compression ratio and explained the critical theoretical distinction: **Standard PCA achieves dimensionality reduction but does NOT provide formal differential privacy**, as principal components can preserve and amplify sensitive attribute variance.
- **Status:** **Implemented & Demonstrated Experimentally.**

---

## 3. Final Master Module-Mapping Table

| Module Number & Name | Key Concepts Addressed | Relevance to Student Dropout Prediction | Operational Status | Empirical Evidence in IA-2 | Limitations & Justifications |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Module 1: Privacy Considerations in ML** | Threat modeling, Membership Inference Attacks (MIA), Model Inversion, Attribute Inference. | Deployed dropout prediction models leak whether vulnerable, at-risk students with tuition debt were in the training set. | **Implemented, Demonstrated & Compared** | `src/mia_evaluation.py`, `results/mia_results.csv`, Figure 8 (`08_mia_vulnerability_comparison.png`). | Evaluates black-box loss-threshold MIA; full shadow-model multi-network attack reserved for advanced extensions. |
| **Module 2: Differential Privacy for ML** | $(\epsilon, \delta)$-DP, DP-SGD, per-sample clipping ($C=1.0$), Gaussian mechanism, RDP accounting. | Primary mathematical defense bounding individual record disclosure risk during model parameter optimization. | **Implemented, Demonstrated & Compared [PRIMARY]** | `src/dp_training.py`, `results/metrics_summary.csv`, Figures 1–7 (`01_class_distribution.png` to `07_training_loss_curves.png`). | Fixed clipping threshold $C=1.0$; imbalanced minority class (`Enrolled`) experiences higher relative utility penalty. |
| **Module 3: Local Differential Privacy for ML** | Local vs Central DP, Warner's Randomized Response, perturbation parameter $p$, unbiased reconstruction. | Client-side student survey collection where the institutional collector is untrusted (e.g. self-reported debt or health status). | **Implemented & Demonstrated** | `src/ldp_demo.py`, `results/ldp_results.json`, Figure 9 (`09_ldp_randomized_response.png`). | Applied to binary indicator attribute (`Debtor`); generalized multi-dimensional piecewise LDP discussed analytically. |
| **Module 4: Privacy-Preserving Data Mining** | PPDP, Quasi-Identifiers, $k$-Anonymity, Equivalence Classes, Homogeneity Attacks. | Publishing tabular student records for institutional research without revealing student dropout status. | **Implemented & Demonstrated** | `src/privacy_preserving_data_mining.py`, `results/k_anonymity_results.json`. | Demonstrates syntactic compliance and failure against homogeneity attacks; proves superiority of DP. |
| **Module 5: Compressive Privacy for ML** | Information-theoretic compression, dimensionality reduction vs formal privacy, PCA variance trade-off. | Low-dimensional latent representation of student academic history for efficient storage and classification. | **Implemented & Demonstrated** | `src/compression_demo.py`, `results/compression_results.csv`, Figure 10 (`10_compression_vs_accuracy.png`). | Standard PCA evaluated; explicitly clarifies that PCA alone does not guarantee formal DP without randomized perturbations. |
