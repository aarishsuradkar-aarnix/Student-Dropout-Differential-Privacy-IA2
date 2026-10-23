# IA-2 Assessment Form — Final Ready-to-Copy Submission Content

**Student Name:** Aarish Suradkar  
**Roll Number:** 16014223001  
**Course:** Security and Privacy for Machine Learning  
**Assessment:** IA-2 — Implementation, Evaluation and Technical Outcomes  
**Project Title:** Student Dropout Prediction Using Differential Privacy  

---

### 1. Final Topics from Modules
- **Module 1:** Threat modeling for student educational records, Membership Inference Attacks (MIA), Model Inversion, and Privacy-Preserving Machine Learning (PPML).
- **Module 2 (Primary):** Differentially Private Stochastic Gradient Descent (DP-SGD), per-sample gradient clipping ($C = 1.0$), calibrated Gaussian noise mechanism, Rényi Differential Privacy (RDP) accounting, and privacy budget sweeps ($\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$ at $\delta = 10^{-5}$).
- **Module 3:** Local Differential Privacy (LDP), Warner's Randomized Response mechanism on binary indicators (`Debtor`), and local privacy–utility trade-offs.
- **Module 4:** Privacy-Preserving Data Mining (PPDM), quasi-identifier generalization (age binning), $k$-anonymity equivalence class evaluation, and Homogeneity Attack demonstration.
- **Module 5:** Compressive Privacy, lossy feature compression via Principal Component Analysis (PCA), and analytical distinction between ordinary PCA and formal DP guarantees.

---

### 2. Refined Problem Statement
To build, evaluate, and critically analyze a machine learning classification system capable of predicting student academic status (`Dropout`, `Enrolled`, `Graduate`) while providing mathematically provable $(\epsilon, \delta)$-Differential Privacy guarantees for individual student records, to empirically measure resistance against black-box Membership Inference Attacks, and to systematically characterize the multi-dimensional trade-offs between privacy budget strictness ($\epsilon$), predictive utility, minority-class degradation, and computational overhead.

---

### 3. Final Objectives and Expected / Achieved Outcomes
1. **Objective 1:** Ingest and verify the 4,424-record UCI Student Dropout benchmark dataset with cryptographic SHA-256 integrity verification. *(Achieved: Confirmed SHA-256 `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`).*
2. **Objective 2:** Construct a zero-leakage preprocessing pipeline preserving all 4,424 records across the original 3 classes using an 80:20 stratified split. *(Achieved: 3,539 train / 885 test records).*
3. **Objective 3:** Implement baseline and DP-SGD classifiers in PyTorch and Opacus across $\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$ at $\delta = 10^{-5}$. *(Achieved: Baseline accuracy 73.22%, DP accuracy 76.27% at $\epsilon=3.0$ and 73.45% at $\epsilon=1.0$).*
4. **Objective 4:** Empirically quantify vulnerability to Membership Inference Attacks (MIA). *(Achieved: Proved attack AUC drops from baseline down toward 0.50 random guess under DP).*
5. **Objective 5:** Provide working implementations for Local DP, $k$-Anonymity, and Compressive PCA. *(Achieved: Fully demonstrated across 10 publication-quality figures).*

---

### 4. Research Gap or Technical Need
Institutional machine learning models trained on sensitive academic and financial records are susceptible to membership inference and reconstruction attacks that compromise student confidentiality. While Differential Privacy provides rigorous mathematical protection, applying DP-SGD to imbalanced tabular educational records presents severe utility challenges—specifically, per-sample clipping and noise injection can disproportionately degrade predictive accuracy on vulnerable minority cohorts (such as the `Enrolled` class). There is a critical academic need to empirically quantify this trade-off, audit MIA defense efficacy, and bridge syntactic privacy methods with formal differential privacy.

---

### 5. Select the Modules Addressed in Your Activity
- Module 1: Privacy Considerations in Machine Learning
- Module 2: Differential Privacy for Machine Learning (Primary)
- Module 3: Local Differential Privacy for Machine Learning
- Module 4: Privacy-Preserving Data Mining Techniques
- Module 5: Compressive Privacy for Machine Learning

---

### 6. Explain How Your Activity Relates to All Five Modules
The activity adopts **Module 2 (DP-SGD)** as the primary practical engine, training neural networks with per-sample gradient clipping and calibrated Gaussian noise under Opacus RDP accounting. **Module 1** establishes the threat model and evaluates black-box Membership Inference Attack (MIA) resistance. **Module 3** explores client-side decentralized data collection via Warner's Randomized Response on the sensitive `Debtor` attribute. **Module 4** evaluates tabular data publishing via $k$-anonymity and demonstrates why syntactic methods fail against Homogeneity Attacks. **Module 5** examines feature compression via PCA, evaluating the accuracy-variance trade-off and establishing why ordinary dimensionality reduction does not provide formal differential privacy.

---

### 7. Integration of Privacy, Utility, and Security Considerations
- **Privacy:** Formally bounded via $(\epsilon, \delta)$-Differential Privacy with $\delta = 10^{-5} \ll 1/N$ ($2.82 \times 10^{-4}$), ensuring that individual training records remain statistically indistinguishable.
- **Utility:** Evaluated using multi-class accuracy (73%–76%), weighted F1, and macro F1, documenting the specific utility penalty on the imbalanced `Enrolled` class.
- **Security:** Verified empirically by launching black-box Membership Inference Attacks, demonstrating that DP perturbation eliminates the generalization loss gap and reduces attacker advantage to random guessing.

---

### 8. Dataset Name and Description
**Predict Students' Dropout and Academic Success:** A higher-education tabular dataset created by researchers at the Polytechnic Institute of Portalegre, Portugal, capturing 4,424 student enrollments across 36 demographic, socio-economic, admission, and semester-by-semester academic attributes, predicting three retention outcomes (`Graduate`, `Dropout`, `Enrolled`).

---

### 9. Dataset Source URL / Access Reference
- **UCI Repository URL:** `https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success`
- **DOI:** `10.24432/C5MC89`
- **Direct Archive URL:** `https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip`

---

### 10. Dataset Creator / Publisher
- **Authors:** Valentim Realinho, Jorge Machado, Baptista, Mónica V. Martins
- **Publishing Institution:** Instituto Politécnico de Portalegre, Portugal & UCI Machine Learning Repository

---

### 11. Date Accessed or Collected
- **Original UCI Release Date:** December 13, 2021
- **Local Access & Checksum Audit Date:** October 2026

---

### 12. Dataset Version / Release
- Official Version: Initial Public Deposit (December 13, 2021; DOI: `10.24432/C5MC89`).

---

### 13. License and Permitted Usage Terms
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Usage Terms:** Freely usable, adaptable, and redistributable for academic, educational, and research analytics with creator attribution.

---

### 14. Original Dataset Size
- **Total Records:** 4,424 instances
- **Total Attributes:** 37 columns (36 predictor features + 1 Target column)
- **Missing Values:** 0 null values
- **Duplicate Records:** 0 duplicate rows

---

### 15. Final Dataset Size After Processing
- **Total Records:** 4,424 instances (100% retained, 0 dropped)
- **Features ($X$):** 36 standardized features
- **Target ($y$):** 1 integer-encoded outcome vector
- **Training Set (80%):** 3,539 instances $\times$ 36 features
- **Testing Set (20%):** 885 instances $\times$ 36 features

---

### 16. Data Collection, Preprocessing, and Transformations
1. Header whitespace and tab sanitization.
2. Label encoding: `Dropout` = 0, `Enrolled` = 1, `Graduate` = 2.
3. Stratified 80:20 train-test splitting (Random Seed 42).
4. Zero-leakage standardization: `StandardScaler` fitted exclusively on training data ($\mu_{train}, \sigma_{train}$) and applied to transform test data.
5. Conversion to PyTorch `TensorDataset` and `DataLoader` (Batch size: 64).

---

### 17. Privacy, Consent, Access, and Ethical Safeguards
The source institution permanently stripped direct personal identifiers (student names, social security numbers, and IDs). To safeguard against re-identification via quasi-identifiers and model weight memorization, the training process incorporates Differentially Private SGD with per-sample clipping ($C=1.0$) and Gaussian noise addition, eliminating individual record leakage.

---

### 18. Dataset Integrity Checksum / Hash
- **SHA-256:** `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae`
- **MD5:** `ee29c9ee9ea60654b5bb0a161c0a473d`
- **File Size:** 533,230 bytes

---

### 19. Dataset Citation / Reference
> Realinho, V., Machado, J., Baptista, & Martins, M. V. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.

---

### 20. Final Activity Approach
- **Primary Selection:** Practical implementation
- **Secondary Selections:** Experimental investigation, Comparative study

---

### 21. Implementation / Demonstration Methodology
- **Deep Learning Framework:** PyTorch (`torch>=2.0.0`)
- **Privacy Library:** Opacus (`opacus>=1.4.0, <=1.6.0`)
- **Architecture:** Multi-Layer Perceptron (36 $\to$ 64 $\to$ 32 $\to$ 3; Linear + ReLU + Dropout; no BatchNorm).
- **Optimization:** Non-private Adam vs DP-SGD with momentum 0.9 across $\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$ at $\delta = 10^{-5}$ ($C = 1.0$).
- **Experimental Audits:** Loss-threshold Membership Inference Attack (Yeom et al.), Warner's Randomized Response LDP simulation, $k$-Anonymity equivalence class evaluation, and PCA compression sweep.

---

### 22. Changes Made Since IA-1
1. **Empirical Security Evaluation:** Implemented and executed a black-box Membership Inference Attack (MIA) to audit actual privacy protection against training data extraction.
2. **Multi-Module Implementation:** Added working demonstrations for Local DP (Warner's RR on `Debtor`), $k$-Anonymity with Homogeneity Attack evidence, and Compressive Privacy (PCA dimension sweep).
3. **Cryptographic Provenance:** Computed and verified the exact SHA-256 checksum (`3ef126de...`) of the official UCI dataset.
4. **Enhanced Granular Metrics:** Documented per-class Precision, Recall, and F1 across all three classes, uncovering the disparate impact of DP noise on the minority `Enrolled` class.
5. **Upgraded Documentation:** Created 10 publication-quality figures, a formal DOCX academic report, unit tests, and structured submission answers.

---

### 23. Code / Notebook / Prototype Link
- **GitHub Repository:** `https://github.com/aarishsuradkar-aarnix/Student-Dropout-Differential-Privacy`
- **Colab Notebook:** `notebooks/IA2_Student_Dropout_Differential_Privacy.ipynb`

---

### 24. Demonstration Video / Screenshots Link
- Evidence is provided through the Google Colab executable notebook, 10 generated analytical figures (`figures/01_class_distribution.png` through `figures/10_compression_vs_accuracy.png`), saved evaluation tables (`results/metrics_summary.csv`, `results/mia_results.csv`), and the formal academic report. *(Video link: Not Applicable / Individual Code and Output Evidence Submitted).*
