# Dataset Provenance and Integrity Verification

**Project Title:** Student Dropout Prediction Using Differential Privacy  
**Assessment:** IA-2 — Implementation, Evaluation and Technical Outcomes  
**Student:** Aarish Suradkar (Roll Number: 16014223001)  
**Course:** Security and Privacy for Machine Learning  

---

## 1. Dataset Identification and Repository Metadata

- **Dataset Name:** Predict Students' Dropout and Academic Success
- **Repository:** UCI Machine Learning Repository (Dataset ID: 697)
- **Persistent Identifier (DOI):** [https://doi.org/10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- **Repository URL:** [https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Direct Archive URL:** `https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip`
- **Creators / Authors:** Valentim Realinho, Jorge Machado, Baptista, Mónica V. Martins
- **Publishing Institution:** Instituto Politécnico de Portalegre, Portugal & UCI ML Repository
- **Date Donated / Released on UCI:** December 13, 2021
- **Date Verified Locally for IA-2:** October 2026

---

## 2. Cryptographic Integrity and Checksum Verification

To satisfy Criterion 3 of the IA-2 marking rubric, the local dataset file was audited using cryptographic hashing:

| Attribute | Measured Value | Verification Status |
| :--- | :--- | :---: |
| **File Path Audited** | `data/data.csv` | Confirmed |
| **File Size in Bytes** | 533,230 bytes | Confirmed |
| **SHA-256 Checksum** | `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae` | **VERIFIED (Exact Match)** |
| **MD5 Checksum** | `ee29c9ee9ea60654b5bb0a161c0a473d` | **VERIFIED (Exact Match)** |

---

## 3. License and Permitted Usage Terms

- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  - *Academic / Research Rights:* The dataset may be freely shared, adapted, modified, and redistributed for educational and research analytics.
  - *Conditions:* Attribution to the original creators and Polytechnic Institute of Portalegre must be preserved.
- **Redistribution Compliance:** The dataset is redistributed locally as `data/data.csv` in full compliance with CC BY 4.0 terms.

---

## 4. Cohort Dimensions and Transformation Accounting

| Stage | Instances (Rows) | Attributes (Columns) | Target Status |
| :--- | :---: | :---: | :--- |
| **Original UCI Dataset** | 4,424 | 37 (36 features + 1 Target) | Semicolon-delimited, raw Portuguese higher-ed records |
| **Missing Values Audit** | 4,424 | 37 | **0 null values** across all columns |
| **Duplicate Audit** | 4,424 | 37 | **0 duplicate rows** detected |
| **Final Processed Matrix ($X$)** | 4,424 | 36 predictor features | Standardized with zero-leakage `StandardScaler` |
| **Final Target Vector ($y$)** | 4,424 | 1 outcome column | Encoded: Dropout (0), Enrolled (1), Graduate (2) |
| **Training Partition ($80\%$)** | 3,539 | 36 features | Stratified split (Seed 42) |
| **Testing Partition ($20\%$)** | 885 | 36 features | Stratified split (Seed 42) |

---

## 5. Ethical Safeguards and Access Restrictions

1. **Student De-identification:** Direct identifiers (names, Portuguese national ID numbers, tax IDs, student email addresses) were permanently expunged by the source institution prior to public release.
2. **Quasi-Identifier Sensitivity:** The records retain demographic, socio-economic (parental occupations, tuition debt status), and academic grades. Although direct linkage is non-trivial, training an unprotected deep neural network on this data exposes students to **Membership Inference Attacks (MIA)**.
3. **Differential Privacy Guarantee:** This IA-2 project implements $(\epsilon, \delta)$-DP via DP-SGD to bound the privacy leakage of individual student training records, providing formal mathematical resistance against re-identification and inference attacks.
