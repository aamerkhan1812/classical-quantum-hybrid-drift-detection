# Hybrid Quantum–Classical Drift Detection System  
A research-grade, end-to-end pipeline for detecting temporal concept drift in tabular data using hybrid quantum–classical MMD kernels, adaptive fusion, statistical thresholding, and a false-positive-constrained meta-learner.

This repository provides:

- A reproducible sliding-window drift detection framework  
- Classical and quantum MMD detectors  
- Adaptive hybrid fusion with rolling thresholds  
- Controlled drift injection for supervised evaluation  
- A calibrated meta-learner ensuring strict false positive control  
- Evaluation across three real-world fraud datasets  

---

## Table of Contents
- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Datasets](#datasets)  
- [Workflow Architecture](#workflow-architecture)  
- [Data Preprocessing](#data-preprocessing)  
- [Classical Drift Detection](#classical-drift-detection)  
- [Quantum Drift Detection](#quantum-drift-detection)  
- [Hybrid Fusion Layer](#hybrid-fusion-layer)  
- [Drift Injection Framework](#drift-injection-framework)  
- [Meta-Learner (FP-Constrained)](#meta-learner-fp-constrained)  
- [Results](#results)  
- [How to Run](#how-to-run)  
- [Environment & Dependencies](#environment--dependencies)  
- [Cite This Work](#cite-this-work)

---

## Overview

This project develops and evaluates a hybrid drift detection system designed for high-dimensional streaming fraud datasets. The pipeline integrates:

- Classical Kernel MMD  
- Quantum Fidelity Kernel (PennyLane-based)  
- Adaptive Hybrid Fusion  
- Rolling Statistical Thresholding  
- Controlled Drift Injection  
- Supervised Meta-Learning  

The goal is to detect distribution shifts (concept drift) while maintaining strict false-positive bounds.


---

## Datasets

Detailed description is provided in:  
**[data/README.md](data/README.md)**

This project evaluates three fraud datasets:

### 1. FDA Fraud Dataset (Kaggle)  
Processed into 150 sliding windows.

### 2. IEEE-CIS Fraud Detection (Kaggle)  
Dimensional reduction via Truncated SVD due to high feature count.

### 3. Credit Card Fraud (Kaggle ULB)  
PCA-based reduction.

---

## Workflow Architecture


---

## Data Preprocessing

Each dataset undergoes:

1. Missing value imputation  
2. Feature pruning & encoding  
3. Scaling  
4. PCA or Truncated SVD  
5. Window slicing into reference/test segments  

Notebook:  
`notebooks/01_preprocessing.ipynb`

---

## Classical Drift Detection

Implements Classical MMD using:

- RBF Kernel  
- Unbiased MMD Estimator  
- Permutation Testing  
- Subsampling for efficiency  

File: `src/detectors/classical_mmd.py`  
Notebook: `notebooks/02_classical.ipynb`

---

## Quantum Drift Detection

Quantum detection uses:

- Fidelity Kernel via PennyLane  
- Amplitude + Phase Encoding  
- 4–6 Qubits  
- Kernel-MMD Testing  

File: `src/detectors/quantum_mmd.py`  
Notebook: `notebooks/03_quantum.ipynb`

---

## Hybrid Fusion Layer

Combines classical & quantum signals with:

- Adaptive rolling threshold  
- Reliability-gating through kernel variance  
- Tunable weighting  

File: `src/detectors/fusion.py`  
Notebook: `notebooks/04_fusion.ipynb`

---

## Drift Injection Framework

Used to generate labels by simulating drift using:

- Mean-shift  
- Fraction-based row perturbation  
- Configurable intensity  

File: `src/injection/run_injection.py`  
Notebook: `notebooks/05_injection.ipynb`

---

## Meta-Learner (FP-Constrained)

A supervised model to consolidate detector signals with strict false-positive control.

Features used:

- Hybrid fusion metrics  
- Classical & quantum p-values  
- Kernel variance  
- Rolling statistics  

Model:

- Logistic Regression  
- 5-fold OOF probabilities  
- Custom FP-capped threshold search  

File: `src/meta_learner/train_meta.py`  
Notebook: `notebooks/06_meta_learner.ipynb`

---

## Results

### Credit Card Fraud Dataset

Performance:


---

## Environment & Dependencies

All dependencies and versions are captured in:

- `environment.yml`  
- `requirements.txt`

These mirror the exact versions used in the final experiments.

---

## Cite This Work

A. Khan, "Hybrid Quantum–Classical Drift Detection Using Fidelity Kernels and FP-Constrained Meta-Learning", 2025.


---

End of README.

