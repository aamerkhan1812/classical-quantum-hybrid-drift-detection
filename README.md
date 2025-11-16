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

## Repository Structure

