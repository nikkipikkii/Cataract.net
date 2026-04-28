# Cataract.net : https://project-cataract-net.onrender.com/
# Cataract Detection & Lens Status Identification  
### Calibrated Deep Learning Ensemble for Robust Screening

**Independent Machine Learning Research Project**  
Bioinformatics · Computer Vision · Medical AI  

---

## Overview

This project develops a deep learning–based screening system for cataract detection and intraocular lens (IOL) identification under heterogeneous imaging conditions.

The system is designed for real-world deployment scenarios (e.g., mobile screening, non-specialist capture), prioritizing robust generalization, calibrated probability estimates, and reliable detection over unstable fine-grained grading.

---

## Key Results

- Binary ROC-AUC (Cataract Detection): 0.838  
- Lens Status Classification Accuracy: ~79.8%  
- Calibration Improvement (ECE): 0.122 → 0.108  

These results indicate strong ranking performance and improved confidence reliability under limited and heterogeneous data.

---

## Problem Formulation

The task is modeled as a multi-objective classification problem under real-world constraints:

- Cataract presence detection (primary task)  
- Lens status classification (auxiliary task)  

Key challenges include domain shift (clinical vs mobile imaging), limited severity labels, duplication risk, and overconfident model predictions.

---

## Methodology

### Model Architecture

- Backbone: EfficientNet-B0 (ImageNet pretrained)  
- Multi-task learning:
  - Severity classification (3-class)  
  - Lens status classification (binary)  
- Masked loss routing for heterogeneous supervision  
- Selective fine-tuning of higher-level layers  

### Training Strategy

- Two-phase augmentation:
  - Conservative domain adaptation  
  - Robustness simulation (blur, compression, illumination shifts)  
- Strict separation of training, calibration, and evaluation  

---

## Data Strategy

The system emphasizes data integrity over scale:

- Subject-level splitting (prevents leakage)  
- pHash-based deduplication  
- Controlled auxiliary dataset usage  

This ensures evaluation reflects real generalization, not memorization.

---

## Calibration & Ensembling

- Temperature scaling applied post-training  
- Independent calibration per model  
- Ensemble of three EfficientNet-B0 variants  

Final prediction:

P_final = (P_A + P_B1 + P_B2) / 3

Calibration improves alignment between predicted probabilities and actual outcomes.

---

## Model Performance

### Strengths

- Strong ranking ability (ROC-AUC > 0.83)  
- High recall for early cataract detection (~0.93)  
- Stable anatomical feature learning via auxiliary task  

### Limitations

- Overlap between severity classes  
- Reduced recall for certain categories  
- Dataset size and imbalance constraints  

These limitations are primarily data-driven.

---

## Deployment Design

The system is structured as a screening tool rather than a diagnostic system.

Key decisions:

- Collapse severity into Cataract Present vs No Cataract  
- Use probability thresholds instead of hard labels  
- Prioritize reliability and interpretability  

---

## Contributions

- Constraint-aware modeling under real-world conditions  
- Multi-task learning with heterogeneous supervision  
- Calibration-aware deep learning  
- Evidence-driven objective refinement  

---

---

## Limitations & Future Work

Future improvements are primarily data-driven:

- Larger and more diverse datasets  
- Better representation of advanced cases  
- Illumination-invariant features  
- Hierarchical detection and grading pipelines  

---

## Author

Nandini Bahukhandi  
Machine Learning · Applied AI · Systems Thinking  

---

## Full Report

See detailed methodology and analysis in the accompanying project report : https://drive.google.com/file/d/1LwCM_gLm0mnjXE4HrzhmARAUxHuIFzmX/view
