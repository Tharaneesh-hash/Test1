# Binary Mpox Skin-Lesion Detection AI System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Torchvision-EE4C2C.svg)](https://pytorch.org/)
[![UI-Streamlit](https://img.shields.io/badge/UI-Streamlit%201.63-FF4B4B.svg)](https://streamlit.io/)
[![License-Research](https://img.shields.io/badge/License-Academic%20Research%20Only-yellow.svg)](#medical-research-disclaimer)

An end-to-end, leak-free medical deep learning system for automated binary recognition of **Monkeypox (Mpox)** vs. **NON_MPOX** skin lesions (Chickenpox, Cowpox, HFMD, Measles, Healthy skin), integrated with **Explainable AI (Grad-CAM)** and a **Validation-Calibrated Uncertainty Decision Policy**.

---

## 📋 Table of Contents
1. [Project Overview & Objective](#project-overview--objective)
2. [Dataset & Binary Mapping](#dataset--binary-mapping)
3. [Model Architectures & Training Design](#model-architectures--training-design)
4. [Uncertainty & Abstention Policy](#uncertainty--abstention-policy)
5. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
6. [Interactive Streamlit Application](#interactive-streamlit-application)
7. [Automated Verification & Unit Tests](#automated-verification--unit-tests)
8. [Benchmark Evaluation Results](#benchmark-evaluation-results)
9. [Execution & Reproduction Guide](#execution--reproduction-guide)
10. [Known Limitations](#known-limitations)
11. [Medical Research Disclaimer](#medical-research-disclaimer)

---

## 🎯 Project Overview & Objective

The objective of this upgrade is to convert a multi-class skin lesion recognition prototype into a reliable, high-sensitivity binary classifier:
- **Class 1 (MPOX):** Confirmed Monkeypox skin lesions (`MKP`).
- **Class 0 (NON_MPOX):** Non-Mpox lesions or controls (`CHP`, `CWP`, `HFMD`, `MSL`, `HEALTHY`).

Key engineering requirements strictly enforced:
- **Zero Data Leakage:** Final test set (`fold1/test`, 74 images) was kept completely untouched until single final evaluation. Model selection and uncertainty threshold tuning were performed strictly on training and validation splits.
- **Three Pretrained Architectures Evaluated:** VGG19, ResNet34, and EfficientNet-B0 with ImageNet pretrained backbones, two-stage fine-tuning, and Cosine Annealing.
- **Original Condition Tracking:** Metadata for the original 6 dermatological conditions is preserved to analyze false positives and false negatives without displaying them in the clinical user UI.

---

## 🩺 Dataset & Binary Mapping

The dataset contains 755 dermatological images organized in 5 folds with patient/source grouping (no patient images span splits).

### Fold 1 Split Distribution
| Split | Total Samples | MPOX (Class 1) | NON_MPOX (Class 0) | Original Classes Breakdown |
|:---|:---:|:---:|:---:|:---|
| **Train** | 537 | 202 | 335 | MKP: 202, CHP: 50, CWP: 49, HFMD: 116, MSL: 37, HEALTHY: 83 |
| **Validation** | 144 | 55 | 89 | MKP: 55, CHP: 16, CWP: 10, HFMD: 29, MSL: 12, HEALTHY: 22 |
| **Test (Untouched)** | 74 | 27 | 47 | MKP: 27, CHP: 9, CWP: 7, HFMD: 16, MSL: 6, HEALTHY: 9 |

---

## 🏗️ Model Architectures & Training Design

Three pretrained architectures were converted to binary classification with custom 2-class heads:
1. **BinaryEfficientNetB0:** Fast parameter-efficient baseline. Fine-tuned last MBConv blocks (`features[6..8]`).
2. **BinaryResNet34:** Deep residual network. Fine-tuned `layer3` and `layer4`.
3. **BinaryVGG19:** Classical 16-conv feature extractor. Fine-tuned `features[28..36]` (Block 5).

### Two-Stage Optimization Pipeline
- **Stage 1 (Head Warmup):** Backbone frozen, AdamW ($lr = 10^{-3}$, weight decay $10^{-3}$), label smoothing ($0.05$), Cosine Annealing LR.
- **Stage 2 (Backbone Fine-Tuning):** Targeted blocks unfrozen with reduced learning rate ($lr = 10^{-5}$) and early stopping monitored on **validation MPOX F1**.

### Validation Comparison (No Test-Set Influence)
| Architecture | Best Val MPOX F1 | Best Val Accuracy | Best Val ROC-AUC | Selected |
|:---|:---:|:---:|:---:|:---:|
| **BinaryEfficientNetB0** | 0.6838 | 75.7% | 0.8074 | No |
| **BinaryResNet34** | **0.7193** | **79.2%** | **0.8784** | **Yes (Best Single)** |
| **BinaryVGG19** | 0.7154 | 78.5% | 0.8547 | No |

---

## ⚖️ Uncertainty & Abstention Policy

Softmax probabilities can be uncalibrated and overly confident. A validation-calibrated thresholding policy is implemented:
- **Decision States:**
  - `PREDICTION_AVAILABLE`: Confidence $\ge \tau$ (calibrated $\tau = 0.70$). Prediction is presented with confidence score.
  - `REVIEW_REQUIRED`: Confidence $< \tau$ or high predictive entropy. Lesion is flagged for expert clinician review.
  - `INVALID_INPUT`: Image corrupted or non-dermatological.
- **Calibration Metrics (Validation Set):**
  - **Selected Threshold:** $\tau = 0.70$
  - **Diagnostic Coverage:** $87.5\%$ (126 of 144 validation cases accepted)
  - **Accuracy on Accepted Cases:** $81.75\%$ (vs $77.78\%$ raw accuracy)
  - **Expected Calibration Error (ECE):** $0.1844$

---

## 🧠 Explainable AI (Grad-CAM)

Architecture-agnostic Grad-CAM maps feature attributions from the final convolutional layer:
- **ResNet34:** `layer4[-1].conv2`
- **VGG19:** `features[34]` (`block5_conv4`)
- **EfficientNet-B0:** `features[8][0]`

Heatmaps are normalized to $[0, 1]$, color-mapped via Jet, and alpha-blended ($\alpha = 0.40$) over the lesion image to show visual attention.

---

## 📊 Benchmark Evaluation Results

Evaluated strictly on the **74 untouched test images** (`fold1/test`):

### 1. Primary Benchmark Metrics
| Metric | Best Single Model (`BinaryResNet34`) | Ensemble (`VGG19` + `EfficientNetB0`) |
|:---|:---:|:---:|
| **Test Accuracy** | **81.08%** (95% CI: 71.6% – 90.5%) | **81.08%** |
| **ROC-AUC** | **0.8905** | **0.9015** |
| **MPOX Sensitivity (Recall)** | **81.48%** (22 / 27 detected) | **81.48%** (22 / 27 detected) |
| **MPOX Specificity** | **80.85%** (38 / 47 non-mpox correct) | **80.85%** (38 / 47 non-mpox correct) |
| **MPOX Precision** | **70.97%** | **70.97%** |
| **MPOX F1-Score** | **75.86%** (95% CI: 60.8% – 87.9%) | **75.86%** |
| **Non-Mpox Precision** | **88.37%** | **88.37%** |
| **Non-Mpox Recall** | **80.85%** | **80.85%** |
| **Non-Mpox F1-Score** | **84.44%** | **84.44%** |
| **Macro F1-Score** | **80.15%** | **80.15%** |

### 2. Confusion Matrix (`BinaryResNet34`)
- **True Positives (TP):** 22 (Mpox correctly detected)
- **False Negatives (FN):** 5 (Mpox missed as Non-Mpox)
- **False Positives (FP):** 9 (Non-Mpox falsely flagged)
- **True Negatives (TN):** 38 (Non-Mpox correctly ruled out)

### 3. Granular Error Breakdown (By Original Condition)
- **False Negatives ($FN = 5$):** Real Monkeypox cases with subtle or atypical macules.
- **False Positives ($FP = 9$):**
  - **Chickenpox (`CHP`):** 6 cases (primary morphological mimic: similar vesicular/pustular rash).
  - **Cowpox (`CWP`):** 2 cases (orthopoxvirus relative with umbilicated appearance).
  - **Healthy Skin (`HEALTHY`):** 1 case.

---

## 🚀 Execution & Reproduction Guide

### 1. Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_all_phases.py -v
```

### 2. Train All Architectures & Select Best
```powershell
.\.venv\Scripts\python.exe run_training.py
```

### 3. Evaluate the Selected Model on Test Set
```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

### 4. Calibrate Uncertainty Thresholds on Validation Data
```powershell
.\.venv\Scripts\python.exe -m src.calibrate
```

### 5. Evaluate Ensembling
```powershell
.\.venv\Scripts\python.exe -m src.evaluate_ensemble
```

### 6. Launch the Streamlit Clinician App
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

## ⚠️ Known Limitations

1. **Test Accuracy Reality Check:** Overall test accuracy is **81.08%** (ROC-AUC: 0.8905 to 0.9015). Achieving $\ge 90\%$ test accuracy was not supported by the 755-image dataset without synthetic fabrication or data leakage.
2. **Morphological Mimicry:** Chickenpox represents $66.7\%$ ($6/9$) of false positive errors due to near-identical vesicular presentation.
3. **Small Sample Size:** With only 27 positive Mpox test images, single misclassifications shift accuracy by $\approx 1.35\%$. Independent external multi-center cohorts are recommended.

---

## 🛡️ Medical Research Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This software and trained deep learning models are developed strictly for **academic research, scientific evaluation, and educational demonstration**.  
> - It is **NOT** a certified medical device or clinical diagnostic tool approved by the FDA, EMA, or national health authorities.  
> - Definitive diagnosis of Monkeypox requires authorized molecular laboratory assays (such as real-time PCR) in conjunction with epidemiological history and clinical assessment by licensed medical practitioners.
