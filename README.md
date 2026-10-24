# Transfer Learning Enabled Mpox Recognition Using VGG19, Grad-CAM, and Uncertainty-Aware Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Torchvision-EE4C2C.svg)](https://pytorch.org/)
[![UI-Streamlit](https://img.shields.io/badge/UI-Streamlit%201.63-FF4B4B.svg)](https://streamlit.io/)
[![License-Research](https://img.shields.io/badge/License-Academic%20Research%20Only-yellow.svg)](#medical-research-disclaimer)

An end-to-end medical deep learning research prototype for the automated differential recognition of **Monkeypox (Mpox)** skin lesions against common confounding dermatological conditions, integrated with **Explainable AI (Grad-CAM)** and an **Uncertainty-Aware Decision Support Policy** with human-in-the-loop referral.

---

## 📋 Table of Contents
1. [Project Overview & Objectives](#project-overview--objectives)
2. [Dataset & Clinical Categorization](#dataset--clinical-categorization)
3. [Architecture & Transfer Learning Design](#architecture--transfer-learning-design)
4. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
5. [Uncertainty-Aware Prediction & Referral Policy](#uncertainty-aware-prediction--referral-policy)
6. [Installation & Environment Setup](#installation--environment-setup)
7. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
8. [Automated Testing](#automated-testing)
9. [Evaluation Metrics & Benchmark Results](#evaluation-metrics--benchmark-results)
10. [Known Limitations](#known-limitations)
11. [Medical Research Disclaimer](#medical-research-disclaimer)

---

## 🎯 Project Overview & Objectives

Early identification of Mpox is critical for patient triage, isolation, and mitigating public health outbreaks. However, dermatological lesions caused by Mpox frequently mimic other infectious pustular, vesicular, and papular conditions.

This project delivers:
1. **Accurate Transfer Learning:** A tailored VGG19 model pretrained on ImageNet, utilizing a custom classification head and two-stage optimization (feature extraction + Block 5 fine-tuning).
2. **Explainability (Grad-CAM):** Gradient-weighted Class Activation Mapping computed directly on VGG19's final convolutional layer (`block5_conv4` / `features[34]`) to visualize which lesion features drove the classification.
3. **Uncertainty-Aware Decision Support:** Transparent confidence thresholding and Shannon predictive entropy to identify ambiguous, out-of-distribution, or low-confidence samples, routing them to `REVIEW_REQUIRED` for expert clinical review.
4. **Interactive Clinician Dashboard:** A responsive Streamlit application providing real-time inference, multi-class probability distributions, explainability overlays, and benchmark performance reports.

---

## 🩺 Dataset & Clinical Categorization

The dataset comprises **755 high-resolution dermatological lesion images** categorized into 6 differential classes:

| Class Code | Medical Diagnosis | Description | Total Samples (Fold 1) |
|:---|:---|:---|:---:|
| **MKP** | **Monkeypox (Mpox)** | Target zoonotic orthopoxvirus pustular lesions | 284 (Train: 202, Val: 55, Test: 27) |
| **CHP** | Chickenpox | Varicella-zoster virus vesicular rash | 75 (Train: 50, Val: 16, Test: 9) |
| **CWP** | Cowpox | Rare orthopoxvirus localized ulcerating skin lesions | 66 (Train: 49, Val: 10, Test: 7) |
| **HEALTHY** | Healthy Skin | Normal cutaneous tissue controls | 114 (Train: 83, Val: 22, Test: 9) |
| **HFMD** | Hand, Foot, and Mouth Disease | Coxsackievirus erythematous papulovesicular rash | 161 (Train: 116, Val: 29, Test: 16) |
| **MSL** | Measles | Morbillivirus maculopapular rash | 55 (Train: 37, Val: 12, Test: 6) |

### Leakage-Free Splitting Methodology
- Images follow the naming pattern `{PREFIX}_{PATIENT_ID}_{IMAGE_ID}.jpg`.
- **Patient/Group Stratification:** The dataset is partitioned into 5 balanced cross-validation folds (`fold1` through `fold5`).
- **Zero Patient Overlap:** All multiple lesion photos from the same patient/case are strictly isolated within a single split (`train`, `valid`, or `test`), preventing data leakage.
- **Untouched Test Set:** The test set (`fold1/test`, 74 images) is kept completely untouched during hyperparameter tuning, model checkpointing, and threshold selection.

---

## 🏗️ Architecture & Transfer Learning Design

### VGG19 Backbone Configuration
- **Pretrained Weights:** ImageNet-1K (`weights=VGG19_Weights.DEFAULT`)
- **Convolutional Backbone:** 16 conv layers organized into 5 blocks with max-pooling.
- **Backbone Freezing:** Initially frozen to preserve rich general visual representations and prevent catastrophic forgetting on medical data.

### Custom Classification Head
```
Input (224 x 224 x 3)
      │
[ VGG19 Features Backbone (Frozen: Block 1 - 5) ]
      │ (512 x 7 x 7)
[ AdaptiveAvgPool2d (1 x 1) ] (Global Average Pooling)
      │ (512)
[ Flatten ]
      │
[ Linear(512 -> 256) + ReLU ]
      │
[ Dropout(p = 0.40) ] (Regularization)
      │
[ Linear(256 -> 6) ] (Output Logits)
      │
[ Softmax Activation ] -> Class Probabilities
```

### Two-Stage Training Procedure
1. **Stage 1 (Classification Head Training):** Backbone weights frozen; only head parameters optimized using Adam ($lr = 10^{-3}$) with ReduceLROnPlateau ($patience = 2, factor = 0.2$) and Early Stopping ($patience = 4$).
2. **Stage 2 (Targeted Fine-Tuning):** Block 5 convolutional layers (`block5_conv1` to `block5_conv4`) unfrozen and fine-tuned using Adam with a substantially reduced learning rate ($lr = 10^{-5}$) to adapt high-level feature extractors to lesion morphology.
3. **Class Imbalance Mitigation:** Cross-entropy loss weighted by inverse class frequency ($w_c = \frac{N}{C \cdot N_c}$) using scikit-learn's balanced weighting.

---

## 🧠 Explainable AI (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM) produces coarse visual attribution maps highlighting lesion features influential in the decision:

1. **Feature Map Extraction:** Hooks capture activations $A^k$ at layer `block5_conv4` (`features[34]`).
2. **Gradient Attribution:** Backpropagation of the chosen class score $y_c$ computes $\frac{\partial y_c}{\partial A^k}$.
3. **Importance Weighting:** Global average pooling computes channel importance weights:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y_c}{\partial A_{i,j}^k}$$
4. **Rectified Linear Combination:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
5. **Normalization & Overlay:** Normalized to $[0, 1]$, interpolated to $(H, W)$, mapped through the Jet colormap, and alpha-blended with the original lesion photo ($\alpha = 0.40$).

*Interpretation Note:* Highlighted regions indicate model feature correlation, not pathological or clinical confirmation.

---

## ⚖️ Uncertainty-Aware Prediction & Referral Policy

Medical machine learning models can be overconfident on out-of-distribution or ambiguous lesions. To safeguard clinical utility, an uncertainty policy is enforced:

### Metrics Computed
- **Maximum Softmax Probability:** $P_{\max} = \max_c p_c$
- **Shannon Predictive Entropy:**
  $$H(p) = -\sum_{c=1}^C p_c \ln(p_c + \epsilon)$$
- **Normalized Entropy:** $\bar{H}(p) = \frac{H(p)}{\ln(C)} \in [0, 1]$

### Tri-State Decision Policy
1. `PREDICTION_AVAILABLE`: Input image is valid and $P_{\max} \ge \tau$ (default $\tau = 0.75$). The model exhibits high predictive confidence.
2. `REVIEW_REQUIRED`: Input image is valid but $P_{\max} < \tau$. The lesion presentation exhibits diagnostic ambiguity across differential classes. **Flagged for human clinical review.**
3. `INVALID_INPUT`: Image file is unreadable, corrupted, or structurally flawed.

### Threshold Calibration
The threshold $\tau$ was tuned exclusively on the **validation set** (without test set exposure) to guarantee acceptable error rates on accepted predictions while retaining $>75\%$ diagnostic coverage.

---

## 💻 Installation & Environment Setup

### Prerequisites
- Python 3.11+
- Windows, Linux, or macOS

### 1. Clone or Open the Repository
```bash
cd "C:\Users\ADMIN\OneDrive\Desktop\Test1"
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Run Automated Unit Tests
Verify model architecture, preprocessing, probability validity, Grad-CAM generation, uncertainty logic, and error handling:
```bash
$env:PYTHONPATH="C:\Users\ADMIN\OneDrive\Desktop\Test1"
python -m pytest tests/test_all_phases.py -v
```

### 2. Train the VGG19 Transfer Learning Model
Train Stage 1 (frozen backbone) and Stage 2 (Block 5 fine-tuning) on `fold1`:
```bash
$env:PYTHONPATH="C:\Users\ADMIN\OneDrive\Desktop\Test1"
python -m src.train --fold fold1 --epochs 10 --fine-tune-epochs 3 --batch-size 16
```
*Outputs saved:*
- Best model checkpoint: `models/vgg19_mpox_best.keras`
- Class mapping dictionary: `models/class_mapping.json`
- Model architecture configuration: `models/model_config.json`
- Training history log: `results/metrics/training_history.json`
- Loss and accuracy curves: `results/plots/training_curves.png`

### 3. Evaluate the Model on the Untouched Test Set
Run standalone evaluation on `fold1/test`:
```bash
$env:PYTHONPATH="C:\Users\ADMIN\OneDrive\Desktop\Test1"
python -m src.evaluate
```
*Outputs saved:*
- Quantitative metrics: `results/metrics/test_metrics.json`
- Confusion matrix plot: `results/plots/confusion_matrix.png`

### 4. Launch the Interactive Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

The automated test suite (`tests/test_all_phases.py`) executes 11 comprehensive tests:
- `test_01_model_instantiation`: Verifies VGG19 architecture and modules.
- `test_02_model_output_dimensions`: Validates 6-class output tensor shape.
- `test_03_image_preprocessing`: Ensures standard ImageNet resizing and normalization.
- `test_04_valid_class_probabilities`: Asserts softmax bounds $[0, 1]$ and sum-to-one constraint.
- `test_05_class_indices_and_mapping`: Validates bijective mapping across 6 classes.
- `test_06_gradcam_generation`: Verifies finite, non-null heatmap and overlay generation.
- `test_07_uncertainty_threshold_logic`: Tests `PREDICTION_AVAILABLE` vs `REVIEW_REQUIRED`.
- `test_08_invalid_image_handling`: Ensures graceful handling of None or corrupt inputs.
- `test_09_missing_checkpoint_error`: Verifies actionable error messages on missing files.
- `test_10_artifact_saving_and_loading`: Confirms persistence and reloadability of models.
- `test_11_metrics_and_config_format`: Validates JSON artifact schemas.

---

## 📊 Evaluation Metrics & Benchmark Results

Evaluated strictly on the **74 untouched test images** (`fold1/test`):

- **Overall Test Accuracy:** Evaluated and logged in `results/metrics/test_metrics.json`
- **Macro F1-Score:** Across all 6 differential diagnoses
- **Mpox Target Sensitivity (Recall):** $\frac{TP}{TP + FN}$
- **Mpox Target Specificity:** $\frac{TN}{TN + FP}$
- **Coverage vs. Risk:** Predictions with confidence $< \tau$ flagged for review to safeguard diagnostic reliability.

---

## ⚠️ Known Limitations
1. **Limited Sample Size:** 755 images across 6 classes reflects real-world dermatological dataset constraints; continuing data collection will improve generalizability.
2. **Skin Tone Representation:** The dataset primarily contains clinically curated web-scraped images; performance across diverse Fitzpatrick skin phototypes requires multi-center prospective validation.
3. **Softmax Confidence Calibration:** Softmax probabilities can reflect training dataset priors; uncertainty thresholds should be adjusted when deploying in novel clinical prevalence settings.

---

## 🛡️ Medical Research Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This software and trained deep learning models are developed strictly for **academic research, scientific evaluation, and educational demonstration**.  
> - It is **NOT** a certified medical device, diagnostic tool, or clinical decision support system approved by the FDA, EMA, or any national health authority.  
> - Outputs must **NEVER** be used as a standalone basis for patient diagnosis, prescription, surgical referral, or quarantine decisions.  
> - Definitive diagnosis of Monkeypox requires authorized molecular testing (e.g., real-time PCR) and comprehensive clinical evaluation by licensed medical practitioners.
