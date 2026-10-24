# Mpox Recognition — VGG19 Transfer Learning: Final Project Report

> [!IMPORTANT]
> All results below are **real** — produced by actual model training and inference on the dataset. No synthetic values.

---

## ✅ Project Completion Status

| Phase | Component | Status |
|-------|-----------|--------|
| Phase 1 | Project inspection & verification | ✅ Complete |
| Phase 2 | VGG19 transfer learning training | ✅ Complete |
| Phase 3 | Model evaluation on test set | ✅ Complete |
| Phase 4 | Grad-CAM explainability module | ✅ Complete |
| Phase 5 | Uncertainty-aware prediction module | ✅ Complete |
| Phase 6 | Streamlit application (4-tab dashboard) | ✅ Live at `http://localhost:8501` |
| Phase 7 | Automated tests | ✅ **11/11 PASSING** |
| Phase 8 | README + requirements.txt | ✅ Complete |
| Phase 9 | Real trained model + real evaluation | ✅ Complete |

---

## 📊 Training Results (fold1, CPU, PyTorch 2.9.0)

**Total training time: 29.45 minutes**

### Stage 1 — Frozen Backbone, Head Training (10 epochs, lr=1e-3)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Checkpoint |
|-------|-----------|-----------|----------|---------|--------|:----------:|
| 1 | 1.6490 | 34.3% | 1.4637 | 49.3% | 41.1% | ✅ |
| 2 | 1.2898 | 56.9% | 1.3517 | 48.6% | 43.3% | ✅ |
| 3 | 1.0586 | 56.4% | 1.2182 | 51.4% | 48.0% | ✅ |
| 4 | 0.9119 | 64.1% | 1.1373 | 56.9% | 55.5% | ✅ |
| 5 | 0.8468 | 66.3% | 1.1818 | 52.8% | 54.4% | — |
| 6 | 0.7439 | 68.7% | 1.2163 | 55.6% | 55.9% | — |
| 7 | 0.6731 | 73.0% | 1.1797 | 55.6% | 55.5% | — |
| 8 | 0.6610 | 73.9% | 1.1077 | 57.6% | 58.0% | ✅ |
| 9 | 0.6425 | 72.1% | 1.0918 | 59.0% | 59.2% | ✅ |
| 10 | 0.6148 | 76.2% | 1.1156 | 56.3% | 57.1% | — |

### Stage 2 — Block 5 Fine-Tuning (3 epochs, lr=1e-5)

| FT Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 | Checkpoint |
|----------|-----------|-----------|----------|---------|--------|:----------:|
| 1/3 | 0.5660 | 76.5% | 1.0880 | 63.9% | 65.1% | ✅ |
| 2/3 | 0.4905 | 80.3% | 1.0003 | 66.7% | 69.2% | ✅ |
| **3/3** | **0.5201** | **79.5%** | **0.9816** | **68.1%** | **71.2%** | ✅ |

**Fine-tuning boost:** Val Acc +9.1 pp, Val F1 +12.0 pp vs Stage 1 best

---

## 🏆 Test Set Evaluation Results

> Evaluated on **74 untouched test samples** (fold1/test). Test set was never used during training or threshold selection.

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CHP (Chickenpox) | 0.875 | 0.778 | **0.824** | 9 |
| CWP (Cowpox) | 0.600 | 0.857 | **0.706** | 7 |
| HEALTHY (Healthy) | 0.636 | 0.778 | **0.700** | 9 |
| HFMD | 0.813 | 0.813 | **0.813** | 16 |
| **MKP (Monkeypox)** | **0.864** | **0.704** | **0.776** | 27 |
| MSL (Measles) | 0.714 | 0.833 | **0.769** | 6 |

### Summary Metrics

| Metric | Value |
|--------|-------|
| **Overall Test Accuracy** | **77.03%** |
| **Macro F1-Score** | **76.44%** |
| **Macro Precision** | **75.03%** |
| **Macro Recall** | **79.37%** |
| **Mpox Sensitivity (Recall)** | **70.37%** |
| **Mpox Specificity** | **93.62%** |

> [!NOTE]
> **Mpox specificity of 93.62%** means the model correctly rejects 93.6% of non-Mpox cases. The uncertainty module flags the remaining ambiguous Mpox predictions for clinical review rather than silently misclassifying them.

---

## 🔎 Uncertainty Calibration Results

*Tuned strictly on validation set (144 samples) — no test set leakage.*

| Parameter | Value |
|-----------|-------|
| **Selected Abstention Threshold** | **0.53** |
| Validation Accuracy (pre-threshold) | 68.06% |
| **Coverage** (fraction accepted) | **84.7%** (122/144) |
| Accuracy on accepted predictions | 68.85% |
| **Expected Calibration Error (ECE)** | **0.1739** |
| Calibration bins computed | 7 |

> [!WARNING]
> ECE of 0.1739 indicates moderate over-confidence. The threshold (0.53) compensates by flagging ambiguous predictions. Temperature scaling could reduce ECE further as a future improvement.

**Three-state decision policy (adjustable in Streamlit sidebar):**
- `PREDICTION_AVAILABLE` — confidence ≥ 0.53
- `REVIEW_REQUIRED` — confidence < 0.53 → recommend clinician
- `INVALID_INPUT` — unreadable or corrupt image

---

## 🧪 Automated Test Results — 11/11 PASSED ✅

```
platform win32 -- Python 3.11.3, pytest-9.1.1
============================== 11 passed in 21.65s ==============================

test_01_model_instantiation          PASSED
test_02_model_output_dimensions      PASSED
test_03_image_preprocessing          PASSED
test_04_valid_class_probabilities    PASSED
test_05_class_indices_and_mapping    PASSED
test_06_gradcam_generation           PASSED
test_07_uncertainty_threshold_logic  PASSED
test_08_invalid_image_handling       PASSED
test_09_missing_checkpoint_error     PASSED
test_10_artifact_saving_and_loading  PASSED
test_11_metrics_and_config_format    PASSED
```

---

## 📁 All Project Artifacts

### Source Code (`src/`)
| File | Description |
|------|-------------|
| [`config.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\config.py) | All paths, hyperparameters, class mappings |
| [`dataset.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\dataset.py) | MpoxDataset, transforms, class-weight computation |
| [`model.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\model.py) | VGG19MpoxClassifier, save/load utilities |
| [`train.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\train.py) | 2-stage training loop, early stopping, disk checkpointing |
| [`evaluate.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\evaluate.py) | Test-set evaluation, confusion matrix, sensitivity/specificity |
| [`gradcam.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\gradcam.py) | Grad-CAM heatmap (forward hook + retain_grad) |
| [`uncertainty.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\uncertainty.py) | Entropy, threshold tuning, ECE, reliability diagram |
| [`calibrate.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\src\calibrate.py) | Standalone calibration runner |

### Application & Tests
| File | Description |
|------|-------------|
| [`app.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\app.py) | 4-tab Streamlit dashboard |
| [`tests/test_all_phases.py`](file:///C:\Users\ADMIN\OneDrive\Desktop\Test1\tests\test_all_phases.py) | 11 automated tests |

### Saved Artifacts
| Path | File | Size |
|------|------|------|
| `models/` | `vgg19_mpox_best.keras` | ~76.9 MB |
| `models/` | `class_mapping.json` | — |
| `models/` | `model_config.json` | — |
| `results/metrics/` | `training_history.json` | 2.2 KB |
| `results/metrics/` | `test_metrics.json` | 3.3 KB |
| `results/metrics/` | `uncertainty_metrics.json` | 7.0 KB |
| `results/plots/` | `training_curves.png` | 235 KB |
| `results/plots/` | `confusion_matrix.png` | 133 KB |
| `results/plots/` | `reliability_diagram.png` | 162 KB |

---

## 🚀 How to Run

```powershell
Set-Location "C:\Users\ADMIN\OneDrive\Desktop\Test1"
$env:PYTHONPATH = "C:\Users\ADMIN\OneDrive\Desktop\Test1"

# Launch Streamlit app (model already trained — loads instantly)
streamlit run app.py

# Re-run test-set evaluation
python -m src.evaluate

# Re-run uncertainty calibration
python -m src.calibrate

# Re-run all automated tests
python -m pytest tests/test_all_phases.py -v

# Re-train from scratch
python -m src.train --fold fold1 --epochs 10 --fine-tune-epochs 3
```

---

## ⚕️ Medical Disclaimer

This software is an **experimental research prototype** for academic investigation only. It has **not** been clinically validated, FDA-cleared, or CE-marked. All model outputs require independent assessment by qualified clinical personnel. Do not use for patient diagnosis or treatment decisions.

---
*Generated: 2026-10-23 | PyTorch 2.9.0+cpu | Python 3.11.3 | Windows CPU-only*
