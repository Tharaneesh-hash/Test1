import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "FOLDS"
DEFAULT_FOLD = "fold1"

# Artifact and Results Directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# BINARY Classification — MPOX vs NON_MPOX
# ─────────────────────────────────────────────
BINARY_CLASS_NAMES = ["NON_MPOX", "MPOX"]   # index 0 = NON_MPOX, 1 = MPOX
NUM_BINARY_CLASSES = 2

MPOX_CLASS_INDEX = 1    # positive class
NON_MPOX_CLASS_INDEX = 0

# Mapping from original 6-class prefix → binary label
ORIGINAL_TO_BINARY = {
    "MKP":     1,   # Monkeypox (Mpox)   → MPOX
    "CHP":     0,   # Chickenpox         → NON_MPOX
    "CWP":     0,   # Cowpox             → NON_MPOX
    "HEALTHY": 0,   # Healthy skin       → NON_MPOX
    "HFMD":    0,   # HFMD               → NON_MPOX
    "MSL":     0,   # Measles            → NON_MPOX
}

# Human-readable names for original 6 classes (kept for deep error analysis)
ORIGINAL_CODE_TO_NAME = {
    "CHP":     "Chickenpox",
    "CWP":     "Cowpox",
    "HEALTHY": "Healthy",
    "HFMD":    "HFMD",
    "MKP":     "Monkeypox",
    "MSL":     "Measles",
}
ORIGINAL_CLASS_CODES = sorted(list(ORIGINAL_CODE_TO_NAME.keys()))

# ─────────────────────────────────────────────
# Master Prompt Decision Categories
# ─────────────────────────────────────────────
STATE_MPOX_SUSPECTED          = "MPOX_SUSPECTED"
STATE_MPOX_NOT_DETECTED        = "MPOX_NOT_DETECTED"
STATE_INCONCLUSIVE             = "INCONCLUSIVE_REFER_FOR_PCR"
STATE_INVALID_IMAGE            = "INVALID_IMAGE"

# Backward compatibility aliases
STATE_PREDICTION_AVAILABLE     = STATE_MPOX_SUSPECTED
STATE_REVIEW_REQUIRED          = STATE_INCONCLUSIVE
STATE_INVALID_INPUT            = STATE_INVALID_IMAGE

# ─────────────────────────────────────────────
# Image Specifications
# ─────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH  = 224
CHANNELS   = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# ─────────────────────────────────────────────
# Clinical Metadata Feature Dimension & Codes
# ─────────────────────────────────────────────
# Features: [lymphadenopathy, fever_relative_to_rash, rash_distribution, lesion_stage]
CLINICAL_FEATURE_DIM = 4

# Values:
# lymphadenopathy: -1=Unknown, 0=No, 1=Yes (swollen lymph nodes)
# fever_relative_to_rash: -1=Unknown, 0=Concurrent/No fever, 1=Prodromal (1-3 days before rash)
# rash_distribution: -1=Unknown, 0=Centripetal/Trunk first, 1=Centrifugal (face, palms, soles, genital)
# lesion_stage: -1=Unknown, 0=Macule/Papule, 1=Vesicle/Umbilicated Pustule, 2=Crusted scab

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
BATCH_SIZE        = 16
INITIAL_EPOCHS    = 15
INITIAL_LR        = 1e-3
FINE_TUNE_EPOCHS  = 5
FINE_TUNE_LR      = 1e-5
WEIGHT_DECAY      = 1e-3
LABEL_SMOOTHING   = 0.05
EARLY_STOP_PATIENCE = 5

# Selective Classification & Conformal Prediction Thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.70    # validation-calibrated
DEFAULT_ABSTAIN_ENTROPY_MAX  = 0.65    # normalized Shannon entropy threshold
CONFORMAL_SIGNIFICANCE_ALPHA = 0.10    # 90% coverage target in conformal prediction

# ─────────────────────────────────────────────
# Saved Model Checkpoint Paths
# ─────────────────────────────────────────────
BINARY_RESNET34_PATH  = MODELS_DIR / "binary_resnet34_best.pt"
BINARY_VGG19_PATH     = MODELS_DIR / "binary_vgg19_best.pt"
BINARY_EFFNET_PATH    = MODELS_DIR / "binary_efficientnet_b0_best.pt"
BINARY_CONVNEXT_PATH  = MODELS_DIR / "binary_convnext_best.pt"
BINARY_SWIN_PATH      = MODELS_DIR / "binary_swin_best.pt"
BINARY_MULTIMODAL_PATH = MODELS_DIR / "binary_multimodal_best.pt"
BINARY_ARCFACE_PATH   = MODELS_DIR / "binary_arcface_best.pt"

# Canonical best model pointer
BEST_BINARY_MODEL_PATH = MODELS_DIR / "binary_best_model.pt"
BEST_MODEL_NAME_FILE   = MODELS_DIR / "best_model_name.txt"

# Metadata / Calibration
CLASS_MAPPING_PATH       = MODELS_DIR / "binary_class_mapping.json"
MODEL_CONFIG_PATH        = MODELS_DIR / "binary_model_config.json"
CONFORMAL_PARAMS_PATH    = MODELS_DIR / "conformal_calibration.json"

# Metrics & Plots
BINARY_METRICS_PATH            = METRICS_DIR / "binary_test_metrics.json"
BINARY_HISTORY_PATH            = METRICS_DIR / "binary_training_history.json"
BINARY_UNCERTAINTY_PATH        = METRICS_DIR / "binary_uncertainty_metrics.json"
BINARY_SELECTIVE_METRICS_PATH  = METRICS_DIR / "binary_selective_metrics.json"
BINARY_CONFUSION_PLOT          = PLOTS_DIR  / "binary_confusion_matrix.png"
BINARY_TRAINING_CURVES         = PLOTS_DIR  / "binary_training_curves.png"
BINARY_ROC_CURVE               = PLOTS_DIR  / "binary_roc_curve.png"
BINARY_PR_CURVE                = PLOTS_DIR  / "binary_pr_curve.png"
BINARY_SELECTIVE_CURVE         = PLOTS_DIR  / "binary_selective_risk_coverage.png"

# Legacy paths (kept for backward compatibility)
VGG19_6CLASS_PATH     = MODELS_DIR / "vgg19_mpox_best.keras"
RESNET34_6CLASS_PATH  = MODELS_DIR / "resnet34_mpox_best.pt"
ENSEMBLE_METRICS_PATH = METRICS_DIR / "ensemble_test_metrics.json"
