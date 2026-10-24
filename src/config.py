import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "FOLDS"
DEFAULT_FOLD = "fold1"

# Sub-split paths for default fold
TRAIN_DIR = DATA_DIR / DEFAULT_FOLD / "train"
VALID_DIR = DATA_DIR / DEFAULT_FOLD / "valid"
TEST_DIR = DATA_DIR / DEFAULT_FOLD / "test"

# Artifact and Results Directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset & Image Specifications
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Canonical 6 Class Mappings
# Prefix code in dataset -> Human-readable medical class name
CODE_TO_NAME = {
    "CHP": "Chickenpox",
    "CWP": "Cowpox",
    "HEALTHY": "Healthy",
    "HFMD": "HFMD",
    "MKP": "Monkeypox",
    "MSL": "Measles",
}

CLASS_CODES = sorted(list(CODE_TO_NAME.keys()))
CLASS_NAMES = [CODE_TO_NAME[c] for c in CLASS_CODES]
NUM_CLASSES = len(CLASS_CODES)

# Target Mpox Class Definition
TARGET_CODE = "MKP"
TARGET_NAME = "Monkeypox"
TARGET_CLASS_INDEX = CLASS_CODES.index(TARGET_CODE)

# Model Architecture Hyperparameters
DENSE_UNITS = 256
DROPOUT_RATE = 0.4
GRADCAM_TARGET_LAYER = "block5_conv4"

# Training Hyperparameters
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LR = 1e-5

# Uncertainty Policy Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.75  # Selected via validation set analysis
STATE_PREDICTION_AVAILABLE = "PREDICTION_AVAILABLE"
STATE_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATE_INVALID_INPUT = "INVALID_INPUT"

# Saved Artifact Filenames
BEST_MODEL_PATH = MODELS_DIR / "vgg19_mpox_best.keras"
BACKUP_MODEL_WEIGHTS = MODELS_DIR / "vgg19_mpox_weights.weights.h5"
CLASS_MAPPING_PATH = MODELS_DIR / "class_mapping.json"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
HISTORY_PATH = METRICS_DIR / "training_history.json"
TEST_METRICS_PATH = METRICS_DIR / "test_metrics.json"
UNCERTAINTY_METRICS_PATH = METRICS_DIR / "uncertainty_metrics.json"
CONFUSION_MATRIX_PLOT = PLOTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PLOT = PLOTS_DIR / "training_curves.png"
RELIABILITY_PLOT = PLOTS_DIR / "reliability_diagram.png"
