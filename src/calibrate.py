"""
src/calibrate.py
----------------
Uncertainty threshold tuning and calibration evaluation module.

Runs model inference on the validation set to:
1. Collect predicted probabilities and ground-truth labels.
2. Call tune_threshold_on_validation() from src/uncertainty.py.
3. Save uncertainty metrics (ECE, threshold sweep, reliability bins) to disk.
4. Save reliability diagram plot.

Must be run AFTER model training is complete and vgg19_mpox_best.keras exists.

Usage:
    python -m src.calibrate [--fold fold1]

Output:
    results/metrics/uncertainty_metrics.json
    results/plots/reliability_diagram.png
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import (
    BEST_MODEL_PATH,
    UNCERTAINTY_METRICS_PATH,
    RELIABILITY_PLOT,
    DEFAULT_FOLD
)
from src.dataset import get_data_loaders
from src.model import load_model
from src.uncertainty import tune_threshold_on_validation, plot_reliability_diagram


def run_calibration(fold_name: str = DEFAULT_FOLD, device_str: str = None) -> dict:
    """
    Performs validation-set uncertainty calibration:
    - Loads trained model.
    - Runs inference on validation split to collect probabilities + labels.
    - Tunes abstention threshold using validation data only (no test leakage).
    - Computes Expected Calibration Error (ECE) and reliability bins.
    - Saves metrics to disk and plots reliability diagram.
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"====================================================")
    print(f"Mpox VGG19 — Uncertainty Calibration & Threshold Tuning")
    print(f"Device: {device} | Fold: {fold_name}")
    print(f"====================================================")

    # Load trained model
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {BEST_MODEL_PATH}. "
            "Run training first: python -m src.train"
        )

    print(f"Loading model from {BEST_MODEL_PATH}...")
    model = load_model(checkpoint_path=BEST_MODEL_PATH, device=device)
    model.eval()

    # Load validation data loader
    _, valid_loader, _, _ = get_data_loaders(fold_name=fold_name, batch_size=16)

    print(f"Running inference on validation set ({fold_name}/valid)...")
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in valid_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_probs = np.array(all_probs, dtype=np.float32)
    val_labels = np.array(all_labels, dtype=np.int64)

    print(f"Validation samples collected: {len(val_labels)}")
    val_acc = float(np.mean(np.argmax(val_probs, axis=1) == val_labels))
    print(f"Validation accuracy (before threshold): {val_acc * 100:.2f}%")

    # Run threshold tuning on validation set
    print("\nRunning threshold sweep on validation set...")
    uncertainty_results = tune_threshold_on_validation(
        val_probs=val_probs,
        val_labels=val_labels,
        target_max_error=0.10,
        min_coverage=0.70
    )

    selected_threshold = uncertainty_results["selected_threshold"]
    ece = uncertainty_results["expected_calibration_error"]
    calibration_bins = uncertainty_results["calibration_bins"]

    print(f"\nSelected Abstention Threshold: {selected_threshold:.2f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Calibration bins computed: {len(calibration_bins)}")

    # Print threshold sweep summary
    val_confs = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)
    val_corrects = val_preds == val_labels

    accepted_mask = val_confs >= selected_threshold
    accepted_count = int(np.sum(accepted_mask))
    coverage = float(accepted_count / len(val_labels))
    if accepted_count > 0:
        accepted_error = float(np.mean(~val_corrects[accepted_mask]))
        accepted_acc = float(np.mean(val_corrects[accepted_mask]))
    else:
        accepted_error = 0.0
        accepted_acc = 1.0

    print(f"\nAt Selected Threshold ({selected_threshold:.2f}):")
    print(f"  Coverage (fraction accepted): {coverage * 100:.1f}%")
    print(f"  Accepted samples: {accepted_count}/{len(val_labels)}")
    print(f"  Accuracy on accepted samples: {accepted_acc * 100:.2f}%")
    print(f"  Error rate on accepted samples: {accepted_error * 100:.2f}%")

    # Enrich results with final threshold evaluation
    uncertainty_results["final_threshold_evaluation"] = {
        "selected_threshold": selected_threshold,
        "validation_accuracy": val_acc,
        "coverage": coverage,
        "accepted_count": accepted_count,
        "flagged_count": len(val_labels) - accepted_count,
        "accepted_accuracy": accepted_acc,
        "accepted_error_rate": accepted_error,
        "ece": ece,
        "method_note": (
            "Threshold selected by maximizing (accuracy × 1.5 + coverage × 0.5 - error × 2.0) "
            "subject to coverage >= 70%. ECE computed across 10 confidence bins. "
            "Threshold is tuned strictly on validation data — test set is not involved."
        ),
        "clinical_caution": (
            "Softmax probability is NOT a certified clinical confidence measure. "
            "Threshold abstention identifies ambiguous predictions but does not guarantee "
            "that accepted predictions are clinically correct. All predictions require "
            "independent clinical validation."
        )
    }

    # Save results to disk
    UNCERTAINTY_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(UNCERTAINTY_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(uncertainty_results, f, indent=4)
    print(f"\nUncertainty metrics saved to {UNCERTAINTY_METRICS_PATH}")

    # Plot reliability diagram
    if calibration_bins:
        plot_reliability_diagram(calibration_bins, ece, save_path=RELIABILITY_PLOT)
    else:
        print("No calibration bins available — reliability diagram skipped.")

    print("\n====================================================")
    print("Calibration complete.")
    print(f"  Threshold: {selected_threshold:.2f}")
    print(f"  ECE:       {ece:.4f}")
    print(f"  Coverage:  {coverage * 100:.1f}%")
    print("====================================================")

    return uncertainty_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uncertainty calibration and threshold tuning for Mpox VGG19 model."
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=DEFAULT_FOLD,
        help="Dataset fold to use for validation-set calibration."
    )
    args = parser.parse_args()
    run_calibration(fold_name=args.fold)
