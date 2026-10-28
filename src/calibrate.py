"""
src/calibrate.py — Binary Mpox Detection Calibration Module.

Uncertainty threshold calibration on the VALIDATION set only.
Steps:
  1. Load the selected binary model (best by val F1).
  2. Run inference on validation split -> collect probabilities + labels.
  3. Sweep thresholds to maximise (accuracy * 1.5 + coverage * 0.5 - error * 2.0)
     subject to coverage >= 70%.
  4. Compute Expected Calibration Error (ECE) and reliability bins.
  5. Calibrate Conformal Prediction non-conformity quantiles.
  6. Save metrics and plots.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from src.config import (
    BEST_BINARY_MODEL_PATH,
    BINARY_UNCERTAINTY_PATH,
    CONFORMAL_PARAMS_PATH,
    PLOTS_DIR,
    DEFAULT_FOLD,
    MPOX_CLASS_INDEX,
)
from src.dataset import get_data_loaders
from src.model import load_binary_model, MultimodalMpoxClassifier
from src.conformal import ConformalPredictor

RELIABILITY_PLOT = PLOTS_DIR / "binary_reliability_diagram.png"


def compute_ece(probs_positive: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_out = []
    n = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs_positive >= lo) & (probs_positive < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(probs_positive[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += (count / n) * abs(avg_conf - avg_acc)
        bins_out.append({
            "bin_lo": round(lo, 2),
            "bin_hi": round(hi, 2),
            "count": count,
            "avg_conf": avg_conf,
            "avg_accuracy": avg_acc,
        })

    return float(ece), bins_out


def sweep_thresholds(conf_scores: np.ndarray, correct: np.ndarray, thresholds: np.ndarray, min_coverage: float = 0.70):
    sweep = []
    best_t = thresholds[0]
    best_s = -999.0
    n = len(conf_scores)

    for t in thresholds:
        mask = conf_scores >= t
        accepted = int(mask.sum())
        coverage = accepted / n if n > 0 else 0.0

        if accepted == 0:
            acc_a = 0.0
            err_a = 0.0
        else:
            acc_a = float(correct[mask].mean())
            err_a = 1.0 - acc_a

        score = acc_a * 1.5 + coverage * 0.5 - err_a * 2.0
        sweep.append({
            "threshold": float(round(t, 2)),
            "coverage": round(coverage, 4),
            "accepted": accepted,
            "accuracy": round(acc_a, 4),
            "error": round(err_a, 4),
            "score": round(score, 4),
        })

        if coverage >= min_coverage and score > best_s:
            best_s = score
            best_t = float(round(t, 2))

    return sweep, best_t


def plot_reliability_diagram(bins_out: list, ece: float, save_path):
    bin_confs = [b["avg_conf"] for b in bins_out]
    bin_accs = [b["avg_accuracy"] for b in bins_out]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="steelblue", label="Model")
    ax.set_xlabel("Confidence (MPOX probability)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = str(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Reliability diagram saved -> {save_path}")


def run_calibration(fold_name: str = DEFAULT_FOLD, device_str: str = None) -> dict:
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("  Binary Mpox — Uncertainty Calibration & Conformal Tuning")
    print(f"  Device: {device} | Fold: {fold_name}")
    print("=" * 60)

    if not BEST_BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {BEST_BINARY_MODEL_PATH}")

    model = load_binary_model(BEST_BINARY_MODEL_PATH, device=device)
    model.eval()

    _, valid_loader, _, _ = get_data_loaders(fold_name=fold_name, batch_size=16, verbose=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in valid_loader:
            imgs = batch[0].to(device)
            labels = batch[1]
            if isinstance(model, MultimodalMpoxClassifier) and len(batch) >= 5:
                clin = batch[4].to(device)
                probs = model.predict_proba(imgs, clin)
            else:
                probs = model.predict_proba(imgs)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    val_probs = np.array(all_probs, dtype=np.float32)
    val_labels = np.array(all_labels, dtype=np.int64)

    mpox_probs = val_probs[:, MPOX_CLASS_INDEX]
    conf_scores = val_probs.max(axis=1)
    preds = val_probs.argmax(axis=1)
    correct = (preds == val_labels).astype(float)
    val_acc = float(correct.mean())

    print(f"  Validation samples: {len(val_labels)} | Raw Accuracy: {val_acc*100:.2f}%")

    # ECE
    ece, bins_out = compute_ece(mpox_probs, val_labels)
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")

    # Threshold sweep
    thresholds = np.arange(0.50, 0.96, 0.05)
    sweep, best_t = sweep_thresholds(conf_scores, correct, thresholds, min_coverage=0.70)

    # Conformal calibration
    conformal = ConformalPredictor(alpha=0.10)
    conformal.calibrate(val_probs, val_labels)
    print(f"  Conformal non-conformity quantile (alpha=0.10): q_val = {conformal.q_val:.4f}")

    accepted_mask = conf_scores >= best_t
    accepted_count = int(accepted_mask.sum())
    coverage_final = accepted_count / len(val_labels)
    acc_final = float(correct[accepted_mask].mean()) if accepted_count > 0 else 0.0
    err_final = 1.0 - acc_final

    print(f"  Selected threshold: {best_t:.2f}")
    print(f"  Coverage: {coverage_final*100:.1f}% ({accepted_count}/{len(val_labels)} samples)")
    print(f"  Accuracy on accepted: {acc_final*100:.2f}%")

    results = {
        "selected_threshold": best_t,
        "conformal_q_val": conformal.q_val,
        "expected_calibration_error": ece,
        "calibration_bins": bins_out,
        "threshold_sweep": sweep,
        "validation_accuracy": round(val_acc, 4),
        "final_threshold_evaluation": {
            "threshold": best_t,
            "coverage": round(coverage_final, 4),
            "accepted_count": accepted_count,
            "flagged_count": len(val_labels) - accepted_count,
            "accepted_accuracy": round(acc_final, 4),
            "accepted_error": round(err_final, 4),
            "ece": round(ece, 6),
        },
    }

    BINARY_UNCERTAINTY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BINARY_UNCERTAINTY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    conformal_data = {
        "alpha": conformal.alpha,
        "q_val": conformal.q_val,
        "selected_threshold": best_t,
        "val_samples": len(val_labels),
    }
    with open(CONFORMAL_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(conformal_data, f, indent=4)

    if bins_out:
        plot_reliability_diagram(bins_out, ece, RELIABILITY_PLOT)

    print("=" * 60)
    print("  Calibration complete.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mpox uncertainty calibration")
    parser.add_argument("--fold", type=str, default=DEFAULT_FOLD)
    args = parser.parse_args()
    run_calibration(fold_name=args.fold)
