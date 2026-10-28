"""
src/conformal.py — Selective Classification & Conformal Prediction Module.

Implements Algorithm 6:
  1. Conformal Prediction Set Estimation (Guaranteed coverage 1 - alpha)
  2. Risk-Controlled Selective Classification (Abstention Gate)
  3. Four Clinical Decision States:
     - MPOX_SUSPECTED
     - MPOX_NOT_DETECTED
     - INCONCLUSIVE_REFER_FOR_PCR
     - INVALID_IMAGE
  4. Accuracy vs. Coverage Tradeoff Evaluation
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import (
    STATE_MPOX_SUSPECTED,
    STATE_MPOX_NOT_DETECTED,
    STATE_INCONCLUSIVE,
    STATE_INVALID_IMAGE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_ABSTAIN_ENTROPY_MAX,
    CONFORMAL_SIGNIFICANCE_ALPHA,
    CONFORMAL_PARAMS_PATH,
    BINARY_SELECTIVE_METRICS_PATH,
    BINARY_SELECTIVE_CURVE,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
)


def compute_shannon_entropy(probs: np.ndarray, eps: float = 1e-8) -> float:
    """Computes normalized Shannon entropy in [0, 1]."""
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)  # Max for 2 classes is 1.0


class ConformalPredictor:
    """
    Split-Conformal Prediction for Binary Medical Screening.
    Guarantees that the true diagnosis is contained in the predicted set with
    probability >= 1 - alpha on exchangeable test data.
    """
    def __init__(self, alpha: float = CONFORMAL_SIGNIFICANCE_ALPHA):
        self.alpha = alpha
        self.q_val: float = 0.50
        self.calibrated: bool = False

    def calibrate(self, val_probs: np.ndarray, val_labels: np.ndarray):
        """
        Calibrates the non-conformity threshold using validation set scores.
        Non-conformity score: S_i = 1 - p(y_i | x_i).
        """
        n = len(val_labels)
        scores = np.zeros(n)
        for i in range(n):
            true_label = int(val_labels[i])
            scores[i] = 1.0 - val_probs[i, true_label]

        # Conformal quantile with finite-sample correction: ceil((n+1)(1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, level))
        self.q_val = float(np.quantile(scores, level, method="higher"))
        self.calibrated = True

    def predict_set(self, probs: np.ndarray) -> List[int]:
        """
        Generates conformal prediction set Gamma(x).
        Includes class c if 1 - p(c | x) <= q_val  => p(c | x) >= 1 - q_val.
        """
        p_threshold = max(0.0, 1.0 - self.q_val)
        pred_set = [c for c in [0, 1] if probs[c] >= p_threshold]
        if not pred_set:
            # Safe fallback if empty: include argmax
            pred_set = [int(np.argmax(probs))]
        return pred_set


class SelectiveClassifier:
    """
    Selective Classification engine mapping model probabilities and conformal
    sets to the four mandated clinical states.
    """
    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        entropy_threshold: float = DEFAULT_ABSTAIN_ENTROPY_MAX,
        conformal_predictor: Optional[ConformalPredictor] = None,
    ):
        self.conf_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.conformal = conformal_predictor or ConformalPredictor()

    def classify(
        self,
        probs: Optional[List[float]],
        is_valid_image: bool = True,
    ) -> Dict[str, Any]:
        """
        Classifies an input sample into one of four states:
          - MPOX_SUSPECTED
          - MPOX_NOT_DETECTED
          - INCONCLUSIVE_REFER_FOR_PCR
          - INVALID_IMAGE
        """
        if not is_valid_image or probs is None or any(np.isnan(p) for p in probs):
            return {
                "decision_state": STATE_INVALID_IMAGE,
                "mpox_probability": 0.0,
                "non_mpox_probability": 0.0,
                "calibrated_model_score": 0.0,
                "confidence": 0.0,
                "entropy": 1.0,
                "prediction_set": [],
                "abstain_recommended": True,
                "clinical_message": "Invalid or corrupted image. Cannot perform screening. Please upload a clear lesion photo.",
            }

        p_array = np.array(probs, dtype=np.float32)
        p_non_mpox = float(p_array[NON_MPOX_CLASS_INDEX])
        p_mpox = float(p_array[MPOX_CLASS_INDEX])

        confidence = max(p_non_mpox, p_mpox)
        entropy = compute_shannon_entropy(p_array)
        pred_idx = int(np.argmax(p_array))

        pred_set = self.conformal.predict_set(p_array) if self.conformal.calibrated else [pred_idx]

        # Clinical triage logic
        is_ambiguous_conformal = len(pred_set) > 1
        is_low_confidence = confidence < self.conf_threshold
        is_high_entropy = entropy > self.entropy_threshold

        if is_low_confidence or is_high_entropy or is_ambiguous_conformal:
            decision_state = STATE_INCONCLUSIVE
            abstain = True
            msg = (
                "Diagnostic ambiguity detected. Lesion morphology exhibits visual overlap with "
                "benign viral mimics (e.g. Chickenpox, Cowpox). Formal clinical evaluation and "
                "laboratory confirmation (e.g. real-time PCR) are strongly recommended."
            )
        else:
            abstain = False
            if pred_idx == MPOX_CLASS_INDEX:
                decision_state = STATE_MPOX_SUSPECTED
                msg = (
                    "Lesion appearance is consistent with Mpox rash characteristics. "
                    "Prompt medical consultation, patient isolation precautions, and molecular PCR testing are advised."
                )
            else:
                decision_state = STATE_MPOX_NOT_DETECTED
                msg = (
                    "Lesion appearance does not meet the verified threshold for Mpox suspicion. "
                    "However, negative screening does not rule out infection. Monitor symptoms closely."
                )

        return {
            "decision_state": decision_state,
            "mpox_probability": round(p_mpox, 4),
            "non_mpox_probability": round(p_non_mpox, 4),
            "calibrated_model_score": round(p_mpox, 4),
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "prediction_set": pred_set,
            "abstain_recommended": abstain,
            "clinical_message": msg,
        }


def evaluate_selective_classification(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the accuracy vs. coverage tradeoff across a sweep of rejection thresholds.
    Generates the selective risk-coverage curve.
    """
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    n_samples = len(y_true)
    y_pred_all = np.argmax(y_probs, axis=1)
    confs = np.max(y_probs, axis=1)

    sweep_results = []
    for tau in thresholds:
        accepted_mask = confs >= tau
        n_accepted = int(np.sum(accepted_mask))
        coverage = n_accepted / n_samples if n_samples > 0 else 0.0

        if n_accepted > 0:
            acc = float(accuracy_score(y_true[accepted_mask], y_pred_all[accepted_mask]))
            f1 = float(f1_score(y_true[accepted_mask], y_pred_all[accepted_mask], pos_label=1, zero_division=0))
            err = 1.0 - acc
        else:
            acc = 1.0
            f1 = 0.0
            err = 0.0

        sweep_results.append({
            "threshold": tau,
            "coverage": round(coverage, 4),
            "accepted_count": n_accepted,
            "referred_count": n_samples - n_accepted,
            "accuracy_on_accepted": round(acc, 4),
            "error_rate": round(err, 4),
            "f1_score": round(f1, 4),
        })

    # Plot selective risk-coverage curve
    coverages = [r["coverage"] * 100 for r in sweep_results]
    accuracies = [r["accuracy_on_accepted"] * 100 for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(coverages, accuracies, "b-o", linewidth=2, label="Accuracy on Accepted")
    ax1.axhline(95.0, color="green", linestyle="--", alpha=0.7, label="95% Target Line")
    ax1.set_xlabel("Clinical Coverage (%)", fontsize=11)
    ax1.set_ylabel("Accuracy on Accepted Cases (%)", fontsize=11, color="blue")
    ax1.set_title("Selective Classification: Accuracy vs. Coverage Tradeoff", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    plt.tight_layout()
    BINARY_SELECTIVE_CURVE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(BINARY_SELECTIVE_CURVE, dpi=200)
    plt.close()

    summary = {
        "sweep": sweep_results,
        "selective_curve_plot": str(BINARY_SELECTIVE_CURVE),
    }

    with open(BINARY_SELECTIVE_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary
