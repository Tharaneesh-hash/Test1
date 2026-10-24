import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME,
    NUM_CLASSES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    STATE_PREDICTION_AVAILABLE,
    STATE_REVIEW_REQUIRED,
    STATE_INVALID_INPUT,
    UNCERTAINTY_METRICS_PATH,
    RELIABILITY_PLOT
)


def calculate_predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Computes Shannon predictive entropy and normalized entropy for a discrete probability vector:
    H(p) = - sum(p_i * ln(p_i + eps))
    Normalized: H_norm(p) = H(p) / ln(num_classes) in [0, 1]
    """
    probs = np.clip(probabilities, eps, 1.0)
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = np.log(len(probabilities))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return entropy, normalized_entropy


def assess_prediction_uncertainty(
    probabilities: Optional[Union[np.ndarray, List[float]]],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    is_valid_input: bool = True
) -> Dict[str, Any]:
    """
    Evaluates model probabilities and applies the human-in-the-loop decision policy.
    
    States:
    - INVALID_INPUT: Corrupt, unreadable, or missing image.
    - REVIEW_REQUIRED: Maximum probability falls below threshold (high predictive uncertainty).
    - PREDICTION_AVAILABLE: Maximum probability meets or exceeds threshold.
    
    Medical Caution:
    Softmax confidence is an indicator of model agreement with training distribution,
    NOT a certified clinical diagnosis. Low confidence warrants clinical specialist review.
    """
    if not is_valid_input or probabilities is None:
        return {
            "decision_state": STATE_INVALID_INPUT,
            "predicted_idx": None,
            "predicted_code": None,
            "predicted_name": None,
            "confidence": 0.0,
            "predictive_entropy": None,
            "normalized_entropy": None,
            "threshold": threshold,
            "review_recommended": True,
            "recommendation_reason": "Input image could not be verified or processed reliably.",
            "probabilities_by_class": {}
        }

    probs = np.array(probabilities, dtype=np.float32)
    if np.isnan(probs).any() or np.sum(probs) <= 0:
        return {
            "decision_state": STATE_INVALID_INPUT,
            "predicted_idx": None,
            "predicted_code": None,
            "predicted_name": None,
            "confidence": 0.0,
            "predictive_entropy": None,
            "normalized_entropy": None,
            "threshold": threshold,
            "review_recommended": True,
            "recommendation_reason": "Prediction produced invalid numerical outputs (NaN/Zero sum).",
            "probabilities_by_class": {}
        }

    # Normalize if not strictly summing to 1
    probs = probs / np.sum(probs)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    entropy, norm_entropy = calculate_predictive_entropy(probs)

    prob_dict = {CLASS_CODES[i]: float(probs[i]) for i in range(len(CLASS_CODES))}
    predicted_code = CLASS_CODES[pred_idx]
    predicted_name = CODE_TO_NAME[predicted_code]

    if confidence >= threshold:
        decision_state = STATE_PREDICTION_AVAILABLE
        review_recommended = False
        reason = (
            f"Prediction meets acceptance criteria (Confidence {confidence*100:.1f}% >= threshold {threshold*100:.1f}%)."
        )
    else:
        decision_state = STATE_REVIEW_REQUIRED
        review_recommended = True
        reason = (
            f"Prediction does not meet acceptance criteria (Confidence {confidence*100:.1f}% < threshold {threshold*100:.1f}%). "
            f"Significant ambiguity among differential diagnoses. Dermatologist/clinical review recommended."
        )

    return {
        "decision_state": decision_state,
        "predicted_idx": pred_idx,
        "predicted_code": predicted_code,
        "predicted_name": predicted_name,
        "confidence": confidence,
        "predictive_entropy": entropy,
        "normalized_entropy": norm_entropy,
        "threshold": threshold,
        "review_recommended": review_recommended,
        "recommendation_reason": reason,
        "probabilities_by_class": prob_dict
    }


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Computes Expected Calibration Error (ECE) across discrete confidence bins.
    ECE = sum_b (|acc(B_b) - conf(B_b)| * |B_b| / N)
    """
    preds = np.argmax(probabilities, axis=1)
    confs = np.max(probabilities, axis=1)
    corrects = (preds == labels)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper) if i > 0 else (confs >= bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(corrects[in_bin]))
            avg_confidence_in_bin = float(np.mean(confs[in_bin]))
            abs_diff = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += abs_diff * prop_in_bin

            bin_details.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "sample_count": int(np.sum(in_bin)),
                "accuracy": accuracy_in_bin,
                "confidence": avg_confidence_in_bin,
                "calibration_gap": float(abs_diff)
            })

    return float(ece), bin_details


def tune_threshold_on_validation(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    target_max_error: float = 0.10,
    min_coverage: float = 0.70
) -> Dict[str, Any]:
    """
    Tunes abstention threshold strictly using validation data.
    Sweeps candidate thresholds to analyze coverage vs risk trade-off.
    Never inspects or uses test-set samples.
    """
    candidate_thresholds = np.linspace(0.50, 0.95, 46)
    val_confs = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)
    val_corrects = (val_preds == val_labels)
    total_val = len(val_labels)

    sweep_results = []
    selected_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    best_tradeoff_score = -float("inf")

    for thresh in candidate_thresholds:
        accepted_mask = val_confs >= thresh
        accepted_count = int(np.sum(accepted_mask))
        flagged_count = total_val - accepted_count
        coverage = float(accepted_count / total_val)

        if accepted_count > 0:
            accepted_acc = float(np.mean(val_corrects[accepted_mask]))
            accepted_error = float(1.0 - accepted_acc)
        else:
            accepted_acc = 1.0
            accepted_error = 0.0

        sweep_results.append({
            "threshold": float(round(thresh, 4)),
            "coverage": coverage,
            "accepted_count": accepted_count,
            "flagged_count": flagged_count,
            "accepted_accuracy": accepted_acc,
            "accepted_error_rate": accepted_error
        })

        # Selection heuristic: Maximize coverage while error rate <= target_max_error
        # Objective score combines high accuracy and acceptable coverage
        if coverage >= min_coverage:
            score = (accepted_acc * 1.5) + (coverage * 0.5) - (accepted_error * 2.0)
            if score > best_tradeoff_score:
                best_tradeoff_score = score
                selected_threshold = float(round(thresh, 2))

    ece, bin_details = compute_expected_calibration_error(val_probs, val_labels)

    results = {
        "methodology": "Validation-set risk-coverage threshold tuning (no test set leakage)",
        "selected_threshold": selected_threshold,
        "validation_samples_count": total_val,
        "expected_calibration_error": ece,
        "calibration_bins": bin_details,
        "threshold_sweep_summary": sweep_results[:10] + sweep_results[-5:]
    }

    return results


def plot_reliability_diagram(
    calibration_bins: List[Dict[str, float]],
    ece: float,
    save_path: Path = RELIABILITY_PLOT
):
    """
    Renders and saves a reliability diagram showing confidence vs observed accuracy.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    confs = [b["confidence"] for b in calibration_bins]
    accs = [b["accuracy"] for b in calibration_bins]
    counts = [b["sample_count"] for b in calibration_bins]

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")

    # Bar plot for bins
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2.0 for b in calibration_bins]
    ax.bar(
        bin_centers,
        accs,
        width=0.08,
        alpha=0.6,
        color="royalblue",
        edgecolor="black",
        label=f"Observed Accuracy (ECE = {ece:.3f})"
    )

    ax.scatter(bin_centers, confs, color="red", zorder=5, label="Mean Confidence")
    ax.set_title("Reliability Diagram (Validation Set Calibration)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Reliability diagram saved to {save_path}")
