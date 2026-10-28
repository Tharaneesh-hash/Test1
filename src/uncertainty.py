"""
src/uncertainty.py — Binary Uncertainty & Decision Support Module.

Maps model probabilities into the Master Prompt's four mandated clinical decision states:
    1. MPOX_SUSPECTED
    2. MPOX_NOT_DETECTED
    3. INCONCLUSIVE_REFER_FOR_PCR
    4. INVALID_IMAGE

Also provides backward-compatible state aliases (PREDICTION_AVAILABLE, REVIEW_REQUIRED, INVALID_INPUT).
"""

import math
import numpy as np
from typing import Optional, Union, List

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    STATE_MPOX_SUSPECTED,
    STATE_MPOX_NOT_DETECTED,
    STATE_INCONCLUSIVE,
    STATE_INVALID_IMAGE,
    STATE_PREDICTION_AVAILABLE,
    STATE_REVIEW_REQUIRED,
    STATE_INVALID_INPUT,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    BINARY_CLASS_NAMES,
)

_NUM_CLASSES = 2  # Binary: 0=NON_MPOX, 1=MPOX


def calculate_predictive_entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats, clamped to avoid log(0)."""
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


def assess_prediction_uncertainty(
    probs: Optional[Union[np.ndarray, List[float]]],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    is_valid_input: bool = True,
) -> dict:
    """
    Returns structured clinical screening decision assessment for binary Mpox detection.

    Args:
        probs:          Array of [non_mpox_prob, mpox_prob] from model.predict_proba().
        threshold:      Confidence threshold calibrated on validation data.
        is_valid_input: Set False for corrupt/unreadable images.

    Returns dict with:
        decision_state:       MPOX_SUSPECTED / MPOX_NOT_DETECTED / INCONCLUSIVE_REFER_FOR_PCR / INVALID_IMAGE
        predicted_class:      "MPOX" or "NON_MPOX"
        predicted_index:      0 or 1
        mpox_probability:     float [0, 1]
        non_mpox_probability: float [0, 1]
        calibrated_model_score: float [0, 1]
        confidence:           max(probs)
        predictive_entropy:   float (nats)
        normalised_entropy:   float [0, 1]
        review_recommended:   bool
        recommendation_reason: str
    """
    if not is_valid_input or probs is None:
        return _invalid_result("Input image is invalid, unreadable, or corrupted.")

    p = np.asarray(probs, dtype=np.float64).flatten()

    if len(p) != _NUM_CLASSES:
        return _invalid_result(f"Expected {_NUM_CLASSES} class probabilities, got {len(p)}.")
    if not np.all(np.isfinite(p)):
        return _invalid_result("Probability array contains NaN or Inf values.")
    if np.any(p < 0) or np.any(p > 1 + 1e-5):
        return _invalid_result("Probability values out of [0, 1] range.")

    # Normalise to sum = 1
    p = p / p.sum()

    mpox_prob     = float(p[MPOX_CLASS_INDEX])
    non_mpox_prob = float(p[NON_MPOX_CLASS_INDEX])
    confidence    = float(np.max(p))
    pred_idx      = int(np.argmax(p))
    pred_class    = "MPOX" if pred_idx == MPOX_CLASS_INDEX else "NON_MPOX"

    entropy       = calculate_predictive_entropy(p)
    max_entropy   = math.log(_NUM_CLASSES)  # ln(2) ≈ 0.693
    norm_entropy  = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Decision triage
    if confidence < threshold or norm_entropy > 0.85:
        state = STATE_INCONCLUSIVE
        review = True
        reason = (
            f"Prediction confidence ({confidence*100:.1f}%) is below calibrated threshold ({threshold*100:.0f}%) "
            f"or predictive entropy ({norm_entropy:.2f}) indicates diagnostic ambiguity. "
            "Lesion appearance resembles confounding look-alikes. Referral for formal clinical evaluation and PCR molecular testing is strongly recommended."
        )
    else:
        review = False
        if pred_idx == MPOX_CLASS_INDEX:
            state = STATE_MPOX_SUSPECTED
            reason = (
                f"Model score ({mpox_prob*100:.1f}%) meets threshold for Mpox suspicion with {confidence*100:.1f}% confidence. "
                "Prompt clinical review, isolation precautions, and PCR confirmation recommended."
            )
        else:
            state = STATE_MPOX_NOT_DETECTED
            reason = (
                f"Model score ({mpox_prob*100:.1f}%) does not meet Mpox suspicion threshold ({confidence*100:.1f}% confidence non-Mpox). "
                "Does not rule out infection if clinical suspicion persists."
            )

    return {
        "decision_state":         state,
        "predicted_class":        pred_class,
        "predicted_index":        pred_idx,
        "mpox_probability":       mpox_prob,
        "non_mpox_probability":   non_mpox_prob,
        "calibrated_model_score": mpox_prob,
        "confidence":             confidence,
        "predictive_entropy":     entropy,
        "normalised_entropy":     norm_entropy,
        "review_recommended":     review,
        "recommendation_reason":  reason,
    }


def _invalid_result(reason: str) -> dict:
    return {
        "decision_state":         STATE_INVALID_IMAGE,
        "predicted_class":        "UNKNOWN",
        "predicted_index":        -1,
        "mpox_probability":       0.0,
        "non_mpox_probability":   0.0,
        "calibrated_model_score": 0.0,
        "confidence":             0.0,
        "predictive_entropy":     float(math.log(_NUM_CLASSES)),
        "normalised_entropy":     1.0,
        "review_recommended":     True,
        "recommendation_reason":  reason,
    }
