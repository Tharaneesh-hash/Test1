"""
src/evaluate.py — Mpox Detection Benchmark Evaluation Script.

Evaluates the selected model on the UNTOUCHED test set.
Computes:
  - Overall accuracy & 95% bootstrap confidence intervals
  - Sensitivity / Recall, Specificity, Precision, F1-Score for MPOX
  - Precision, Recall, F1-Score for NON_MPOX
  - ROC-AUC and PR-AUC (Precision-Recall AUC)
  - Confusion Matrix (TP, TN, FP, FN)
  - Detailed subgroup error attribution against look-alikes (Chickenpox, Cowpox)
  - Selective classification evaluation (Coverage vs Accuracy Tradeoff)
  - Generates confusion matrix, ROC curve, and PR curve plots
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

from src.config import (
    BEST_BINARY_MODEL_PATH,
    BEST_MODEL_NAME_FILE,
    BINARY_METRICS_PATH,
    BINARY_CONFUSION_PLOT,
    BINARY_ROC_CURVE,
    BINARY_PR_CURVE,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    BINARY_CLASS_NAMES,
    DEFAULT_FOLD,
    ORIGINAL_CODE_TO_NAME,
)
from src.dataset import get_data_loaders
from src.model import load_binary_model, MultimodalMpoxClassifier
from src.conformal import evaluate_selective_classification


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=500, ci=0.95):
    """Non-parametric bootstrap confidence interval for a metric."""
    n = len(y_true)
    scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            s = 0.0
        scores.append(s)
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["NON_MPOX", "MPOX"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor="gray", cbar=True
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11)
    ax.set_ylabel("True Diagnosis", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


def plot_roc_curve(y_true, y_scores, auc: float, save_path: Path, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(f"{model_name} — ROC Curve (Mpox Screening)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  ROC curve saved -> {save_path}")


def plot_pr_curve(y_true, y_scores, pr_auc: float, save_path: Path, model_name: str):
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="darkorange", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title(f"{model_name} — Precision-Recall Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  PR curve saved -> {save_path}")


def evaluate_binary_model(
    checkpoint_path: Path = BEST_BINARY_MODEL_PATH,
    fold_name: str = DEFAULT_FOLD,
    device_str: Optional[str] = None,
    save_metrics: bool = True,
) -> Dict[str, Any]:
    """
    Full binary evaluation on the UNTOUCHED test set.
    """
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*60}")
    print(f"Mpox Screening Evaluation | Fold: {fold_name} | Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    model = load_binary_model(checkpoint_path, device=device)
    model.eval()

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        arch_name = ckpt.get("arch", "Unknown") if isinstance(ckpt, dict) else "Unknown"
    except Exception:
        arch_name = "Unknown"

    _, _, test_loader, _ = get_data_loaders(fold_name, batch_size=16, verbose=False)

    all_preds, all_labels, all_probs = [], [], []
    all_orig_codes, all_paths = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(device)
            labels = batch[1]
            orig_codes = batch[2]
            paths = batch[3]

            if isinstance(model, MultimodalMpoxClassifier) and len(batch) >= 5:
                clin_data = batch[4].to(device)
                probs = model.predict_proba(imgs, clin_data)
            else:
                probs = model.predict_proba(imgs)

            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs[:, MPOX_CLASS_INDEX].cpu().numpy().tolist())
            all_orig_codes.extend(list(orig_codes))
            all_paths.extend(list(paths))

    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_scores = np.array(all_probs)

    # Core Metrics
    acc      = float(accuracy_score(y_true, y_pred))
    prec_mpx = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_mpx  = float(recall_score(y_true, y_pred,    pos_label=1, zero_division=0))  # Sensitivity
    f1_mpx   = float(f1_score(y_true, y_pred,        pos_label=1, zero_division=0))
    prec_non = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_non  = float(recall_score(y_true, y_pred,    pos_label=0, zero_division=0))
    f1_non   = float(f1_score(y_true, y_pred,        pos_label=0, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except Exception:
        pr_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Bootstrap Confidence Intervals (95%)
    acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score)
    f1_lo, f1_hi   = bootstrap_ci(y_true, y_pred, lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0))

    # Look-Alike Subgroup Analysis
    chp_mask = np.array([c == "CHP" for c in all_orig_codes])
    cwp_mask = np.array([c == "CWP" for c in all_orig_codes])

    chp_total = int(np.sum(chp_mask))
    chp_correct_ruleout = int(np.sum((y_pred == 0) & chp_mask))
    chp_spec = float(chp_correct_ruleout / chp_total) if chp_total > 0 else 0.0

    cwp_total = int(np.sum(cwp_mask))
    cwp_correct_ruleout = int(np.sum((y_pred == 0) & cwp_mask))
    cwp_spec = float(cwp_correct_ruleout / cwp_total) if cwp_total > 0 else 0.0

    # Granular Error Attribution
    fn_orig = Counter()
    fp_orig = Counter()
    for i, (true, pred) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        if true == 1 and pred == 0:
            fn_orig[all_orig_codes[i]] += 1
        elif true == 0 and pred == 1:
            fp_orig[all_orig_codes[i]] += 1

    # Print Report
    print(f"\n{'─'*50}")
    print(f"  MODEL: {arch_name}")
    print(f"  Test Samples:    {len(y_true)}  (MPOX={int(y_true.sum())}, NON_MPOX={int((y_true==0).sum())})")
    print(f"{'─'*50}")
    print(f"  Overall Accuracy: {acc*100:.2f}%  (95% CI: [{acc_lo*100:.1f}%, {acc_hi*100:.1f}%])")
    print(f"  ROC-AUC:          {auc:.4f}")
    print(f"  PR-AUC:           {pr_auc:.4f}")
    print(f"  Macro F1:         {macro_f1*100:.2f}%")
    print()
    print(f"  MPOX Target Class (Positive):")
    print(f"    Sensitivity (Recall):  {rec_mpx*100:.2f}%  ({tp}/{int(y_true.sum())} detected)")
    print(f"    Specificity:           {specificity*100:.2f}%  ({tn}/{int((y_true==0).sum())} non-mpox ruled out)")
    print(f"    Precision:             {prec_mpx*100:.2f}%")
    print(f"    F1-Score:              {f1_mpx*100:.2f}%  (95% CI: [{f1_lo*100:.1f}%, {f1_hi*100:.1f}%])")
    print()
    print(f"  Look-Alike Subgroup Performance:")
    print(f"    Chickenpox (CHP) Rule-out: {chp_correct_ruleout}/{chp_total} ({chp_spec*100:.1f}%)")
    print(f"    Cowpox (CWP) Rule-out:     {cwp_correct_ruleout}/{cwp_total} ({cwp_spec*100:.1f}%)")
    print()
    print(f"  Confusion Matrix: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'─'*50}")

    # Selective classification evaluation
    probs_2d = np.stack([1.0 - y_scores, y_scores], axis=1)
    selective_summary = evaluate_selective_classification(y_true, probs_2d)

    results = {
        "model_arch": arch_name,
        "checkpoint": str(checkpoint_path),
        "dataset_fold": fold_name,
        "test_samples_count": int(len(y_true)),
        "test_mpox_count": int(y_true.sum()),
        "test_non_mpox_count": int((y_true == 0).sum()),
        "accuracy": acc,
        "accuracy_95ci": [acc_lo, acc_hi],
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "macro_f1": macro_f1,
        "mpox": {
            "sensitivity_recall": rec_mpx,
            "specificity": specificity,
            "precision": prec_mpx,
            "f1_score": f1_mpx,
            "f1_95ci": [f1_lo, f1_hi],
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
        },
        "non_mpox": {
            "precision": prec_non,
            "recall": rec_non,
            "f1_score": f1_non,
        },
        "subgroup_analysis": {
            "chickenpox": {"total": chp_total, "correct_ruleout": chp_correct_ruleout, "specificity": chp_spec},
            "cowpox":     {"total": cwp_total, "correct_ruleout": cwp_correct_ruleout, "specificity": cwp_spec},
        },
        "confusion_matrix": cm.tolist(),
        "error_analysis": {
            "false_negatives_by_original_class": dict(fn_orig),
            "false_positives_by_original_class": dict(fp_orig),
        },
        "selective_classification": selective_summary,
        "clinical_disclaimer": (
            "Academic research prototype. NOT a certified medical device. "
            "All predictions require independent clinical validation and molecular confirmation."
        ),
    }

    if save_metrics:
        BINARY_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BINARY_METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"  Metrics saved -> {BINARY_METRICS_PATH}")

        plot_confusion_matrix(cm, BINARY_CONFUSION_PLOT, title=f"Binary Mpox Detection ({arch_name})")
        plot_roc_curve(y_true, y_scores, auc, BINARY_ROC_CURVE, arch_name)
        plot_pr_curve(y_true, y_scores, pr_auc, BINARY_PR_CURVE, arch_name)

    return results


if __name__ == "__main__":
    evaluate_binary_model()
