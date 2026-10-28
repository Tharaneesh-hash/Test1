"""
src/evaluate_ensemble.py — Binary Mpox Detection
--------------------------------------------------
Optional ensemble evaluation (VGG19 + ResNet34 soft-voting).

Ensemble weights are tuned on VALIDATION data only.
Final test-set evaluation is run ONCE after weight selection.

Usage:
    python -m src.evaluate_ensemble [--fold fold1]
"""

import json
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)

from src.config import (
    BINARY_VGG19_PATH,
    BINARY_RESNET34_PATH,
    BINARY_EFFNET_PATH,
    BEST_BINARY_MODEL_PATH,
    BEST_MODEL_NAME_FILE,
    MPOX_CLASS_INDEX,
    BINARY_CLASS_NAMES,
    DEFAULT_FOLD,
    ORIGINAL_CODE_TO_NAME,
    METRICS_DIR,
    PLOTS_DIR,
)
from src.dataset import get_data_loaders
from src.model import (
    BinaryVGG19, BinaryResNet34, BinaryEfficientNetB0,
    BinaryEnsemble, load_binary_model, save_binary_model,
)
from src.evaluate import bootstrap_ci, plot_confusion_matrix, plot_roc_curve


def _collect_probs(model, loader, device):
    """Run inference and collect (labels, mpox_probs, orig_codes)."""
    model.eval()
    all_labels, all_probs, all_codes = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            labels = batch[1]
            orig_codes = batch[2]
            probs = model.predict_proba(imgs)
            all_probs.extend(probs[:, MPOX_CLASS_INDEX].cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_codes.extend(list(orig_codes))
    return np.array(all_labels), np.array(all_probs), all_codes


def tune_ensemble_weights_on_validation(
    model_a, model_b,
    valid_loader,
    device,
    weight_steps: int = 11,
) -> float:
    """
    Grid-search ensemble blending weight (w_a) on VALIDATION set.
    Returns best w_a for model_a (w_b = 1 - w_a).
    Metric: validation MPOX F1.
    """
    labels_a, probs_a, _ = _collect_probs(model_a, valid_loader, device)
    labels_b, probs_b, _ = _collect_probs(model_b, valid_loader, device)

    assert np.array_equal(labels_a, labels_b), "Validation label mismatch!"

    best_f1 = -1.0
    best_w  = 0.5
    for w in np.linspace(0, 1, weight_steps):
        blended = w * probs_a + (1 - w) * probs_b
        preds   = (blended >= 0.5).astype(int)
        f1      = f1_score(labels_a, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w  = float(w)

    print(f"  Best ensemble weight (val): w_a={best_w:.2f}, w_b={1-best_w:.2f}  -> val F1={best_f1:.4f}")
    return best_w


def evaluate_ensemble(
    fold_name: str = DEFAULT_FOLD,
    device_str: str = None,
) -> dict:
    """
    1. Load available binary checkpoints.
    2. Tune blend weights on validation only.
    3. Compare ensemble vs best individual on validation.
    4. Evaluate on test set ONCE if ensemble is justified.
    """
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("  Binary Mpox Ensemble Evaluation")
    print(f"  Fold: {fold_name} | Device: {device}")
    print("=" * 60)

    # ── 1. Load available models ───────────────────────────────────────────
    available = {}
    for name, path in [
        ("VGG19",          BINARY_VGG19_PATH),
        ("ResNet34",       BINARY_RESNET34_PATH),
        ("EfficientNetB0", BINARY_EFFNET_PATH),
    ]:
        if path.exists():
            try:
                available[name] = load_binary_model(path, device=device)
                print(f"  Loaded {name} from {path}")
            except Exception as e:
                print(f"  WARNING: Could not load {name}: {e}")
        else:
            print(f"  SKIP {name} — checkpoint not found: {path}")

    if len(available) < 2:
        print("  Need at least 2 models for ensemble. Skipping.")
        return {"skipped": True, "reason": "Fewer than 2 model checkpoints available"}

    # ── 2. Tune weights on validation ─────────────────────────────────────
    train_loader, valid_loader, test_loader, _ = get_data_loaders(
        fold_name, batch_size=16, verbose=False
    )

    model_names = list(available.keys())
    best_val_f1      = -1.0
    best_pair        = None
    best_weights     = (0.5, 0.5)

    for (na, nb) in [
        pair for i, pair in enumerate(product(model_names, model_names)) if i < len(model_names) ** 2
        if na != nb
        for na, nb in [(pair[0], pair[1])]
    ] if False else [
        (model_names[i], model_names[j])
        for i in range(len(model_names))
        for j in range(i + 1, len(model_names))
    ]:
        print(f"\n  Testing ensemble: {na} + {nb}")
        w_a = tune_ensemble_weights_on_validation(
            available[na], available[nb], valid_loader, device
        )
        w_b = 1.0 - w_a

        # Val F1 at best w
        labels_a, probs_a, _ = _collect_probs(available[na], valid_loader, device)
        labels_b, probs_b, _ = _collect_probs(available[nb], valid_loader, device)
        blended = w_a * probs_a + w_b * probs_b
        preds   = (blended >= 0.5).astype(int)
        val_f1  = f1_score(labels_a, preds, pos_label=1, zero_division=0)
        val_acc = accuracy_score(labels_a, preds)
        print(f"  Ensemble {na}x{w_a:.2f} + {nb}x{w_b:.2f}  -> val F1={val_f1:.4f}  val acc={val_acc*100:.1f}%")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_pair    = (na, nb)
            best_weights = (w_a, w_b)

    print(f"\n  Best ensemble pair: {best_pair}  w={best_weights}  val F1={best_val_f1:.4f}")

    # ── 3. Read individual val F1 for comparison ──────────────────────────
    comparison_path = METRICS_DIR / "val_model_comparison.json"
    best_individual_f1 = 0.0
    best_individual_name = "Unknown"
    if comparison_path.exists():
        with open(comparison_path) as f:
            comp = json.load(f)
        for r in comp.get("validation_comparison", []):
            if r["best_val_f1_mpox"] > best_individual_f1:
                best_individual_f1   = r["best_val_f1_mpox"]
                best_individual_name = r["arch"]
        print(f"\n  Best individual model (val): {best_individual_name}  F1={best_individual_f1:.4f}")

    improved = best_val_f1 > best_individual_f1 + 0.005  # ensemble must be meaningfully better
    print(f"  Ensemble justified: {improved}")

    results = {
        "best_ensemble_pair":     list(best_pair) if best_pair else None,
        "best_ensemble_weights":  list(best_weights),
        "ensemble_val_f1":        best_val_f1,
        "best_individual_val_f1": best_individual_f1,
        "best_individual_name":   best_individual_name,
        "ensemble_justified":     improved,
    }

    if not improved:
        print("\n  Ensemble does NOT outperform best individual model by >0.5 F1 pp.")
        print("  Retaining single best model. No test-set evaluation needed for ensemble.")
        summary_path = METRICS_DIR / "ensemble_comparison.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Comparison saved -> {summary_path}")
        return results

    # ── 4. Evaluate ensemble on test set ─────────────────────────────────
    print("\n  Ensemble outperforms individual — evaluating on test set ONCE.")
    na, nb = best_pair
    wa, wb = best_weights

    labels_va, probs_va, _ = _collect_probs(available[na], test_loader, device)
    labels_vb, probs_vb, _ = _collect_probs(available[nb], test_loader, device)
    blended_test  = wa * probs_va + wb * probs_vb
    preds_test    = (blended_test >= 0.5).astype(int)
    y_true        = labels_va

    acc      = float(accuracy_score(y_true, preds_test))
    prec_mpx = float(precision_score(y_true, preds_test, pos_label=1, zero_division=0))
    rec_mpx  = float(recall_score(y_true, preds_test,    pos_label=1, zero_division=0))
    f1_mpx   = float(f1_score(y_true, preds_test,        pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, preds_test, average="macro",    zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, blended_test))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, preds_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f"\n  Ensemble Test Results:")
    print(f"    Accuracy:    {acc*100:.2f}%")
    print(f"    MPOX Recall: {rec_mpx*100:.2f}%")
    print(f"    Specificity: {specificity*100:.2f}%")
    print(f"    Precision:   {prec_mpx*100:.2f}%")
    print(f"    F1 (MPOX):   {f1_mpx*100:.2f}%")
    print(f"    Macro F1:    {macro_f1*100:.2f}%")
    print(f"    AUC:         {auc:.4f}")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")

    results.update({
        "test_accuracy":    acc,
        "test_mpox_f1":     f1_mpx,
        "test_macro_f1":    macro_f1,
        "test_auc":         auc,
        "test_sensitivity": rec_mpx,
        "test_specificity": specificity,
        "test_precision":   prec_mpx,
        "test_confusion_matrix": cm.tolist(),
    })

    ensemble_metrics_path = METRICS_DIR / "binary_ensemble_test_metrics.json"
    with open(ensemble_metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Ensemble test metrics saved -> {ensemble_metrics_path}")

    # Plots
    cm_path  = PLOTS_DIR / "binary_ensemble_confusion_matrix.png"
    roc_path = PLOTS_DIR / "binary_ensemble_roc_curve.png"
    plot_confusion_matrix(cm, cm_path, f"Ensemble ({na}+{nb}) - Binary Mpox")
    plot_roc_curve(y_true, blended_test, auc, roc_path, f"Ensemble ({na}x{wa:.2f}+{nb}x{wb:.2f})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Mpox ensemble evaluation")
    parser.add_argument("--fold", type=str, default=DEFAULT_FOLD)
    args = parser.parse_args()
    evaluate_ensemble(fold_name=args.fold)
