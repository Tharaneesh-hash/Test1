"""
scripts/retrain_vgg19.py — Retrain VGG19 on the updated dataset with 287 new images,
evaluate on the independent 74-sample test set, and compare with the baseline.
"""
import sys
sys.path.insert(0, ".")
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.config import (
    DEFAULT_FOLD,
    BINARY_VGG19_PATH,
    BEST_BINARY_MODEL_PATH,
    BEST_MODEL_NAME_FILE,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    BINARY_CLASS_NAMES,
    METRICS_DIR,
    PLOTS_DIR,
)
from src.dataset import get_data_loaders
from src.model import BinaryVGG19, save_binary_model, load_binary_model
from src.train import train_binary_model

def wilson_ci(k, n, confidence=0.95):
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95%
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def evaluate_on_test(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs, all_paths, all_orig_codes = [], [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            orig_codes = batch[2]
            paths = batch[3]
            
            probs = model.predict_proba(imgs)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, MPOX_CLASS_INDEX].cpu().numpy())
            all_paths.extend(paths)
            all_orig_codes.extend(orig_codes)
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=MPOX_CLASS_INDEX, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=MPOX_CLASS_INDEX, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=MPOX_CLASS_INDEX, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 95% Confidence Intervals
    acc_ci = wilson_ci(int(tp + tn), len(y_true))
    sens_ci = wilson_ci(int(tp), int(tp + fn))
    spec_ci = wilson_ci(int(tn), int(tn + fp))
    
    # Analyze errors
    errors = []
    for p, tr, pr, sc, cod in zip(all_paths, y_true, y_pred, y_prob, all_orig_codes):
        if tr != pr:
            err_type = "False Negative (FN)" if (tr == 1 and pr == 0) else "False Positive (FP)"
            errors.append({
                "filename": Path(p).name,
                "original_code": cod,
                "true_label": "MPOX" if tr == 1 else "NON_MPOX",
                "pred_label": "MPOX" if pr == 1 else "NON_MPOX",
                "mpox_score": round(float(sc), 4),
                "error_type": err_type
            })
            
    return {
        "accuracy": round(float(acc), 4),
        "accuracy_95ci": [round(acc_ci[0], 4), round(acc_ci[1], 4)],
        "sensitivity": round(float(rec), 4),
        "sensitivity_95ci": [round(sens_ci[0], 4), round(sens_ci[1], 4)],
        "specificity": round(float(spec), 4),
        "specificity_95ci": [round(spec_ci[0], 4), round(spec_ci[1], 4)],
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "total_test_samples": len(y_true),
        "total_errors": len(errors),
        "errors": errors
    }

def main():
    print("=" * 70)
    print("  RETRAINING VGG19 ON ENRICHED DATASET (WITH 287 NEW MPOX IMAGES)")
    print("=" * 70)
    
    t0 = time.time()
    # Train BinaryVGG19
    train_results = train_binary_model(
        arch_name="BinaryVGG19",
        fold_name=DEFAULT_FOLD,
        initial_epochs=10,
        fine_tune_epochs=4,
        batch_size=16,
    )
    print(f"\nVGG19 training completed in {(time.time() - t0) / 60:.1f} minutes.")
    print(f"Best Validation MPOX F1: {train_results['best_val_f1_mpox']:.4f}")
    
    # Load the newly trained best VGG19 checkpoint
    device = torch.device("cpu")
    model = load_binary_model(BINARY_VGG19_PATH, device=device)
    
    # Evaluate on independent test set
    _, _, test_loader, _ = get_data_loaders(DEFAULT_FOLD, batch_size=1, verbose=False)
    test_eval = evaluate_on_test(model, test_loader, device)
    
    # Save test metrics
    metrics_path = METRICS_DIR / "vgg19_retrained_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "BinaryVGG19_Retrained",
            "training_results": {
                "best_val_f1_mpox": train_results["best_val_f1_mpox"],
                "best_val_acc": train_results["best_val_acc"],
                "total_epochs_trained": train_results["total_epochs_trained"]
            },
            "test_evaluation": test_eval
        }, f, indent=2)
        
    print("\n" + "=" * 70)
    print("  FINAL INDEPENDENT TEST EVALUATION (74 UNTOUCHED IMAGES)")
    print("=" * 70)
    print(f"  Test Accuracy:    {test_eval['accuracy']:.2%}  (95% CI: {test_eval['accuracy_95ci'][0]:.1%} - {test_eval['accuracy_95ci'][1]:.1%})")
    print(f"  Mpox Sensitivity: {test_eval['sensitivity']:.2%}  (95% CI: {test_eval['sensitivity_95ci'][0]:.1%} - {test_eval['sensitivity_95ci'][1]:.1%})")
    print(f"  Specificity:      {test_eval['specificity']:.2%}  (95% CI: {test_eval['specificity_95ci'][0]:.1%} - {test_eval['specificity_95ci'][1]:.1%})")
    print(f"  Precision:        {test_eval['precision']:.2%}")
    print(f"  F1-Score:         {test_eval['f1_score']:.4f}")
    print(f"  ROC-AUC:          {test_eval['roc_auc']:.4f}")
    cm = test_eval["confusion_matrix"]
    print(f"  Confusion Matrix: TP={cm['tp']}, TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}")
    print(f"  Total Errors:     {test_eval['total_errors']} / {test_eval['total_test_samples']}")
    print(f"\nSaved metrics -> {metrics_path}")

if __name__ == "__main__":
    main()
