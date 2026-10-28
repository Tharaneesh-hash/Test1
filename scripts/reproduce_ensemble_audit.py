"""
scripts/reproduce_ensemble_audit.py — Audit and reproduce the ensemble evaluation on fold1/test.
Identifies all 12 ensemble errors and audits exact predictions.
"""
import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.config import (
    DEFAULT_FOLD,
    BINARY_VGG19_PATH,
    BINARY_RESNET34_PATH,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    ORIGINAL_CODE_TO_NAME,
    RESULTS_DIR,
)
from src.dataset import get_data_loaders
from src.model import load_binary_model

device = torch.device("cpu")
model_vgg = load_binary_model(BINARY_VGG19_PATH, device=device)
model_res = load_binary_model(BINARY_RESNET34_PATH, device=device)
model_vgg.eval()
model_res.eval()

_, _, test_loader, _ = get_data_loaders(DEFAULT_FOLD, batch_size=1, verbose=False)

w_vgg = 0.60
w_res = 0.40

records = []
errors = []

with torch.no_grad():
    for batch in test_loader:
        img = batch[0].to(device)
        label = int(batch[1][0])
        orig_code = str(batch[2][0])
        fpath = Path(batch[3][0])
        fname = fpath.name

        p_vgg = float(model_vgg.predict_proba(img)[0, MPOX_CLASS_INDEX])
        p_res = float(model_res.predict_proba(img)[0, MPOX_CLASS_INDEX])
        p_ens = w_vgg * p_vgg + w_res * p_res
        pred = int(p_ens >= 0.5)

        is_error = (pred != label)
        err_type = "Correct"
        if is_error:
            err_type = "False Negative (FN)" if (label == 1 and pred == 0) else "False Positive (FP)"

        rec = {
            "filename": fname,
            "filepath": str(fpath),
            "original_code": orig_code,
            "condition_name": ORIGINAL_CODE_TO_NAME.get(orig_code, orig_code),
            "true_binary_label": "MPOX" if label == 1 else "NON_MPOX",
            "pred_binary_label": "MPOX" if pred == 1 else "NON_MPOX",
            "p_vgg": round(p_vgg, 4),
            "p_res": round(p_res, 4),
            "p_ens": round(p_ens, 4),
            "error_type": err_type,
            "is_error": is_error,
        }
        records.append(rec)
        if is_error:
            errors.append(rec)

y_true = np.array([r["true_binary_label"] == "MPOX" for r in records], dtype=int)
y_pred = np.array([r["pred_binary_label"] == "MPOX" for r in records], dtype=int)
y_prob = np.array([r["p_ens"] for r in records], dtype=float)

acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp)

print("=" * 70)
print("ENSEMBLE AUDIT & REPRODUCIBILITY VERIFICATION")
print("=" * 70)
print(f"Total Test Samples: {len(records)}")
print(f"Accuracy:           {acc:.4f} ({int(tp+tn)}/{len(records)}) = {acc*100:.2f}%")
print(f"Sensitivity:        {rec:.4f} ({tp}/{tp+fn}) = {rec*100:.2f}%")
print(f"Specificity:        {spec:.4f} ({tn}/{tn+fp}) = {spec*100:.2f}%")
print(f"Precision:          {prec:.4f}")
print(f"F1-Score:           {f1:.4f}")
print(f"ROC-AUC:            {auc:.4f}")
print(f"Confusion Matrix:   TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Total Errors:       {len(errors)} (FN={fn}, FP={fp})")

print("\n" + "=" * 70)
print(f"DETAILED BREAKDOWN OF ALL {len(errors)} ENSEMBLE ERRORS")
print("=" * 70)
for i, err in enumerate(errors, 1):
    print(f"{i:02d}. [{err['error_type']}] {err['filename']} ({err['condition_name']}) | True: {err['true_binary_label']} | Pred: {err['pred_binary_label']} | P(Ens): {err['p_ens']:.2%} (VGG: {err['p_vgg']:.2%}, ResNet: {err['p_res']:.2%})")

# Save detailed report
out_path = RESULTS_DIR / "metrics" / "ensemble_12_errors_audit.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "reproduced_metrics": {
            "accuracy": round(float(acc), 4),
            "sensitivity": round(float(rec), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            "total_errors": len(errors)
        },
        "errors": errors,
        "all_records": records
    }, f, indent=2)

print(f"\nSaved error audit -> {out_path}")
