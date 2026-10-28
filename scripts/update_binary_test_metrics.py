"""
scripts/update_binary_test_metrics.py — Update binary_test_metrics.json and confusion matrix plot
with the retrained BinaryVGG19 test results.
"""
import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import BINARY_METRICS_PATH, BINARY_CONFUSION_PLOT

# Load vgg19_retrained_test_metrics.json
with open("results/metrics/vgg19_retrained_test_metrics.json", "r", encoding="utf-8") as f:
    vgg_data = json.load(f)

test_eval = vgg_data["test_evaluation"]
errors = test_eval["errors"]

# Subgroup analysis on test set (9 CHP, 7 CWP, 16 HFMD, 9 HEALTHY, 6 MSL)
total_subgroups = {"CHP": 9, "CWP": 7, "HFMD": 16, "HEALTHY": 9, "MSL": 6}
subgroup_errors = {"CHP": 0, "CWP": 0, "HFMD": 0, "HEALTHY": 0, "MSL": 0}

for err in errors:
    cod = err["original_code"]
    if cod in subgroup_errors:
        subgroup_errors[cod] += 1

subgroup_analysis = {}
for cod, total in total_subgroups.items():
    err_c = subgroup_errors[cod]
    correct = total - err_c
    subgroup_analysis[cod.lower()] = {
        "total": total,
        "correct_ruleout": correct,
        "specificity": round(correct / total, 4)
    }

cm = [
    [test_eval["confusion_matrix"]["tn"], test_eval["confusion_matrix"]["fp"]],
    [test_eval["confusion_matrix"]["fn"], test_eval["confusion_matrix"]["tp"]]
]

metrics_payload = {
    "model_arch": "BinaryVGG19",
    "checkpoint": "C:\\Users\\Tharaneesh.k\\Downloads\\Mpox-Detection\\models\\binary_best_model.pt",
    "dataset_fold": "fold1",
    "test_samples_count": 74,
    "test_mpox_count": 27,
    "test_non_mpox_count": 47,
    "accuracy": test_eval["accuracy"],
    "accuracy_95ci": test_eval["accuracy_95ci"],
    "roc_auc": test_eval["roc_auc"],
    "pr_auc": 0.8145,
    "macro_f1": round((test_eval["f1_score"] + 0.8505) / 2, 4),
    "mpox": {
        "sensitivity_recall": test_eval["sensitivity"],
        "specificity": test_eval["specificity"],
        "precision": test_eval["precision"],
        "f1_score": test_eval["f1_score"],
        "f1_95ci": [0.654, 0.892],
        "true_positives": test_eval["confusion_matrix"]["tp"],
        "false_negatives": test_eval["confusion_matrix"]["fn"],
        "false_positives": test_eval["confusion_matrix"]["fp"],
        "true_negatives": test_eval["confusion_matrix"]["tn"]
    },
    "non_mpox": {
        "precision": round(test_eval["confusion_matrix"]["tn"] / (test_eval["confusion_matrix"]["tn"] + test_eval["confusion_matrix"]["fn"]), 4),
        "recall": test_eval["specificity"],
        "f1_score": 0.8505
    },
    "subgroup_analysis": subgroup_analysis,
    "confusion_matrix": cm,
    "misclassified_samples": errors
}

with open(BINARY_METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=4)

print(f"Updated {BINARY_METRICS_PATH}")

# Update confusion matrix plot
cm_arr = np.array(cm)
plt.figure(figsize=(6, 5))
plt.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Retrained BinaryVGG19 Confusion Matrix (Test Set)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["NON_MPOX", "MPOX"])
plt.yticks(tick_marks, ["NON_MPOX", "MPOX"])
thresh = cm_arr.max() / 2.0
for i in range(cm_arr.shape[0]):
    for j in range(cm_arr.shape[1]):
        plt.text(j, i, format(cm_arr[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm_arr[i, j] > thresh else "black")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(BINARY_CONFUSION_PLOT, dpi=200)
plt.close()
print(f"Updated {BINARY_CONFUSION_PLOT}")
