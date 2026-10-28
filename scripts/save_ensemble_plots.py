"""
scripts/save_ensemble_plots.py — Save plots for the retrained ensemble.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

metrics_path = Path("results/metrics/binary_ensemble_test_metrics.json")
with open(metrics_path, "r", encoding="utf-8") as f:
    data = json.load(f)

cm = np.array(data["test_confusion_matrix"])
class_names = ["NON_MPOX", "MPOX"]

# Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Retrained Ensemble (VGG19 + ResNet34) Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
cm_save = Path("results/plots/binary_ensemble_confusion_matrix.png")
plt.savefig(cm_save, dpi=200)
plt.close()
print("Saved ensemble confusion matrix ->", cm_save)
