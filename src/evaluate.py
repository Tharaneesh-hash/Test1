import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.config import (
    BEST_MODEL_PATH,
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME,
    TARGET_CODE,
    TARGET_CLASS_INDEX,
    DEFAULT_FOLD,
    TEST_METRICS_PATH,
    CONFUSION_MATRIX_PLOT
)
from src.dataset import get_data_loaders
from src.model import load_model, VGG19MpoxClassifier


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: Path,
    title: str = "Confusion Matrix - Mpox & Differential Diagnoses"
):
    """
    Renders and saves a high-resolution styled confusion matrix heatmap.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )

    ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth Diagnosis", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")


def evaluate_model(
    model: Optional[VGG19MpoxClassifier] = None,
    checkpoint_path: Path = BEST_MODEL_PATH,
    fold_name: str = DEFAULT_FOLD,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates the trained VGG19 model strictly on the untouched test set.
    Calculates:
    - Global accuracy
    - Per-class precision, recall, F1-score, and support
    - Macro and weighted F1-score
    - Binary Mpox sensitivity (recall) and specificity against all differential classes
    - Full confusion matrix
    - Scikit-learn classification report
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if model is None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = load_model(checkpoint_path=checkpoint_path, device=device)
    else:
        model.to(device)
        model.eval()

    # Load test DataLoader (untouched split)
    _, _, test_loader, _ = get_data_loaders(fold_name=fold_name, batch_size=16)

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    print(f"Running inference on untouched test set ({fold_name}/test)...")
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(paths)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 1. Global Metrics
    acc = float(accuracy_score(y_true, y_pred))

    # 2. Per-Class and Averaged Metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_CODES))), zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_CODES))))

    # 4. Mpox Target Class (MKP) Specificity and Sensitivity
    # Binary formulation: MKP (Positive) vs non-MKP (Negative)
    mkp_idx = TARGET_CLASS_INDEX
    tp = int(cm[mkp_idx, mkp_idx])
    fn = int(np.sum(cm[mkp_idx, :]) - tp)
    fp = int(np.sum(cm[:, mkp_idx]) - tp)
    tn = int(np.sum(cm) - (tp + fn + fp))

    mpox_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    mpox_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 5. Per-Class Performance Dictionary
    per_class_metrics = {}
    for i, code in enumerate(CLASS_CODES):
        per_class_metrics[code] = {
            "name": CODE_TO_NAME[code],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }

    # 6. Detailed Text Classification Report
    target_names = [f"{c} ({CODE_TO_NAME[c]})" for c in CLASS_CODES]
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_CODES))),
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    evaluation_results = {
        "dataset_fold": fold_name,
        "test_samples_count": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "mpox_target_evaluation": {
            "target_code": TARGET_CODE,
            "target_name": CODE_TO_NAME[TARGET_CODE],
            "sensitivity_recall": mpox_sensitivity,
            "specificity": mpox_specificity,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_text,
        "clinical_disclaimer": "Experimental medical-research prototype. Not clinically validated for patient diagnosis."
    }

    # Save to disk
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Test evaluation metrics saved to {TEST_METRICS_PATH}")

    # Plot Confusion Matrix
    plot_confusion_matrix(cm, CLASS_CODES, CONFUSION_MATRIX_PLOT)

    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION REPORT")
    print("="*50)
    print(report_text)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Macro F1-Score: {macro_f1*100:.2f}%")
    print(f"Mpox Sensitivity: {mpox_sensitivity*100:.2f}% | Specificity: {mpox_specificity*100:.2f}%")
    print("="*50)

    return evaluation_results


if __name__ == "__main__":
    evaluate_model()
