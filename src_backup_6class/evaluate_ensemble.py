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
    METRICS_DIR,
    PLOTS_DIR
)
from src.dataset import get_data_loaders
from src.model import load_model, load_resnet_model, load_ensemble_model


def evaluate_all_models(fold_name: str = DEFAULT_FOLD, device_str: Optional[str] = None):
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating models on fold {fold_name} using device {device}...")

    _, _, test_loader, _ = get_data_loaders(fold_name=fold_name, batch_size=16)

    vgg_model = load_model(BEST_MODEL_PATH, device=device)
    resnet_model = load_resnet_model(device=device)
    ensemble_model = load_ensemble_model(device=device)

    models_dict = {
        "vgg19": vgg_model,
        "resnet34": resnet_model,
        "ensemble": ensemble_model
    }

    results = {}
    for name, model in models_dict.items():
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, lbls, _ in test_loader:
                imgs = imgs.to(device)
                probs = model.predict_proba(imgs)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(lbls.numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(CLASS_CODES))), zero_division=0
        )
        macro_f1 = float(np.mean(f1))
        macro_prec = float(np.mean(precision))
        macro_rec = float(np.mean(recall))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_CODES))))
        mkp_idx = TARGET_CLASS_INDEX
        tp = int(cm[mkp_idx, mkp_idx])
        fn = int(np.sum(cm[mkp_idx, :]) - tp)
        fp = int(np.sum(cm[:, mkp_idx]) - tp)
        tn = int(np.sum(cm) - (tp + fn + fp))

        mkp_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        mkp_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class = {}
        for i, code in enumerate(CLASS_CODES):
            per_class[code] = {
                "name": CODE_TO_NAME[code],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }

        target_names = [f"{c} ({CODE_TO_NAME[c]})" for c in CLASS_CODES]
        rep_text = classification_report(
            y_true, y_pred, labels=list(range(len(CLASS_CODES))), target_names=target_names, digits=4, zero_division=0
        )

        res = {
            "model_name": name,
            "dataset_fold": fold_name,
            "test_samples_count": len(y_true),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "mpox_target_evaluation": {
                "target_code": TARGET_CODE,
                "target_name": CODE_TO_NAME[TARGET_CODE],
                "sensitivity_recall": mkp_sens,
                "specificity": mkp_spec,
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "true_negatives": tn
            },
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report_text": rep_text
        }
        results[name] = res

        save_p = METRICS_DIR / f"{name}_test_metrics.json"
        with open(save_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)
        print(f"Saved {name} metrics to {save_p}")
        print(f"[{name.upper()}] Acc: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}% | MKP Sens: {mkp_sens*100:.2f}% | MKP Spec: {mkp_spec*100:.2f}%")

    # Plot comparison confusion matrix for ensemble
    ens_cm = np.array(results["ensemble"]["confusion_matrix"])
    cm_path = PLOTS_DIR / "ensemble_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        ens_cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_CODES, yticklabels=CLASS_CODES, cbar=True, ax=ax, linewidths=0.5, linecolor="gray"
    )
    ax.set_title("Ensemble (VGG19 + ResNet34) Confusion Matrix", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, labelpad=8)
    ax.set_ylabel("True Ground Truth Diagnosis", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved ensemble confusion matrix to {cm_path}")

    return results


if __name__ == "__main__":
    evaluate_all_models()
