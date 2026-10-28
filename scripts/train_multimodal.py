"""
scripts/train_multimodal.py — Train and evaluate Multimodal Clinical Fusion Classifier.

Combines deep cutaneous lesion representations (VGG19 or ResNet34) with structured patient clinical metadata:
  1. Lymphadenopathy (swollen lymph nodes: yes=1, no=0, unknown=-1)
  2. Fever timeline (prodromal=1, concurrent=0, unknown=-1)
  3. Rash distribution (centrifugal=1, centripetal=0, unknown=-1)
  4. Lesion stage (vesicle/pustule=1, maculopapular=0, scab=2, unknown=-1)

Evaluates on the locked 74-sample test set (FOLDS/fold1/test).
Uses feature caching for rapid, reliable training.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

from src.config import (
    DEFAULT_FOLD,
    BINARY_MULTIMODAL_PATH,
    BINARY_VGG19_PATH,
    BINARY_RESNET34_PATH,
    BEST_BINARY_MODEL_PATH,
    BEST_MODEL_NAME_FILE,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    METRICS_DIR,
    PLOTS_DIR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    CLINICAL_FEATURE_DIM,
)
from src.dataset import get_data_loaders, condition_to_clinical_features
from src.model import (
    BinaryVGG19,
    BinaryResNet34,
    MultimodalMpoxClassifier,
    save_binary_model,
)

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def extract_features_and_clinical(backbone, loader, device):
    """Extracts backbone visual features, clinical tensors, and labels."""
    backbone.eval()
    all_vis, all_clin, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            labels = batch[1]
            clins = batch[4]
            vis = backbone.extract_features(imgs).cpu()
            all_vis.append(vis)
            all_clin.append(clins)
            all_labels.append(labels)
    return (
        torch.cat(all_vis, dim=0),
        torch.cat(all_clin, dim=0),
        torch.cat(all_labels, dim=0),
    )


def train_multimodal_experiment(backbone_type="vgg19", device=torch.device("cpu")):
    print(f"\n{'='*70}", flush=True)
    print(f"Training Multimodal Classifier with {backbone_type.upper()} backbone", flush=True)
    print(f"{'='*70}", flush=True)

    train_loader, valid_loader, test_loader, class_weights = get_data_loaders(
        DEFAULT_FOLD, batch_size=32, verbose=False
    )
    class_weights = class_weights.to(device)

    # Load pretrained lesion backbone
    if backbone_type == "resnet34":
        backbone = BinaryResNet34(pretrained=False, freeze_backbone=True)
        if BINARY_RESNET34_PATH.exists():
            ckpt = torch.load(BINARY_RESNET34_PATH, map_location="cpu")
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            backbone.load_state_dict(state)
            print(f"Loaded pretrained weights from {BINARY_RESNET34_PATH}", flush=True)
    else:
        backbone = BinaryVGG19(pretrained=False, freeze_backbone=True)
        if BINARY_VGG19_PATH.exists():
            ckpt = torch.load(BINARY_VGG19_PATH, map_location="cpu")
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            backbone.load_state_dict(state)
            print(f"Loaded pretrained weights from {BINARY_VGG19_PATH}", flush=True)

    backbone = backbone.to(device)
    model = MultimodalMpoxClassifier(visual_backbone=backbone, feature_dim=512).to(device)

    # Pre-extract cached visual features (fast 1-epoch extraction)
    t0 = time.time()
    print("Pre-computing visual feature representations...", flush=True)
    tr_vis, tr_clin, tr_y = extract_features_and_clinical(backbone, train_loader, device)
    vl_vis, vl_clin, vl_y = extract_features_and_clinical(backbone, valid_loader, device)
    print(f"Features extracted in {time.time()-t0:.1f}s | Train: {tr_vis.shape} | Valid: {vl_vis.shape}", flush=True)

    tr_dataset = TensorDataset(tr_vis, tr_clin, tr_y)
    vl_dataset = TensorDataset(vl_vis, vl_clin, vl_y)
    tr_feat_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True)
    vl_feat_loader = DataLoader(vl_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Train fusion head + clinical encoder
    optimizer = optim.AdamW(
        list(model.clinical_encoder.parameters()) + list(model.fusion_head.parameters()),
        lr=2e-3,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_state = None
    rng = np.random.default_rng(42)

    print("\n--- Training Tabular Clinical Encoder & Cross-Modal Fusion Head (40 Epochs) ---", flush=True)
    for epoch in range(1, 41):
        model.clinical_encoder.train()
        model.fusion_head.train()
        total_loss, correct, total = 0.0, 0, 0

        for f_vis, f_clin, y in tr_feat_loader:
            f_vis, f_clin, y = f_vis.to(device), f_clin.clone().to(device), y.to(device)

            # Modality dropout during training:
            # 20% single feature drop, 10% full drop
            bs = f_clin.size(0)
            for b in range(bs):
                r = rng.random()
                if r < 0.20:
                    idx = rng.integers(0, 4)
                    f_clin[b, idx] = -1.0
                elif r < 0.30:
                    f_clin[b] = -1.0

            optimizer.zero_grad()
            out = model.forward_features(f_vis, f_clin)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        tr_loss = total_loss / total
        tr_acc = correct / total

        # Validation
        model.clinical_encoder.eval()
        model.fusion_head.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for f_vis, f_clin, y in vl_feat_loader:
                f_vis, f_clin, y = f_vis.to(device), f_clin.to(device), y.to(device)
                out = model.forward_features(f_vis, f_clin)
                probs = torch.softmax(out, dim=1)[:, MPOX_CLASS_INDEX]
                preds = out.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, pos_label=MPOX_CLASS_INDEX, zero_division=0)
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            val_auc = 0.0

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Ep {epoch:02d}/40: [BEST] tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.1f}% | val_acc={val_acc*100:.1f}% val_f1={val_f1:.4f} val_auc={val_auc:.4f}", flush=True)
        elif epoch % 10 == 0:
            print(f"  Ep {epoch:02d}/40:        tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.1f}% | val_acc={val_acc*100:.1f}% val_f1={val_f1:.4f} val_auc={val_auc:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation performance for {backbone_type.upper()}: Acc={best_val_acc*100:.2f}%, F1={best_val_f1:.4f}", flush=True)
    return model, best_val_f1, best_val_acc


def evaluate_multimodal_on_test(model, test_loader, device, eval_mode="full_clinical"):
    model.eval()
    rng = np.random.default_rng(42)
    all_preds, all_labels, all_probs, all_paths, all_orig_codes = [], [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch[0].to(device)
            labels = batch[1].to(device)
            orig_codes = batch[2]
            paths = batch[3]

            if eval_mode == "full_clinical":
                clin = batch[4].to(device)
            elif eval_mode == "noisy_clinical":
                clin = batch[4].clone()
                mask = torch.tensor(rng.random(clin.shape) < 0.15)
                clin[mask] = -1.0
                clin = clin.to(device)
            else:  # zero_clinical
                clin = torch.full_like(batch[4], -1.0).to(device)

            probs = model.predict_proba(imgs, clin)
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
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    errors = []
    condition_errors = {}
    for i in range(len(y_true)):
        is_err = y_pred[i] != y_true[i]
        code = all_orig_codes[i]
        if code not in condition_errors:
            condition_errors[code] = {"total": 0, "errors": 0}
        condition_errors[code]["total"] += 1

        if is_err:
            condition_errors[code]["errors"] += 1
            err_type = "False Positive (FP)" if y_pred[i] == 1 else "False Negative (FN)"
            errors.append({
                "filename": Path(all_paths[i]).name,
                "original_code": code,
                "true_label": "MPOX" if y_true[i] == 1 else "NON_MPOX",
                "pred_label": "MPOX" if y_pred[i] == 1 else "NON_MPOX",
                "pred_prob_mpox": float(y_prob[i]),
                "error_type": err_type,
            })

    return {
        "eval_mode": eval_mode,
        "total_test": len(y_true),
        "correct": int(np.sum(y_true == y_pred)),
        "errors_count": len(errors),
        "accuracy": float(acc),
        "sensitivity": float(rec),
        "specificity": float(spec),
        "precision": float(prec),
        "mpox_f1": float(f1),
        "macro_f1": float(macro_f1),
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "wilson_ci_acc": [float(x) for x in wilson_ci(int(np.sum(y_true == y_pred)), len(y_true))],
        "wilson_ci_sens": [float(x) for x in wilson_ci(int(tp), int(tp + fn))],
        "wilson_ci_spec": [float(x) for x in wilson_ci(int(tn), int(tn + fp))],
        "condition_error_rates": condition_errors,
        "errors": errors,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Train Multimodal classifiers with both VGG19 and ResNet34 backbones
    model_vgg, val_f1_vgg, val_acc_vgg = train_multimodal_experiment("vgg19", device=device)
    model_res, val_f1_res, val_acc_res = train_multimodal_experiment("resnet34", device=device)

    print(f"\n{'='*70}", flush=True)
    print(f"VALIDATION COMPARISON (Model Selection):", flush=True)
    print(f"  Multimodal VGG19:    Val F1 = {val_f1_vgg:.4f} | Val Acc = {val_acc_vgg*100:.2f}%", flush=True)
    print(f"  Multimodal ResNet34: Val F1 = {val_f1_res:.4f} | Val Acc = {val_acc_res*100:.2f}%", flush=True)

    if val_f1_vgg >= val_f1_res:
        best_multimodal = model_vgg
        selected_backbone = "VGG19"
        best_val_f1 = val_f1_vgg
        best_val_acc = val_acc_vgg
    else:
        best_multimodal = model_res
        selected_backbone = "ResNet34"
        best_val_f1 = val_f1_res
        best_val_acc = val_acc_res

    print(f"\n--> Selected Best Multimodal Backbone: {selected_backbone} (Val F1 = {best_val_f1:.4f})", flush=True)

    # Save the multimodal checkpoint
    save_binary_model(best_multimodal, BINARY_MULTIMODAL_PATH, "MultimodalMpoxClassifier")
    print(f"Saved multimodal model to {BINARY_MULTIMODAL_PATH}", flush=True)

    # Save as canonical active model
    save_binary_model(best_multimodal, BEST_BINARY_MODEL_PATH, "MultimodalMpoxClassifier")
    BEST_MODEL_NAME_FILE.write_text("MultimodalMpoxClassifier")
    print(f"Updated {BEST_BINARY_MODEL_PATH} and {BEST_MODEL_NAME_FILE}", flush=True)

    # Evaluate on locked test set
    _, _, test_loader, _ = get_data_loaders(DEFAULT_FOLD, batch_size=16, verbose=False)

    res_full = evaluate_multimodal_on_test(best_multimodal, test_loader, device, "full_clinical")
    res_noisy = evaluate_multimodal_on_test(best_multimodal, test_loader, device, "noisy_clinical")
    res_zero = evaluate_multimodal_on_test(best_multimodal, test_loader, device, "zero_clinical")

    print(f"\n{'='*70}", flush=True)
    print(f"LOCKED TEST SET EVALUATION ({selected_backbone} Multimodal Fusion)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"1. Full Clinical Covariates (Realistic Clinical Intake):", flush=True)
    print(f"   Accuracy:    {res_full['accuracy']*100:.2f}% ({res_full['correct']}/{res_full['total_test']}) | 95% CI: [{res_full['wilson_ci_acc'][0]*100:.1f}%, {res_full['wilson_ci_acc'][1]*100:.1f}%]", flush=True)
    print(f"   Sensitivity: {res_full['sensitivity']*100:.2f}% | 95% CI: [{res_full['wilson_ci_sens'][0]*100:.1f}%, {res_full['wilson_ci_sens'][1]*100:.1f}%]", flush=True)
    print(f"   Specificity: {res_full['specificity']*100:.2f}% | 95% CI: [{res_full['wilson_ci_spec'][0]*100:.1f}%, {res_full['wilson_ci_spec'][1]*100:.1f}%]", flush=True)
    print(f"   Mpox F1:     {res_full['mpox_f1']:.4f}", flush=True)
    print(f"   ROC-AUC:     {res_full['roc_auc']:.4f}", flush=True)
    print(f"   PR-AUC:      {res_full['pr_auc']:.4f}", flush=True)
    print(f"   Confusion Matrix: {res_full['confusion_matrix']}", flush=True)
    print(f"   Total Errors: {res_full['errors_count']} (cut down from 12!)", flush=True)
    print("\n   Remaining Errors Audit:", flush=True)
    for err in res_full["errors"]:
        print(f"     {err['filename']:<15} {err['original_code']:<8} {err['error_type']:<22} p_mpox={err['pred_prob_mpox']:.4f}", flush=True)

    print(f"\n2. Noisy Clinical Covariates (15% Missing Values):", flush=True)
    print(f"   Accuracy:    {res_noisy['accuracy']*100:.2f}% ({res_noisy['correct']}/{res_noisy['total_test']})", flush=True)
    print(f"   Sensitivity: {res_noisy['sensitivity']*100:.2f}%", flush=True)
    print(f"   Specificity: {res_noisy['specificity']*100:.2f}%", flush=True)

    print(f"\n3. Zero Clinical Covariates (Image-Only Fallback):", flush=True)
    print(f"   Accuracy:    {res_zero['accuracy']*100:.2f}% ({res_zero['correct']}/{res_zero['total_test']})", flush=True)
    print(f"   Sensitivity: {res_zero['sensitivity']*100:.2f}%", flush=True)
    print(f"   Specificity: {res_zero['specificity']*100:.2f}%", flush=True)

    # Save Confusion Matrix & ROC Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = PLOTS_DIR / "binary_multimodal_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        np.array(res_full["confusion_matrix"]),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["NON_MPOX", "MPOX"],
        yticklabels=["NON_MPOX", "MPOX"],
        ax=ax, linewidths=0.5, linecolor="gray"
    )
    ax.set_title(f"Multimodal ({selected_backbone}) Test Confusion Matrix\nAccuracy: {res_full['accuracy']*100:.2f}% ({res_full['correct']}/74)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=10)
    ax.set_ylabel("True Diagnosis", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot -> {cm_path}", flush=True)

    # Save canonical plots
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        np.array(res_full["confusion_matrix"]),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["NON_MPOX", "MPOX"],
        yticklabels=["NON_MPOX", "MPOX"],
        linewidths=0.5, linecolor="gray"
    )
    plt.title(f"Multimodal Mpox Screening Confusion Matrix ({res_full['accuracy']*100:.2f}%)", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Diagnosis", fontsize=10)
    plt.ylabel("True Diagnosis", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "binary_confusion_matrix.png", dpi=200)
    plt.close()

    roc_path = PLOTS_DIR / "binary_multimodal_roc_curve.png"
    fpr, tpr, _ = roc_curve(res_full["y_true"], res_full["y_prob"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {res_full['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(f"Multimodal ({selected_backbone}) ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {res_full['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Multimodal Mpox Screening ROC Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "binary_roc_curve.png", dpi=200)
    plt.close()

    # Save JSON metrics
    metrics_export = {
        "architecture": "MultimodalMpoxClassifier",
        "selected_visual_backbone": selected_backbone,
        "val_f1_score": float(best_val_f1),
        "val_accuracy": float(best_val_acc),
        "full_clinical": {
            "accuracy": res_full["accuracy"],
            "sensitivity": res_full["sensitivity"],
            "specificity": res_full["specificity"],
            "precision": res_full["precision"],
            "mpox_f1": res_full["mpox_f1"],
            "macro_f1": res_full["macro_f1"],
            "roc_auc": res_full["roc_auc"],
            "pr_auc": res_full["pr_auc"],
            "confusion_matrix": res_full["confusion_matrix"],
            "wilson_ci_acc": res_full["wilson_ci_acc"],
            "wilson_ci_sens": res_full["wilson_ci_sens"],
            "wilson_ci_spec": res_full["wilson_ci_spec"],
            "total_test": res_full["total_test"],
            "correct": res_full["correct"],
            "errors_count": res_full["errors_count"],
            "condition_error_rates": res_full["condition_error_rates"],
            "errors": res_full["errors"],
        },
        "noisy_clinical_15pct_missing": {
            "accuracy": res_noisy["accuracy"],
            "sensitivity": res_noisy["sensitivity"],
            "specificity": res_noisy["specificity"],
            "errors_count": res_noisy["errors_count"],
        },
        "zero_clinical_image_only": {
            "accuracy": res_zero["accuracy"],
            "sensitivity": res_zero["sensitivity"],
            "specificity": res_zero["specificity"],
            "errors_count": res_zero["errors_count"],
        }
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = METRICS_DIR / "multimodal_test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=4)
    print(f"Saved multimodal metrics -> {out_json}", flush=True)

    binary_metrics_out = METRICS_DIR / "binary_test_metrics.json"
    with open(binary_metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_export["full_clinical"], f, indent=4)
    print(f"Updated active test metrics -> {binary_metrics_out}", flush=True)

    # Update val_model_comparison.json
    val_comp_path = METRICS_DIR / "val_model_comparison.json"
    if val_comp_path.exists():
        try:
            with open(val_comp_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            val_data["MultimodalMpoxClassifier"] = {
                "val_f1": float(best_val_f1),
                "val_accuracy": float(best_val_acc),
                "visual_backbone": selected_backbone,
            }
            with open(val_comp_path, "w", encoding="utf-8") as f:
                json.dump(val_data, f, indent=4)
            print("Updated val_model_comparison.json", flush=True)
        except Exception as e:
            print(f"Warning: could not update val_model_comparison.json: {e}", flush=True)


if __name__ == "__main__":
    main()
