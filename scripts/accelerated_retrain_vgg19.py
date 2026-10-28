"""
scripts/accelerated_retrain_vgg19.py — Accelerated, mathematically exact retraining of VGG19
using feature caching for Stage 1, followed by Block 5 fine-tuning in Stage 2.
Evaluates on the untouched 74-sample independent test set and saves all evaluation artifacts.
"""
import sys
sys.path.insert(0, ".")
import time
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
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
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
)
from src.dataset import get_data_loaders
from src.model import BinaryVGG19, save_binary_model, load_binary_model

def wilson_ci(k, n, confidence=0.95):
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
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
    
    acc_ci = wilson_ci(int(tp + tn), len(y_true))
    sens_ci = wilson_ci(int(tp), int(tp + fn))
    spec_ci = wilson_ci(int(tn), int(tn + fp))
    
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
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_test_samples": len(y_true),
        "total_errors": len(errors),
        "errors": errors,
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "cm_raw": cm.tolist()
    }

def main():
    print("=" * 70)
    print("  ACCELERATED RETRAINING: VGG19 ON ENRICHED DATASET")
    print("  Integrated: 287 New Mpox Images | Quarantined: 27 Test Leaks")
    print("=" * 70)

    device = torch.device("cpu")
    t_start = time.time()

    # 1. Data Loaders (Train=781, Valid=187, Test=74)
    train_loader, valid_loader, test_loader, class_weights = get_data_loaders(
        DEFAULT_FOLD, batch_size=32, verbose=True
    )
    class_weights = class_weights.to(device)

    # 2. Instantiate VGG19 with pretrained ImageNet weights
    print("\nLoading Pretrained VGG19 Backbone...")
    model = BinaryVGG19(pretrained=True, freeze_backbone=True).to(device)

    # 3. Cache Features for Stage 1 (Fast Feature Extraction)
    print("\n--- Extracting and Caching Backbone Features (One Pass) ---")
    t0 = time.time()
    
    def extract_all(loader):
        feats, labels = [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0].to(device)
                lbls = batch[1]
                f = model.extract_features(imgs)
                feats.append(f.cpu())
                labels.append(lbls)
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

    train_feats, train_labels = extract_all(train_loader)
    val_feats, val_labels = extract_all(valid_loader)
    print(f"Features cached in {time.time() - t0:.1f}s | Train: {train_feats.shape} | Valid: {val_feats.shape}")

    feat_train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=32, shuffle=True)

    # 4. Stage 1: Train Classification Head
    print("\n--- Stage 1: Training Classification Head (25 Epochs on Cached Features) ---")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    head = model.classifier_head
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    best_val_f1 = 0.0
    best_head_state = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}

    for epoch in range(1, 26):
        head.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b in feat_train_loader:
            optimizer.zero_grad()
            out = head[1:](x_b)  # head is Sequential(Flatten, Linear, BN, ReLU, Dropout, Linear)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_b.size(0)
            correct += (out.argmax(dim=1) == y_b).sum().item()
            total += y_b.size(0)
        scheduler.step()

        tr_loss = total_loss / total
        tr_acc = correct / total

        # Validate
        head.eval()
        with torch.no_grad():
            val_out = head[1:](val_feats)
            vl_loss = criterion(val_out, val_labels).item()
            val_preds = val_out.argmax(dim=1).numpy()
            val_probs = torch.softmax(val_out, dim=1)[:, MPOX_CLASS_INDEX].numpy()

        vl_acc = accuracy_score(val_labels.numpy(), val_preds)
        vl_f1 = f1_score(val_labels.numpy(), val_preds, pos_label=MPOX_CLASS_INDEX, zero_division=0)
        vl_auc = roc_auc_score(val_labels.numpy(), val_probs)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)
        history["val_auc"].append(vl_auc)

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            print(f"  Stage1 Ep {epoch:02d}/25 | tr_acc={tr_acc*100:.1f}% | val_acc={vl_acc*100:.1f}% | val_f1={vl_f1:.4f} [BEST]")
        elif epoch % 5 == 0:
            print(f"  Stage1 Ep {epoch:02d}/25 | tr_acc={tr_acc*100:.1f}% | val_acc={vl_acc*100:.1f}% | val_f1={vl_f1:.4f}")

    # Restore best head weights
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    print(f"Stage 1 Best Val F1: {best_val_f1:.4f}")

    # 5. Stage 2: Fine-Tuning Block 5 Conv Layers
    print("\n--- Stage 2: Fine-Tuning Block 5 Conv Layers (3 Epochs, lr=5e-5) ---")
    model.unfreeze_block5()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ft_optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=WEIGHT_DECAY)
    ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(ft_optimizer, T_max=3, eta_min=1e-6)

    best_state_overall = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(1, 4):
        t_ep = time.time()
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for batch in train_loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            ft_optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.5)
            ft_optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(dim=1) == lbls).sum().item()
            tr_total += lbls.size(0)
        ft_scheduler.step()

        # Validate
        model.eval()
        val_preds, val_targets, val_probs_list = [], [], []
        vl_loss_total = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                imgs, lbls = batch[0].to(device), batch[1].to(device)
                out = model(imgs)
                l = criterion(out, lbls)
                vl_loss_total += l.item() * imgs.size(0)
                probs = torch.softmax(out, dim=1)[:, MPOX_CLASS_INDEX]
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_targets.extend(lbls.cpu().numpy())
                val_probs_list.extend(probs.cpu().numpy())

        ep_dt = time.time() - t_ep
        vl_acc = accuracy_score(val_targets, val_preds)
        vl_f1 = f1_score(val_targets, val_preds, pos_label=MPOX_CLASS_INDEX, zero_division=0)
        vl_auc = roc_auc_score(val_targets, val_probs_list)
        print(f"  Stage2 Ep {epoch:02d}/03 [{ep_dt:.0f}s] | tr_acc={tr_correct/tr_total*100:.1f}% | val_acc={vl_acc*100:.1f}% | val_f1={vl_f1:.4f} | val_auc={vl_auc:.4f}")

        if vl_f1 >= best_val_f1:
            best_val_f1 = vl_f1
            best_state_overall = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    [BEST] New checkpoint saved with Val F1 = {best_val_f1:.4f}")

    # Restore best overall weights and save
    model.load_state_dict(best_state_overall)
    save_binary_model(model, BINARY_VGG19_PATH, "BinaryVGG19")
    print(f"\nSaved best VGG19 model checkpoint -> {BINARY_VGG19_PATH}")

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("VGG19 Retrained — Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.plot(history["val_f1"], label="Val MPOX F1", linestyle="--")
    plt.title("VGG19 Retrained — Accuracy & F1")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_save = PLOTS_DIR / "BinaryVGG19_retrained_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_save, dpi=200)
    plt.close()
    print(f"Training plots saved -> {plot_save}")

    # 6. Evaluate on Locked 74-Sample Independent Test Set
    print("\n" + "=" * 70)
    print("  EVALUATING RETRAINED VGG19 ON INDEPENDENT TEST SET (74 IMAGES)")
    print("=" * 70)
    test_eval = evaluate_on_test(model, test_loader, device)

    # Plot Confusion Matrix
    cm = np.array(test_eval["cm_raw"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Retrained VGG19 Confusion Matrix (Test Set)")
    plt.colorbar()
    tick_marks = np.arange(len(BINARY_CLASS_NAMES))
    plt.xticks(tick_marks, BINARY_CLASS_NAMES)
    plt.yticks(tick_marks, BINARY_CLASS_NAMES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_plot_save = PLOTS_DIR / "vgg19_retrained_confusion_matrix.png"
    plt.savefig(cm_plot_save, dpi=200)
    plt.close()

    # Save metrics JSON
    metrics_path = METRICS_DIR / "vgg19_retrained_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "BinaryVGG19_Retrained",
            "training_duration_seconds": round(time.time() - t_start, 1),
            "best_val_f1_mpox": round(best_val_f1, 4),
            "test_evaluation": {
                "accuracy": test_eval["accuracy"],
                "accuracy_95ci": test_eval["accuracy_95ci"],
                "sensitivity": test_eval["sensitivity"],
                "sensitivity_95ci": test_eval["sensitivity_95ci"],
                "specificity": test_eval["specificity"],
                "specificity_95ci": test_eval["specificity_95ci"],
                "precision": test_eval["precision"],
                "f1_score": test_eval["f1_score"],
                "roc_auc": test_eval["roc_auc"],
                "confusion_matrix": test_eval["confusion_matrix"],
                "total_errors": test_eval["total_errors"],
                "total_test_samples": test_eval["total_test_samples"],
                "errors": test_eval["errors"]
            }
        }, f, indent=2)

    # Print summary
    print(f"  Test Accuracy:    {test_eval['accuracy']:.2%}  (95% CI: {test_eval['accuracy_95ci'][0]:.1%} - {test_eval['accuracy_95ci'][1]:.1%})")
    print(f"  Mpox Sensitivity: {test_eval['sensitivity']:.2%}  (95% CI: {test_eval['sensitivity_95ci'][0]:.1%} - {test_eval['sensitivity_95ci'][1]:.1%})")
    print(f"  Specificity:      {test_eval['specificity']:.2%}  (95% CI: {test_eval['specificity_95ci'][0]:.1%} - {test_eval['specificity_95ci'][1]:.1%})")
    print(f"  Precision:        {test_eval['precision']:.2%}")
    print(f"  F1-Score:         {test_eval['f1_score']:.4f}")
    print(f"  ROC-AUC:          {test_eval['roc_auc']:.4f}")
    print(f"  Total Errors:     {test_eval['total_errors']} / {test_eval['total_test_samples']}")
    print(f"  Total Time:       {(time.time() - t_start) / 60:.1f} minutes")
    print(f"\nSaved metrics -> {metrics_path}")
    print(f"Saved confusion matrix -> {cm_plot_save}")

if __name__ == "__main__":
    main()
