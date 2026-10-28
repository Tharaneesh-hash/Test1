"""
src/train.py — Mpox Detection Training Pipeline.

Trains and validates deep architectures (VGG19, ResNet34, EfficientNet-B0,
ConvNeXt-Tiny, Swin Transformer, and Multimodal Clinical Fusion) with:
  - Stage 1: Frozen backbone -> train classification head (AdamW, Cosine Annealing)
  - Stage 2: Targeted backbone unfreezing -> fine-tuning (lower LR)
  - Early stopping monitored on validation F1-MPOX
  - Best checkpoint saved on validation performance only
  - Zero test-set leakage (test split evaluated only once)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import (
    BATCH_SIZE,
    INITIAL_EPOCHS,
    INITIAL_LR,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    EARLY_STOP_PATIENCE,
    DEFAULT_FOLD,
    BINARY_VGG19_PATH,
    BINARY_RESNET34_PATH,
    BINARY_EFFNET_PATH,
    BINARY_CONVNEXT_PATH,
    BINARY_SWIN_PATH,
    BINARY_MULTIMODAL_PATH,
    BEST_BINARY_MODEL_PATH,
    BEST_MODEL_NAME_FILE,
    MPOX_CLASS_INDEX,
    METRICS_DIR,
    PLOTS_DIR,
)
from src.dataset import get_data_loaders
from src.model import (
    BinaryVGG19,
    BinaryResNet34,
    BinaryEfficientNetB0,
    BinaryConvNeXt,
    BinarySwinTransformer,
    MultimodalMpoxClassifier,
    save_binary_model,
)


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        imgs = batch[0].to(device)
        labels = batch[1].to(device)
        optimizer.zero_grad()

        if isinstance(model, MultimodalMpoxClassifier) and len(batch) >= 5:
            clin_data = batch[4].to(device)
            out = model(imgs, clin_data)
        else:
            out = model(imgs)

        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for batch in loader:
        imgs = batch[0].to(device)
        labels = batch[1].to(device)

        if isinstance(model, MultimodalMpoxClassifier) and len(batch) >= 5:
            clin_data = batch[4].to(device)
            out = model(imgs, clin_data)
        else:
            out = model(imgs)

        loss = criterion(out, labels)
        probs = torch.softmax(out, dim=1)[:, MPOX_CLASS_INDEX]
        total_loss += loss.item() * imgs.size(0)

        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, pos_label=MPOX_CLASS_INDEX, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / n, float(acc), float(f1), float(auc)


def plot_training_history(history: Dict[str, List], model_name: str, save_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "r--s", label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r--s", label="Val Acc")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.4)

    axes[2].plot(epochs, history["val_f1"],  "g-^", label="Val MPOX F1")
    axes[2].plot(epochs, history["val_auc"], "m-D", label="Val AUC")
    axes[2].set_title(f"{model_name} — Val F1 & AUC")
    axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Training curves saved -> {save_path}")


def train_binary_model(
    arch_name: str = "BinaryResNet34",
    fold_name: str = DEFAULT_FOLD,
    initial_epochs: int = INITIAL_EPOCHS,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device_str: Optional[str] = None,
    use_roi: bool = False,
) -> Dict:
    """
    Full two-stage training for one architecture.
    Returns validation metrics dict (does NOT touch test set).
    """
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*60}")
    print(f"Training {arch_name}  |  Device: {device}  |  Fold: {fold_name} | ROI: {use_roi}")
    print(f"{'='*60}")

    train_loader, valid_loader, _, class_weights = get_data_loaders(
        fold_name, batch_size, use_roi=use_roi, verbose=True
    )
    class_weights = class_weights.to(device)

    # Build model
    model_map = {
        "BinaryVGG19":              BinaryVGG19,
        "BinaryResNet34":           BinaryResNet34,
        "BinaryEfficientNetB0":     BinaryEfficientNetB0,
        "BinaryConvNeXt":           BinaryConvNeXt,
        "BinarySwinTransformer":    BinarySwinTransformer,
        "MultimodalMpoxClassifier": MultimodalMpoxClassifier,
    }
    if arch_name not in model_map:
        raise ValueError(f"Unknown arch: {arch_name}. Choose from {list(model_map)}")
    model = model_map[arch_name](pretrained=True, freeze_backbone=True).to(device)

    save_path_map = {
        "BinaryVGG19":              BINARY_VGG19_PATH,
        "BinaryResNet34":           BINARY_RESNET34_PATH,
        "BinaryEfficientNetB0":     BINARY_EFFNET_PATH,
        "BinaryConvNeXt":           BINARY_CONVNEXT_PATH,
        "BinarySwinTransformer":    BINARY_SWIN_PATH,
        "MultimodalMpoxClassifier": BINARY_MULTIMODAL_PATH,
    }
    save_path = save_path_map[arch_name]

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    def run_stage(stage: int, n_epochs: int, lr: float):
        nonlocal best_val_f1, best_state, patience_counter

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl_loss, vl_acc, vl_f1, vl_auc = validate(model, valid_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)
            history["val_f1"].append(vl_f1)
            history["val_auc"].append(vl_auc)

            dt = time.time() - t0
            print(
                f"  Stage{stage} Ep {epoch:02d}/{n_epochs} [{dt:.0f}s] "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.1f}% "
                f"val_acc={vl_acc*100:.1f}% val_f1={vl_f1:.4f} val_auc={vl_auc:.4f}",
                flush=True
            )

            # Checkpoint on best validation F1-MPOX
            if vl_f1 > best_val_f1:
                best_val_f1 = vl_f1
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_binary_model(model, save_path, arch_name)
                print(f"    [BEST] New best val MPOX F1 = {best_val_f1:.4f} - checkpoint saved", flush=True)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping (no F1 improvement for {EARLY_STOP_PATIENCE} epochs)", flush=True)
                    return True
        return False

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}
    best_val_f1, best_state, patience_counter = 0.0, None, 0

    # Stage 1: Frozen backbone
    print(f"\n--- Stage 1: Frozen backbone, training head only (lr={INITIAL_LR}) ---")
    run_stage(1, initial_epochs, INITIAL_LR)

    # Restore best weights before Stage 2
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    patience_counter = 0

    # Stage 2: Unfreeze deep layers
    if fine_tune_epochs > 0:
        print(f"\n--- Stage 2: Fine-tuning backbone (lr={FINE_TUNE_LR}) ---")
        if hasattr(model, "unfreeze_last_stage"):
            model.unfreeze_last_stage()
        elif hasattr(model, "unfreeze_block5"):
            model.unfreeze_block5()
        elif hasattr(model, "unfreeze_last_block"):
            model.unfreeze_last_block()
        elif hasattr(model, "unfreeze_layer3"):
            model.unfreeze_layer3()
        run_stage(2, fine_tune_epochs, FINE_TUNE_LR)

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    save_binary_model(model, save_path, arch_name)

    hist_path = METRICS_DIR / f"{arch_name}_training_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)

    curve_path = PLOTS_DIR / f"{arch_name}_training_curves.png"
    plot_training_history(history, arch_name, curve_path)

    result = {
        "arch": arch_name,
        "best_val_f1_mpox": best_val_f1,
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0.0,
        "best_val_auc": max(history["val_auc"]) if history["val_auc"] else 0.0,
        "save_path": str(save_path),
        "total_epochs": len(history["train_loss"]),
    }
    print(f"\n  [{arch_name}] Training complete. Best Val MPOX F1: {best_val_f1:.4f}")
    return result


def select_best_model_on_validation(results: List[Dict]) -> Dict:
    """Selects best model using validation MPOX F1 only."""
    best = max(results, key=lambda r: r["best_val_f1_mpox"])
    print(f"\n{'='*60}")
    print(f"BEST MODEL (by val MPOX F1): {best['arch']}")
    print(f"  Val MPOX F1 = {best['best_val_f1_mpox']:.4f}")
    print(f"  Val Accuracy = {best['best_val_acc']*100:.2f}%")
    print(f"{'='*60}")
    return best


def train_all_and_select(
    fold_name: str = DEFAULT_FOLD,
    batch_size: int = BATCH_SIZE,
    device_str: Optional[str] = None,
    architectures: Optional[List[str]] = None,
):
    import shutil

    if architectures is None:
        architectures = ["BinaryResNet34", "BinaryConvNeXt", "MultimodalMpoxClassifier"]

    all_results = []
    for arch in architectures:
        result = train_binary_model(
            arch_name=arch,
            fold_name=fold_name,
            batch_size=batch_size,
            device_str=device_str,
        )
        all_results.append(result)

    best = select_best_model_on_validation(all_results)

    BEST_BINARY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["save_path"], BEST_BINARY_MODEL_PATH)

    with open(BEST_MODEL_NAME_FILE, "w") as f:
        f.write(best["arch"])

    summary_path = METRICS_DIR / "val_model_comparison.json"
    with open(summary_path, "w") as f:
        json.dump({"validation_comparison": all_results, "selected": best}, f, indent=4)
    print(f"\nValidation comparison saved -> {summary_path}")
    print(f"Best model checkpoint -> {BEST_BINARY_MODEL_PATH}")

    return best, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Binary Mpox Models")
    parser.add_argument("--fold",   default=DEFAULT_FOLD)
    parser.add_argument("--archs",  nargs="+", default=["BinaryResNet34", "BinaryConvNeXt"])
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS)
    parser.add_argument("--ft-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train_all_and_select(
        fold_name=args.fold,
        batch_size=args.batch,
        architectures=args.archs,
    )
