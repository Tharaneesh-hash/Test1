import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

from src.config import (
    BEST_MODEL_PATH,
    CLASS_MAPPING_PATH,
    MODEL_CONFIG_PATH,
    HISTORY_PATH,
    TRAINING_CURVES_PLOT,
    NUM_CLASSES,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    INITIAL_LR,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    DEFAULT_FOLD
)
from src.dataset import get_data_loaders
from src.model import VGG19MpoxClassifier, save_model_artifacts


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(all_labels)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return val_loss, float(val_acc), float(val_macro_f1)


def plot_training_history(history: Dict[str, List[float]], save_path: Path):
    """
    Plots training and validation loss and accuracy curves over epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("Training & Validation Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy curve
    ax1_acc = history["train_acc"]
    ax2.plot(epochs, ax1_acc, "b-o", label="Training Accuracy")
    ax2.plot(epochs, history["val_acc"], "r--s", label="Validation Accuracy")
    ax2.set_title("Training & Validation Categorical Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")


def train_mpox_vgg19(
    fold_name: str = DEFAULT_FOLD,
    initial_epochs: int = INITIAL_EPOCHS,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the two-stage transfer learning training procedure:
    Stage 1: Frozen VGG19 feature backbone, training classification head with Adam (lr=1e-3).
    Stage 2: Unfreeze VGG19 Block 5 conv layers for fine-tuning with Adam (lr=1e-5).
    Includes Early Stopping, ReduceLROnPlateau, and Best Checkpointing.
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"==================================================")
    print(f"Starting Mpox VGG19 Transfer Learning Pipeline")
    print(f"Device: {device}")
    print(f"Dataset Fold: {fold_name} | Batch Size: {batch_size}")
    print(f"==================================================")

    # Load DataLoaders
    train_loader, valid_loader, test_loader, class_weights = get_data_loaders(
        fold_name=fold_name,
        batch_size=batch_size
    )
    class_weights = class_weights.to(device)

    # Initialize Model with frozen backbone
    model = VGG19MpoxClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    # Loss function with class weights to account for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Stage 1 Optimizer: Only train parameters that require grad (head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=INITIAL_LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }

    best_val_loss = float("inf")
    best_weights = None
    patience = 4
    patience_counter = 0

    print("\n--- STAGE 1: Training Classification Head (Backbone Frozen) ---")
    start_time = time.time()

    for epoch in range(1, initial_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{initial_epochs:02d} [{epoch_time:.1f}s] - "
              f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f} - LR: {current_lr:.6f}")

        # Best Model Checkpointing — save to disk immediately on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            # Save checkpoint to disk immediately so a crash doesn't lose progress
            save_model_artifacts(model, BEST_MODEL_PATH, MODEL_CONFIG_PATH, CLASS_MAPPING_PATH)
            print(f"  [Checkpoint] New best validation loss: {best_val_loss:.4f}. Saved to {BEST_MODEL_PATH}.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] No improvement in validation loss for {patience} epochs. Stopping Stage 1.")
                break

    # Restore best weights from Stage 1
    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(device)

    # Optional Stage 2: Fine-tuning Block 5 conv layers
    if fine_tune_epochs > 0:
        print("\n--- STAGE 2: Fine-Tuning VGG19 Block 5 Layers ---")
        model.unfreeze_block5()

        ft_params = [p for p in model.parameters() if p.requires_grad]
        ft_optimizer = optim.Adam(ft_params, lr=FINE_TUNE_LR)
        ft_scheduler = ReduceLROnPlateau(ft_optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-7)

        for ft_epoch in range(1, fine_tune_epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, ft_optimizer, device)
            val_loss, val_acc, val_f1 = validate(model, valid_loader, criterion, device)
            ft_scheduler.step(val_loss)

            current_lr = ft_optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_f1)
            history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start
            total_epoch_idx = initial_epochs + ft_epoch
            print(f"FT Epoch {ft_epoch:02d}/{fine_tune_epochs:02d} [{epoch_time:.1f}s] - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f} - LR: {current_lr:.7f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                save_model_artifacts(model, BEST_MODEL_PATH, MODEL_CONFIG_PATH, CLASS_MAPPING_PATH)
                print(f"  [Checkpoint] New best validation loss during fine-tuning: {best_val_loss:.4f}. Saved to disk.")

    # Load absolute best weights before saving
    if best_weights is not None:
        model.load_state_dict(best_weights)

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time / 60:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Save artifacts
    save_model_artifacts(model, BEST_MODEL_PATH, MODEL_CONFIG_PATH, CLASS_MAPPING_PATH)
    print(f"Saved best model checkpoint to {BEST_MODEL_PATH}")

    # Save History JSON and curves
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {HISTORY_PATH}")

    plot_training_history(history, TRAINING_CURVES_PLOT)

    return {
        "best_val_loss": best_val_loss,
        "total_time_seconds": total_training_time,
        "history": history
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VGG19 Transfer Learning for Mpox Recognition")
    parser.add_argument("--fold", type=str, default=DEFAULT_FOLD, help="Dataset fold to train on")
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS, help="Stage 1 training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train_mpox_vgg19(
        fold_name=args.fold,
        initial_epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size
    )
