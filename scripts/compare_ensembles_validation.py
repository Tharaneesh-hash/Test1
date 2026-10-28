"""
scripts/compare_ensembles_validation.py — Compare ensemble combinations on validation set only.
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from src.config import DEFAULT_FOLD, MPOX_CLASS_INDEX
from src.dataset import get_data_loaders
from src.model import load_binary_model, BinaryVGG19, BinaryResNet34

device = torch.device("cpu")
vgg_retrained = load_binary_model(Path("models/binary_vgg19_best.pt"), device=device)
resnet_retrained = load_binary_model(Path("models/binary_resnet34_best.pt"), device=device)
resnet_baseline = load_binary_model(Path("models/binary_resnet34_baseline_backup.pt"), device=device)

_, valid_loader, test_loader, _ = get_data_loaders(DEFAULT_FOLD, batch_size=16, verbose=False)

def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch[0].to(device), batch[1]
            p = model.predict_proba(imgs)[:, MPOX_CLASS_INDEX]
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(lbls.numpy())
    return np.array(all_probs), np.array(all_labels)

# Validation probabilities
p_vgg_val, y_val = get_probs(vgg_retrained, valid_loader)
p_res_new_val, _ = get_probs(resnet_retrained, valid_loader)
p_res_base_val, _ = get_probs(resnet_baseline, valid_loader)

# Tune weights on validation
print("=" * 60)
print("VALIDATION-BASED ENSEMBLE COMPARISON")
print("=" * 60)

for name, p_res_val in [
    ("Retrained VGG19 + Retrained ResNet34", p_res_new_val),
    ("Retrained VGG19 + Baseline ResNet34", p_res_base_val),
]:
    best_f1, best_w, best_acc = 0.0, 0.5, 0.0
    for w in np.linspace(0, 1, 21):
        blended = w * p_vgg_val + (1 - w) * p_res_val
        preds = (blended >= 0.5).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        acc = accuracy_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_w = float(w)
            best_acc = acc
    print(f"Combination: {name}")
    print(f"  Best weight for VGG: {best_w:.2f} (ResNet: {1-best_w:.2f})")
    print(f"  Val F1: {best_f1:.4f} | Val Accuracy: {best_acc*100:.2f}%\n")
