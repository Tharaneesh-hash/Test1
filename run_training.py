"""
Binary Mpox Detection — Full Training Run
Trains EfficientNet-B0, ResNet34, VGG19 sequentially.
Selects best by validation MPOX F1. Does NOT touch test set.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, r'c:\Users\Tharaneesh.k\Downloads\Mpox-Detection')

from src.train import train_binary_model, select_best_model_on_validation
from src.config import (
    BEST_BINARY_MODEL_PATH, BEST_MODEL_NAME_FILE, METRICS_DIR
)
import shutil

print("=" * 70)
print("  BINARY MPOX DETECTION — TRAINING ALL ARCHITECTURES")
print("=" * 70)
print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

all_results = []

# ── 1. EfficientNet-B0 (fastest, ~15-20 min on CPU) ───────────────────────
print("\n>>> [1/3] Training EfficientNet-B0 ...")
t0 = time.time()
result_effnet = train_binary_model(
    arch_name="BinaryEfficientNetB0",
    fold_name="fold1",
    initial_epochs=15,
    fine_tune_epochs=5,
    batch_size=16,
)
all_results.append(result_effnet)
print(f"  EffNet done in {(time.time()-t0)/60:.1f} min — Val F1: {result_effnet['best_val_f1_mpox']:.4f}")

# ── 2. ResNet34 (medium speed, ~30-40 min) ─────────────────────────────────
print("\n>>> [2/3] Training ResNet34 ...")
t0 = time.time()
result_resnet = train_binary_model(
    arch_name="BinaryResNet34",
    fold_name="fold1",
    initial_epochs=15,
    fine_tune_epochs=5,
    batch_size=16,
)
all_results.append(result_resnet)
print(f"  ResNet34 done in {(time.time()-t0)/60:.1f} min — Val F1: {result_resnet['best_val_f1_mpox']:.4f}")

# ── 3. VGG19 (slowest, ~60-70 min) ────────────────────────────────────────
print("\n>>> [3/3] Training VGG19 ...")
t0 = time.time()
result_vgg = train_binary_model(
    arch_name="BinaryVGG19",
    fold_name="fold1",
    initial_epochs=15,
    fine_tune_epochs=5,
    batch_size=16,
)
all_results.append(result_vgg)
print(f"  VGG19 done in {(time.time()-t0)/60:.1f} min — Val F1: {result_vgg['best_val_f1_mpox']:.4f}")

# ── Select best by validation F1 ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  VALIDATION COMPARISON (NO TEST SET USED)")
print("=" * 70)
for r in all_results:
    print(f"  {r['arch']:30s}  Val F1={r['best_val_f1_mpox']:.4f}  Val Acc={r['best_val_acc']*100:.1f}%  Val AUC={r['best_val_auc']:.4f}")

best = select_best_model_on_validation(all_results)

# Copy best to canonical path
BEST_BINARY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(best["save_path"], BEST_BINARY_MODEL_PATH)
with open(BEST_MODEL_NAME_FILE, "w") as f:
    f.write(best["arch"])

# Save comparison JSON (val only)
summary_path = METRICS_DIR / "val_model_comparison.json"
with open(summary_path, "w") as f:
    json.dump({"validation_comparison": all_results, "selected": best}, f, indent=4)

print(f"\n  Best model: {best['arch']}")
print(f"  Checkpoint: {BEST_BINARY_MODEL_PATH}")
print(f"  Comparison: {summary_path}")
print(f"\n  Training complete at {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("  Next step: run src/evaluate.py for test-set evaluation")
