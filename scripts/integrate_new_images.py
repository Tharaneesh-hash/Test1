"""
scripts/integrate_new_images.py — Safely integrates 287 verified new Mpox images into fold1
(train: ~85%, valid: ~15%) and quarantines test-leakage images.
"""
import sys
sys.path.insert(0, ".")
import json
import shutil
from pathlib import Path
import numpy as np

# Load audit results
audit_file = Path("results/metrics/new_images_audit.json")
with open(audit_file, "r", encoding="utf-8") as f:
    audit_data = json.load(f)

details = audit_data["details"]

# 1. Setup Quarantine Folders
quarantine_dir = Path("review_folder/test_overlap_quarantine")
quarantine_dir.mkdir(parents=True, exist_ok=True)

new_dir = Path("Monkeypox_Training img")

quarantined_count = 0
quarantine_manifest = []

# Quarantine exact test leaks
for item in details["TEST_LEAK_EXACT"]:
    src = new_dir / item["new_filename"]
    if src.exists():
        shutil.copy2(src, quarantine_dir / item["new_filename"])
        quarantined_count += 1
        quarantine_manifest.append(item)

# Quarantine perceptual test leaks
for item in details["TEST_LEAK_PHASH"]:
    src = new_dir / item["new_filename"]
    if src.exists():
        shutil.copy2(src, quarantine_dir / item["new_filename"])
        quarantined_count += 1
        quarantine_manifest.append(item)

print(f"Quarantined {quarantined_count} test-leakage images into {quarantine_dir}")

# 2. Split and Integrate the 287 Genuinely New Images
genuinely_new = details["GENUINELY_NEW"]
print(f"Integrating {len(genuinely_new)} genuinely new Mpox images...")

rng = np.random.default_rng(42)
shuffled_indices = rng.permutation(len(genuinely_new))

# 85% train, 15% validation
n_valid = int(len(genuinely_new) * 0.15)
valid_indices = set(shuffled_indices[:n_valid])
train_indices = set(shuffled_indices[n_valid:])

print(f"Split planned: {len(train_indices)} -> Train, {len(valid_indices)} -> Valid")

train_dest = Path("FOLDS/fold1/train")
valid_dest = Path("FOLDS/fold1/valid")

integration_manifest = {
    "quarantined_test_leaks": quarantine_manifest,
    "integrated_train": [],
    "integrated_valid": []
}

added_train = 0
added_valid = 0

for idx, item in enumerate(genuinely_new):
    src = Path(item["path"])
    raw_name = item["filename"]
    
    # Prefix with MKP_ for seamless downstream binary mapping
    if raw_name.startswith("MSLDv2_MKP_"):
        new_name = raw_name.replace("MSLDv2_MKP_", "MKP_MSLDv2_")
    elif raw_name.startswith("MID_MonkeyPox_"):
        new_name = raw_name.replace("MID_MonkeyPox_", "MKP_MID_")
    elif raw_name.startswith("MID_MonkeyPox"):
        new_name = raw_name.replace("MID_MonkeyPox", "MKP_MID_00")
    elif raw_name.startswith("MSID_monkeypox"):
        new_name = raw_name.replace("MSID_monkeypox", "MKP_MSID_")
    else:
        new_name = f"MKP_NEW_{raw_name}"
    
    if idx in valid_indices:
        target_path = valid_dest / new_name
        shutil.copy2(src, target_path)
        added_valid += 1
        integration_manifest["integrated_valid"].append({
            "original_filename": raw_name,
            "integrated_filename": new_name,
            "destination": str(target_path)
        })
    else:
        target_path = train_dest / new_name
        shutil.copy2(src, target_path)
        added_train += 1
        integration_manifest["integrated_train"].append({
            "original_filename": raw_name,
            "integrated_filename": new_name,
            "destination": str(target_path)
        })

print(f"Successfully integrated {added_train} images into {train_dest}")
print(f"Successfully integrated {added_valid} images into {valid_dest}")

manifest_path = Path("results/metrics/dataset_integration_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({
        "quarantined_count": quarantined_count,
        "added_train_count": added_train,
        "added_valid_count": added_valid,
        "manifest": integration_manifest
    }, f, indent=2)

print("Saved integration manifest ->", manifest_path)
