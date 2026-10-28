"""
scripts/audit_new_images_full.py — Comprehensive audit of all 524 images in 'Monkeypox_Training img'.
Saves categorization to results/metrics/new_images_audit.json.
"""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import hashlib
import json
from PIL import Image
import numpy as np
from collections import defaultdict

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def phash(path):
    try:
        im = Image.open(path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        arr = np.array(im.getdata(), dtype=np.float32)
        diff = arr > arr.mean()
        return "".join(["1" if b else "0" for b in diff])
    except Exception:
        return ""

def hamming_dist(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# 1. Index all existing dataset images in fold1
fold1_dir = Path("FOLDS/fold1")
splits = ["train", "valid", "test"]

hash_to_existing = {}
phash_to_existing = defaultdict(list)
patient_to_existing = defaultdict(set)

for s in splits:
    for f in (fold1_dir / s).glob("*.*"):
        if f.is_file():
            h = sha256(f)
            hash_to_existing[h] = (s, f.name)
            p_h = phash(f)
            if p_h:
                phash_to_existing[p_h].append((s, f.name))
            parts = f.stem.split("_")
            if len(parts) >= 2:
                patient_to_existing[f"{parts[0]}_{parts[1]}"].add(s)

print(f"Indexed existing dataset: {len(hash_to_existing)} files.")
test_patients = {p for p, ss in patient_to_existing.items() if "test" in ss}

# 2. Audit the 524 new images
new_dir = Path("Monkeypox_Training img")
new_files = sorted(list(new_dir.glob("*.*")))

categorization = {
    "TEST_LEAK_EXACT": [],
    "TEST_LEAK_PATIENT": [],
    "TEST_LEAK_PHASH": [],
    "ALREADY_IN_TRAIN": [],
    "ALREADY_IN_VALID": [],
    "GENUINELY_NEW": [],
}

for nf in new_files:
    if not nf.is_file():
        continue
    h = sha256(nf)
    p_h = phash(nf)
    
    # Check exact match with test
    if h in hash_to_existing and hash_to_existing[h][0] == "test":
        categorization["TEST_LEAK_EXACT"].append({
            "new_filename": nf.name,
            "matched_test_file": hash_to_existing[h][1],
            "reason": "Exact SHA-256 hash collision with independent test set"
        })
        continue
    
    # Check patient overlap with test
    parts = nf.stem.split("_")
    is_patient_leak = False
    if "MSLDv2_MKP" in nf.name and len(parts) >= 4:
        p_id = f"{parts[1]}_{parts[2]}" # e.g. MKP_10
        if p_id in test_patients:
            categorization["TEST_LEAK_PATIENT"].append({
                "new_filename": nf.name,
                "matched_test_patient": p_id,
                "reason": f"Patient identifier {p_id} belongs to independent test set"
            })
            is_patient_leak = True
    if is_patient_leak:
        continue
    
    # Check near-duplicate phash with test
    is_phash_leak = False
    if p_h:
        for exist_ph, exist_list in phash_to_existing.items():
            if hamming_dist(p_h, exist_ph) <= 2:
                for ex_split, ex_name in exist_list:
                    if ex_split == "test":
                        categorization["TEST_LEAK_PHASH"].append({
                            "new_filename": nf.name,
                            "matched_test_file": ex_name,
                            "reason": f"Perceptual hash collision (Hamming <= 2) with test file {ex_name}"
                        })
                        is_phash_leak = True
                        break
            if is_phash_leak:
                break
    if is_phash_leak:
        continue
    
    # Check exact match with train/valid
    if h in hash_to_existing:
        split_name = hash_to_existing[h][0]
        if split_name == "train":
            categorization["ALREADY_IN_TRAIN"].append({
                "new_filename": nf.name,
                "matched_train_file": hash_to_existing[h][1]
            })
        elif split_name == "valid":
            categorization["ALREADY_IN_VALID"].append({
                "new_filename": nf.name,
                "matched_valid_file": hash_to_existing[h][1]
            })
        continue
    
    # If passed all checks, it is genuinely new!
    categorization["GENUINELY_NEW"].append({
        "filename": nf.name,
        "path": str(nf),
        "source": nf.stem.split("_")[0]
    })

out_json = Path("results/metrics/new_images_audit.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "total_audited": len(new_files),
        "summary": {k: len(v) for k, v in categorization.items()},
        "details": categorization
    }, f, indent=2)

print("\nSaved detailed audit to:", out_json)
print("Summary:")
for k, v in categorization.items():
    print(f"  {k}: {len(v)}")
