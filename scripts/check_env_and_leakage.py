"""
scripts/check_env_and_leakage.py — Check environment packages and detect any potential leakage
between 'Monkeypox_Training img' and the independent test set.
"""
from pathlib import Path
import hashlib
from PIL import Image
import numpy as np

# 1. Environment check
print("=" * 60)
print("1. ENVIRONMENT & FRAMEWORK CHECK")
print("=" * 60)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except ImportError:
    print("TensorFlow: NOT installed")

try:
    import torch
    import torchvision
    print("PyTorch:", torch.__version__, "| Torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("PyTorch: NOT installed")

# 2. Duplicate and Leakage check against test set
print("\n" + "=" * 60)
print("2. LEAKAGE AUDIT: NEW IMAGES vs TEST SET")
print("=" * 60)

def hash_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def perceptual_hash(img_path):
    # Simple 8x8 average hash
    try:
        im = Image.open(img_path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.array(im.getdata(), dtype=np.float32)
        avg = pixels.mean()
        diff = pixels > avg
        return "".join(["1" if b else "0" for b in diff])
    except Exception:
        return ""

new_dir = Path("Monkeypox_Training img")
new_files = list(new_dir.glob("*"))

test_dir = Path("FOLDS/fold1/test")
test_files = list(test_dir.glob("*"))

print(f"Comparing {len(new_files)} new images against {len(test_files)} independent test images...")

test_exact_hashes = {}
test_phash_map = {}

for tf in test_files:
    if tf.is_file():
        h = hash_file(tf)
        test_exact_hashes[h] = tf.name
        ph = perceptual_hash(tf)
        if ph:
            test_phash_map[ph] = tf.name

exact_leaks = []
perceptual_leaks = []

for nf in new_files:
    if nf.is_file():
        h = hash_file(nf)
        if h in test_exact_hashes:
            exact_leaks.append((nf.name, test_exact_hashes[h]))
        
        ph = perceptual_hash(nf)
        if ph and ph in test_phash_map:
            perceptual_leaks.append((nf.name, test_phash_map[ph]))

print(f"Exact hash matches with test set: {len(exact_leaks)}")
if exact_leaks:
    print("  EXACT LEAKS FOUND:", exact_leaks)

print(f"Perceptual hash matches (near-duplicates) with test set: {len(perceptual_leaks)}")
if perceptual_leaks:
    print("  PERCEPTUAL MATCHES:", perceptual_leaks[:10])

# Also check patient ID overlaps (e.g. MKP_71, MKP_69 in test set)
print("\n" + "=" * 60)
print("3. PATIENT ID OVERLAP AUDIT")
print("=" * 60)
# Extract patient IDs in test set
test_patient_ids = set()
for tf in test_files:
    parts = tf.stem.split("_")
    if len(parts) >= 2:
        test_patient_ids.add(f"{parts[0]}_{parts[1]}")

print("Test set patient IDs:", sorted(list(test_patient_ids)))

patient_overlap = []
for nf in new_files:
    # Check if MSLDv2_MKP_XX matches any test patient
    if "MSLDv2_MKP" in nf.name:
        parts = nf.stem.split("_") # e.g. MSLDv2, MKP, 80, 01
        if len(parts) >= 4:
            patient_tag = f"{parts[1]}_{parts[2]}" # MKP_80
            if patient_tag in test_patient_ids:
                patient_overlap.append((nf.name, patient_tag))

print(f"Patient overlap between new images and test set: {len(patient_overlap)}")
if patient_overlap:
    print("  PATIENT OVERLAPS:", patient_overlap)
