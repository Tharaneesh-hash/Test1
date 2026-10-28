"""
scripts/inspect_new_images.py — Analyze newly added images in 'Monkeypox_Training img'.
"""
from pathlib import Path
from collections import Counter
from PIL import Image
import hashlib

new_dir = Path("Monkeypox_Training img")
if not new_dir.exists():
    print("Directory does not exist:", new_dir)
    exit(1)

all_files = list(new_dir.glob("*"))
print(f"Total files in '{new_dir}': {len(all_files)}")

# Check extensions
exts = Counter(f.suffix.lower() for f in all_files)
print("File extensions:", dict(exts))

# Group by naming convention
prefixes = Counter()
corrupted = []
valid_images = []
hashes = {}
duplicates_internal = []

for f in all_files:
    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        # Verify readability
        try:
            with Image.open(f) as img:
                img.verify()
            # Compute hash for duplicate detection
            with open(f, "rb") as fp:
                file_hash = hashlib.sha256(fp.read()).hexdigest()
            if file_hash in hashes:
                duplicates_internal.append((f.name, hashes[file_hash]))
            else:
                hashes[file_hash] = f.name
            
            # Identify prefix
            parts = f.stem.split("_")
            prefix = parts[0]
            if len(parts) > 1 and parts[0] in ["MID", "MSLDv2"]:
                prefix = f"{parts[0]}_{parts[1]}"
            prefixes[prefix] += 1
            valid_images.append(f)
        except Exception as e:
            corrupted.append((f.name, str(e)))

print("\nValid images:", len(valid_images))
print("Corrupted images:", len(corrupted))
if corrupted:
    print("Corrupted list:", corrupted)

print("\nNaming prefix distribution:")
for p, count in prefixes.most_common():
    print(f"  {p}: {count}")

print(f"\nInternal duplicates detected: {len(duplicates_internal)}")
if duplicates_internal:
    print("Sample internal duplicates:", duplicates_internal[:5])
