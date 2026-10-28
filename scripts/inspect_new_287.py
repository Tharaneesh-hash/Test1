"""
scripts/inspect_new_287.py — Inspect sources and patient clusters among the 287 new images.
"""
import json
from pathlib import Path
from collections import Counter

# Read the categorization from previous audit
from scripts.audit_new_images_full import categorization

new_names = categorization["GENUINELY_NEW"]
print("Total genuinely new images:", len(new_names))

sources = Counter()
patients = set()

for name in new_names:
    parts = name.split("_")
    source = parts[0]
    sources[source] += 1
    # Check patient id if available
    if "MSLDv2" in name:
        patients.add(f"MSLDv2_{parts[2]}")
    elif "MID" in name:
        patients.add(f"MID_{parts[-1].split('.')[0]}")
    elif "MSID" in name:
        patients.add(f"MSID_{name.split('.')[0]}")

print("\nSource distribution:")
for s, c in sources.most_common():
    print(f"  {s}: {c}")

print(f"\nDistinct patient/source entities: {len(patients)}")
