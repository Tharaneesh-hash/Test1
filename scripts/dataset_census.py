"""
scripts/dataset_census.py — Full census of current dataset across splits.
"""
from pathlib import Path
from collections import Counter

fold_dir = Path("FOLDS/fold1")
for split in ["train", "valid", "test"]:
    s_dir = fold_dir / split
    files = list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.png"))
    c = Counter(f.name.split("_")[0] for f in files)
    print(f"[{split.upper()}] Total: {len(files)}")
    for k, v in c.most_common():
        print(f"   {k}: {v}")
    print()
