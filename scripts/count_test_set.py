"""
scripts/count_test_set.py — Count full test set by class.
"""
from collections import Counter
from pathlib import Path

test_dir = Path("FOLDS/fold1/test")
counts = Counter()
for f in test_dir.iterdir():
    if f.is_file():
        prefix = f.name.split("_")[0]
        counts[prefix] += 1

print("Total files:", sum(counts.values()))
for k, v in counts.most_common():
    print(f"  {k}: {v}")
