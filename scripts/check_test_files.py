"""
scripts/check_test_files.py
"""
from pathlib import Path

p = Path("FOLDS/fold1/test")
all_files = list(p.glob("*"))
print(f"Total entries in {p}: {len(all_files)}")
jpgs = list(p.glob("*.jpg"))
pngs = list(p.glob("*.png"))
print(f"JPGs: {len(jpgs)}, PNGs: {len(pngs)}")
from collections import Counter
c = Counter(f.name.split("_")[0] for f in jpgs + pngs)
print(dict(c))
