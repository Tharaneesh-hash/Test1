"""
scripts/copy_to_artifact.py
"""
import shutil
from pathlib import Path

art_dir = Path(r"C:\Users\Tharaneesh.k\.gemini\antigravity\brain\587e6be8-80f3-4796-8266-6646a51d83cd\error_analysis_orig")
art_dir.mkdir(parents=True, exist_ok=True)

test_dir = Path("FOLDS/fold1/test")
error_files = [
    "MKP_71_03.jpg", "MKP_71_01.jpg", "MKP_69_02.jpg", "MKP_71_04.jpg", "MKP_71_02.jpg",
    "CHP_14_02.jpg", "CHP_14_01.jpg", "CHP_19_01.jpg", "CHP_49_01.jpg", "HEALTHY_79_01.jpg",
    "CHP_08_01.jpg", "CWP_09_01.jpg", "CWP_40_02.jpg", "CHP_08_02.jpg"
]

copied = 0
for f in error_files:
    src = test_dir / f
    if src.exists():
        shutil.copy2(src, art_dir / f)
        copied += 1

print(f"Copied {copied}/{len(error_files)} original error images to {art_dir}")
