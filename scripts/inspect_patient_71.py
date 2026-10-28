"""
scripts/inspect_patient_71.py — Deep dive into MKP_71 and MKP_69.
"""

from pathlib import Path
from PIL import Image
import numpy as np

for fname in ["MKP_71_01.jpg", "MKP_71_02.jpg", "MKP_71_03.jpg", "MKP_71_04.jpg", "MKP_69_02.jpg"]:
    fpath = Path(f"FOLDS/fold1/test/{fname}")
    im = Image.open(fpath)
    arr = np.array(im)
    print(f"=== {fname} ===")
    print("Dimensions:", arr.shape)
    # Check color channels
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    print(f"Mean R={r.mean():.1f}, G={g.mean():.1f}, B={b.mean():.1f}")
    # Inspect dynamic range and quantiles
    q10, q50, q90 = np.percentile(arr, [10, 50, 90])
    print(f"Quantiles: 10%={q10:.1f}, 50%={q50:.1f}, 90%={q90:.1f}")
    # Check if there is an off-center lesion or multiple diffuse spots
    # Split into 3x3 grid
    h, w, _ = arr.shape
    grid_means = []
    for row in range(3):
        row_vals = []
        for col in range(3):
            cell = arr[row*h//3:(row+1)*h//3, col*w//3:(col+1)*w//3]
            row_vals.append(round(cell.mean(), 1))
        grid_means.append(row_vals)
    print("3x3 Grid luminance:")
    for row in grid_means:
        print("  ", row)
    print()
