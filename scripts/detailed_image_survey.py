"""
scripts/detailed_image_survey.py — Inspect colors, textures, and lesion characteristics
of the 14 misclassified images.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import json

with open("results/metrics/misclassified_analysis.json", "r") as f:
    data = json.load(f)

for item in data["items"]:
    fname = item["filename"]
    fpath = item["filepath"]
    img = Image.open(fpath).convert("RGB")
    arr = np.array(img)
    
    # Check if there is text in the corners or borders
    # (High local contrast in corner patches usually indicates watermarks/text/ruler)
    h, w, _ = arr.shape
    patch_size = 20
    corners = {
        "top_left": arr[:patch_size, :patch_size],
        "top_right": arr[:patch_size, -patch_size:],
        "bottom_left": arr[-patch_size:, :patch_size],
        "bottom_right": arr[-patch_size:, -patch_size:],
    }
    corner_info = []
    for cname, cpatch in corners.items():
        grad = np.std(cpatch)
        mean_val = np.mean(cpatch)
        if grad > 30 and mean_val > 180:
            corner_info.append(f"{cname}: text/ruler/watermark suspected (std={grad:.1f}, mean={mean_val:.1f})")
        elif mean_val < 35:
            corner_info.append(f"{cname}: dark vignette/border (mean={mean_val:.1f})")
    
    # Check central lesion characteristics
    center_patch = arr[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
    center_std = np.std(center_patch)
    center_mean = np.mean(center_patch)
    
    # Look at red-green separation in center
    rg_diff = center_patch[:, :, 0].astype(float) - center_patch[:, :, 1].astype(float)
    erythematous_center = float(np.mean(rg_diff))
    
    print(f"File: {fname} | True: {item['original_condition_code']} | Pred: {item['predicted_binary_label']} ({item['mpox_probability']:.1%})")
    print(f"  Center: mean={center_mean:.1f}, std={center_std:.1f}, erythema (R-G)={erythematous_center:.1f}")
    if corner_info:
        print(f"  Corner Flags: {', '.join(corner_info)}")
    else:
        print("  Corner Flags: Clean borders (no overt text/vignette in corners)")
    print()
