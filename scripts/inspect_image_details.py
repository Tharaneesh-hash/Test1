"""
scripts/inspect_image_details.py — Detailed examination of visual contents,
histograms, border artifacts, and features across the 14 misclassified images.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import json

def main():
    json_path = Path("results/metrics/misclassified_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data["items"]:
        fpath = Path(item["filepath"])
        im = Image.open(fpath).convert("RGB")
        arr = np.array(im, dtype=np.float32)
        h, w, _ = arr.shape
        
        # Check border vs center luminance
        top = arr[:10, :, :].mean()
        bot = arr[-10:, :, :].mean()
        left = arr[:, :10, :].mean()
        right = arr[:, -10:, :].mean()
        center = arr[h//4:3*h//4, w//4:3*w//4, :].mean()
        
        # Check pure black or white borders (padding/frames)
        pure_black_edge_pct = np.mean(arr[:5, :, :] < 10) * 100
        pure_white_edge_pct = np.mean(arr[:5, :, :] > 245) * 100
        
        # Check color dominance (R, G, B)
        mean_r, mean_g, mean_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
        
        # Estimate erythema / redness ratio
        # Erythema index: (R - G) / (R + G + 1e-5)
        erythema = ((arr[:, :, 0] - arr[:, :, 1]) / (arr[:, :, 0] + arr[:, :, 1] + 1e-5)).mean()
        
        # Detect if there are extreme localized bright spots (e.g. white borders, flash glare, or text)
        high_bright_pct = np.mean(arr > 245) * 100
        low_dark_pct = np.mean(arr < 15) * 100

        print(f"File: {item['filename']} [{item['error_type']}]")
        print(f"  Condition: {item['original_condition_code']} | Score: {item['mpox_probability']:.2%}")
        print(f"  Edge lum: T={top:.1f}, B={bot:.1f}, L={left:.1f}, R={right:.1f} vs Center={center:.1f}")
        print(f"  RGB: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f} | Erythema Index: {erythema:.3f}")
        print(f"  Pure Black Edge %: {pure_black_edge_pct:.1f}% | High Bright %: {high_bright_pct:.1f}% | Low Dark %: {low_dark_pct:.1f}%")
        print()

if __name__ == "__main__":
    main()
