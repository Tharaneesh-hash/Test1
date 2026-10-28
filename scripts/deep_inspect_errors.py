"""
scripts/deep_inspect_errors.py — Deep qualitative and quantitative visual analysis
of all 14 misclassified images to detect shortcuts (backgrounds, watermarks, borders,
lighting artifacts) vs genuine lesion morphology using pure PIL + numpy.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image

def analyze():
    json_path = Path("results/metrics/misclassified_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("DETAILED ERROR INSPECTION ACROSS 14 MISCLASSIFIED CASES")
    print("=" * 80)

    for i, item in enumerate(data["items"], 1):
        fpath = Path(item["filepath"])
        pil_img = Image.open(fpath).convert("RGB")
        w, h = pil_img.size
        img_np = np.array(pil_img, dtype=np.float32)

        # Grayscale
        gray = 0.2989 * img_np[:, :, 0] + 0.5870 * img_np[:, :, 1] + 0.1140 * img_np[:, :, 2]
        
        # Simple sharpness measure: gradient magnitude variance
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        sharpness = float(grad_mag.mean())
        
        # Contrast: standard deviation of grayscale
        contrast = float(gray.std())
        mean_brightness = float(gray.mean())
        
        # Color saturation approximation: (max(R,G,B) - min(R,G,B)) / (max(R,G,B) + 1e-5)
        c_max = np.max(img_np, axis=2)
        c_min = np.min(img_np, axis=2)
        saturation = float(np.mean((c_max - c_min) / (c_max + 1e-5)) * 100)
        
        # Check corner statistics (corner regions: 10% on each corner)
        cw, ch = max(5, int(w * 0.1)), max(5, int(h * 0.1))
        corners = [
            gray[:ch, :cw],
            gray[:ch, -cw:],
            gray[-ch:, :cw],
            gray[-ch:, -cw:]
        ]
        corner_means = [float(c.mean()) for c in corners]
        corner_stds = [float(c.std()) for c in corners]

        # Read the CAM overlay to check peak activation coordinates
        cam_p = Path(item["cam_overlay_path"])
        peak_loc = "Unknown"
        if cam_p.exists():
            cam_pil = Image.open(cam_p).convert("RGB")
            cam_np = np.array(cam_pil, dtype=np.float32)
            # In jet colormap, high activation is red (R high, B low)
            heat_est = cam_np[:, :, 0] - cam_np[:, :, 2]
            max_idx = np.unravel_index(np.argmax(heat_est), heat_est.shape)
            peak_y_pct = round((max_idx[0] / h) * 100, 1)
            peak_x_pct = round((max_idx[1] / w) * 100, 1)
            peak_loc = f"({peak_x_pct}%, {peak_y_pct}%)"

        print(f"\n[{i:02d}] {item['filename']} | {item['error_type']}")
        print(f"     True Condition: {item['original_condition_code']} ({item['original_condition_name']})")
        print(f"     Predicted: {item['predicted_binary_label']} | Mpox Score: {item['mpox_probability']:.2%}")
        print(f"     Dimensions: {w}x{h} | Sharpness: {sharpness:.1f} | Contrast: {contrast:.1f} | Brightness: {mean_brightness:.1f}")
        print(f"     Saturation: {saturation:.1f}% | Edge Brightness: {item['edge_brightness']}")
        print(f"     Corner Means: {[round(m, 1) for m in corner_means]} | Corner Stds: {[round(s, 1) for s in corner_stds]}")
        print(f"     Border/Center Ratio: {item['border_to_center_ratio']} | CAM Peak Location: {peak_loc}")

if __name__ == "__main__":
    analyze()
