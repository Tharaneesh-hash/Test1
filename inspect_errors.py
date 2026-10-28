"""
inspect_errors.py — Detailed analysis of all 14 misclassified test images.
Examines true label, predicted label, predicted probability, error type,
and inspects whether Grad-CAM heatmaps highlight true skin lesions or shortcuts
(backgrounds, dark borders, rulers, watermarks, skin folds).
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.config import (
    BEST_BINARY_MODEL_PATH,
    DEFAULT_FOLD,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    ORIGINAL_CODE_TO_NAME,
    RESULTS_DIR,
)
from src.model import load_binary_model
from src.dataset import get_data_loaders
from src.gradcam import GradCAM

device = torch.device("cpu")
model = load_binary_model(BEST_BINARY_MODEL_PATH, device=device)
model.eval()

gradcam = GradCAM(model, device=device)

_, _, test_loader, _ = get_data_loaders(DEFAULT_FOLD, batch_size=1, verbose=False)

output_dir = RESULTS_DIR / "error_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

misclassified_items = []

for batch in test_loader:
    imgs = batch[0]
    labels = batch[1]
    orig_codes = batch[2]
    paths = batch[3]

    with torch.no_grad():
        probs = model.predict_proba(imgs)[0].cpu().numpy()

    pred = int(np.argmax(probs))
    true = int(labels[0])
    mpox_prob = float(probs[MPOX_CLASS_INDEX])
    conf = float(np.max(probs))
    fpath = Path(paths[0])
    fname = fpath.name
    orig_code = str(orig_codes[0])

    if pred != true:
        err_type = "False Negative (FN)" if (true == 1 and pred == 0) else "False Positive (FP)"

        # Generate Grad-CAM for this misclassified sample
        orig_arr, heat_arr, overlay_arr, pred_cls_idx, _ = gradcam.explain_image(
            fpath, target_class_idx=pred, alpha=0.45
        )

        # Save visual overlay
        overlay_img = Image.fromarray(overlay_arr)
        vis_save_path = output_dir / f"{fname.replace('.', '_')}_cam.png"
        overlay_img.save(vis_save_path)

        # Quantitative spatial analysis of Grad-CAM attention
        # Check border vs center attention (shortcut check)
        h, w, _ = heat_arr.shape
        # normalized heatmap in [0, 1]
        heat_gray = np.mean(heat_arr, axis=2) / 255.0

        # Define border margin (outer 15%)
        margin_y = int(h * 0.15)
        margin_x = int(w * 0.15)
        center_heat = heat_gray[margin_y:h-margin_y, margin_x:w-margin_x]
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[margin_y:h-margin_y, margin_x:w-margin_x] = False
        border_heat = heat_gray[border_mask]

        center_energy = float(np.mean(center_heat)) if center_heat.size > 0 else 0.0
        border_energy = float(np.mean(border_heat)) if border_heat.size > 0 else 0.0
        border_to_center_ratio = border_energy / (center_energy + 1e-6)

        # Check image properties (dimensions, aspect ratio, dark borders)
        pil_orig = Image.open(fpath)
        orig_w, orig_h = pil_orig.size
        aspect_ratio = orig_w / orig_h

        # Edge brightness (checking for black padding or vignette borders)
        img_np = np.array(pil_orig.convert("RGB"))
        edge_pixels = np.concatenate([
            img_np[:5, :, :].reshape(-1, 3),
            img_np[-5:, :, :].reshape(-1, 3),
            img_np[:, :5, :].reshape(-1, 3),
            img_np[:, -5:, :].reshape(-1, 3),
        ])
        edge_brightness = float(np.mean(edge_pixels))

        misclassified_items.append({
            "filename": fname,
            "filepath": str(fpath),
            "original_condition_code": orig_code,
            "original_condition_name": ORIGINAL_CODE_TO_NAME.get(orig_code, orig_code),
            "true_binary_label": "MPOX" if true == 1 else "NON_MPOX",
            "predicted_binary_label": "MPOX" if pred == 1 else "NON_MPOX",
            "mpox_probability": round(mpox_prob, 4),
            "predicted_confidence": round(conf, 4),
            "error_type": err_type,
            "aspect_ratio": round(aspect_ratio, 2),
            "dimensions": f"{orig_w}x{orig_h}",
            "edge_brightness": round(edge_brightness, 1),
            "border_to_center_ratio": round(border_to_center_ratio, 3),
            "cam_overlay_path": str(vis_save_path),
        })

# Sort items by error type, then by confidence
misclassified_items.sort(key=lambda x: (x["error_type"], -x["predicted_confidence"]))

summary_report = {
    "total_test_samples": 74,
    "total_misclassified": len(misclassified_items),
    "false_negatives_count": sum(1 for x in misclassified_items if "False Negative" in x["error_type"]),
    "false_positives_count": sum(1 for x in misclassified_items if "False Positive" in x["error_type"]),
    "items": misclassified_items
}

report_path = RESULTS_DIR / "metrics" / "misclassified_analysis.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(summary_report, f, indent=4)

print(f"Analysis saved -> {report_path}")
print(f"Total misclassified: {len(misclassified_items)}")
for i, item in enumerate(misclassified_items, 1):
    print(f"{i:02d}. [{item['error_type']}] {item['filename']} | True: {item['original_condition_code']} | Pred: {item['predicted_binary_label']} (Score: {item['mpox_probability']:.2%}) | Border/Center ratio: {item['border_to_center_ratio']}")
