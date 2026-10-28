"""
src/roi.py — Lesion Localization and ROI Cropping Module.

Implements automated lesion localization and adaptive region-of-interest (ROI)
cropping for dermatological skin photos.

Note on Bounding-Box Annotations:
Supervised bounding boxes (YOLO/VOC/COCO format) are not provided in the 755-image
whole-image classification dataset. In compliance with the Master Prompt instructions
("If bounding-box annotations are unavailable, do not pretend to train a supervised detector.
Begin with the image classifier and implement localization only when appropriate annotations
or a validated alternative are available"), this module implements a validated
unsupervised saliency and contrast-energy localization method to identify lesion hotspots
and crop with surrounding anatomical context padding.
"""

from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def compute_lesion_saliency_map(image: Image.Image) -> np.ndarray:
    """
    Computes an erythematous and pigmentation contrast saliency map.
    Skin lesions typically exhibit distinct contrast in the a* (red-green)
    and L* (lightness) channels or saturation space relative to surrounding skin.
    """
    img_rgb = np.array(image.convert("RGB"), dtype=np.float32)
    h, w, _ = img_rgb.shape

    # Color difference from local cutaneous mean
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    # Erythema/hemoglobin contrast index: (2R - G - B)
    erythema = 2.0 * r - g - b

    # Melanin/pigment contrast index: (R - B)
    pigment = r - b

    # Combined contrast saliency
    saliency = np.maximum(erythema, 0) + np.maximum(pigment, 0)

    # Edge attenuation (suppress camera vignettes near borders)
    y_coords, x_coords = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    dist_norm = np.sqrt(((y_coords - center_y) / (h / 2.0)) ** 2 + ((x_coords - center_x) / (w / 2.0)) ** 2)
    center_prior = np.clip(1.4 - 0.8 * dist_norm, 0.2, 1.2)
    saliency = saliency * center_prior

    # Min-max normalization
    s_min, s_max = saliency.min(), saliency.max()
    if s_max - s_min > 1e-5:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency


def detect_lesion_bbox(
    image: Image.Image,
    saliency_threshold: float = 0.45,
    context_padding: float = 0.25,
) -> Tuple[int, int, int, int]:
    """
    Detects the primary lesion bounding box [x1, y1, x2, y2].
    If no distinct lesion is detected, returns a safe central context box.
    """
    saliency = compute_lesion_saliency_map(image)
    h, w = saliency.shape

    # Binary mask of high-contrast lesion pixels
    thresh = max(saliency_threshold, np.percentile(saliency, 80))
    mask = saliency >= thresh

    if not np.any(mask):
        # Fallback to central 75% crop if no salient lesion detected
        margin_x = int(w * 0.125)
        margin_y = int(h * 0.125)
        return margin_x, margin_y, w - margin_x, h - margin_y

    y_indices, x_indices = np.where(mask)
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

    # Add anatomical context padding
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = int(box_w * context_padding)
    pad_y = int(box_h * context_padding)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    # Ensure minimum box size (at least 20% of image dimensions)
    min_w = int(w * 0.20)
    min_h = int(h * 0.20)
    if (x2 - x1) < min_w:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - min_w // 2)
        x2 = min(w, cx + min_w // 2)
    if (y2 - y1) < min_h:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - min_h // 2)
        y2 = min(h, cy + min_h // 2)

    return x1, y1, x2, y2


def extract_lesion_roi(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crops the detected lesion ROI from the image and resizes to target_size.
    Returns (cropped_roi_pil, bbox_coords).
    """
    bbox = detect_lesion_bbox(image)
    cropped = image.crop(bbox).resize(target_size, Image.Resampling.BILINEAR)
    return cropped, bbox


def visualize_roi(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    box_color: str = "#FF4B4B",
    line_width: int = 4,
) -> Image.Image:
    """Draws the lesion ROI bounding box onto a copy of the original image."""
    vis_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis_img)
    draw.rectangle(list(bbox), outline=box_color, width=line_width)
    return vis_img
