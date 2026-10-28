"""
scripts/inspect_features_deep.py — Inspect specific visual features, lesion patterns,
and potential shortcut clues across the 14 misclassified images.
"""

from pathlib import Path
from PIL import Image
import numpy as np

def examine_images():
    paths = [
        # 5 False Negatives
        ("MKP_71_03.jpg", "FOLDS/fold1/test/MKP_71_03.jpg"),
        ("MKP_71_01.jpg", "FOLDS/fold1/test/MKP_71_01.jpg"),
        ("MKP_69_02.jpg", "FOLDS/fold1/test/MKP_69_02.jpg"),
        ("MKP_71_04.jpg", "FOLDS/fold1/test/MKP_71_04.jpg"),
        ("MKP_71_02.jpg", "FOLDS/fold1/test/MKP_71_02.jpg"),
        # 9 False Positives
        ("CHP_14_02.jpg", "FOLDS/fold1/test/CHP_14_02.jpg"),
        ("CHP_14_01.jpg", "FOLDS/fold1/test/CHP_14_01.jpg"),
        ("CHP_19_01.jpg", "FOLDS/fold1/test/CHP_19_01.jpg"),
        ("CHP_49_01.jpg", "FOLDS/fold1/test/CHP_49_01.jpg"),
        ("HEALTHY_79_01.jpg", "FOLDS/fold1/test/HEALTHY_79_01.jpg"),
        ("CHP_08_01.jpg", "FOLDS/fold1/test/CHP_08_01.jpg"),
        ("CWP_09_01.jpg", "FOLDS/fold1/test/CWP_09_01.jpg"),
        ("CWP_40_02.jpg", "FOLDS/fold1/test/CWP_40_02.jpg"),
        ("CHP_08_02.jpg", "FOLDS/fold1/test/CHP_08_02.jpg"),
    ]

    for fname, p in paths:
        im = Image.open(p)
        arr = np.array(im)
        # Check if there are pure white lines or text (e.g. watermark or text overlay)
        # Text usually has high local gradient and high luminance (>240)
        white_pixels = np.sum(arr > 240) / arr.size * 100
        # Check if there are pure black border bands (e.g. letterboxing)
        top_black = np.all(arr[:3, :, :] < 10)
        bot_black = np.all(arr[-3:, :, :] < 10)
        left_black = np.all(arr[:, :3, :] < 10)
        right_black = np.all(arr[:, -3:, :] < 10)
        
        # Check color spread
        std_rgb = [float(arr[:, :, c].std()) for c in range(3)]
        
        print(f"[{fname}] Size: {arr.shape[1]}x{arr.shape[0]} | Letterbox black: T={top_black}, B={bot_black}, L={left_black}, R={right_black}")
        print(f"  White pixel %: {white_pixels:.2f}% | RGB stds: {std_rgb}")

if __name__ == "__main__":
    examine_images()
