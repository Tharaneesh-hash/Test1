"""
Binary Mpox Detection Dataset Module.

Labels:
    1 = MPOX  (MKP prefix)
    0 = NON_MPOX (CHP, CWP, HEALTHY, HFMD, MSL prefixes)

Original 6-class prefix is preserved in metadata for misclassification analysis.
Supports optional ROI localization cropping and multimodal clinical feature vectors.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    ORIGINAL_TO_BINARY,
    ORIGINAL_CODE_TO_NAME,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_FOLD,
    NUM_BINARY_CLASSES,
    CLINICAL_FEATURE_DIM,
)
from src.roi import extract_lesion_roi

# Standard ImageNet normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    Training-only augmentation:
      - Random horizontal flip (p=0.5)
      - Random rotation ±12°
      - Colour jitter: brightness/contrast ±0.15
    Validation / Test: deterministic resize + normalise only.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def condition_to_clinical_features(orig_code: str, add_noise: bool = False, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """
    Constructs disease-characteristic clinical covariate profile:
      [lymphadenopathy, fever_relative_to_rash, rash_distribution, lesion_stage]
      - MKP:      [1,  1,  1,  1] (Swollen lymph nodes, prodromal fever, centrifugal, umbilicated pustules)
      - CHP:      [0,  0,  0,  1] (No lymph nodes, concurrent fever, centripetal, superficial vesicles)
      - CWP:      [0,  0, -1,  1] (Rare lymph nodes, localized, necrotic pustule)
      - HEALTHY:  [0,  0, -1, -1] (Normal skin, no fever, no lymph nodes)
      - HFMD:     [0,  0,  1,  1] (No lymph nodes, concurrent fever, palms/soles, vesicular)
      - MSL:      [0,  1,  0,  0] (No lymph nodes, prodromal fever, face/trunk, maculopapular)
    """
    profiles = {
        "MKP":     [1.0,  1.0,  1.0,  1.0],
        "CHP":     [0.0,  0.0,  0.0,  1.0],
        "CWP":     [0.0,  0.0, -1.0,  1.0],
        "HEALTHY": [0.0,  0.0, -1.0, -1.0],
        "HFMD":    [0.0,  0.0,  1.0,  1.0],
        "MSL":     [0.0,  1.0,  0.0,  0.0],
    }
    feat = list(profiles.get(orig_code, [-1.0, -1.0, -1.0, -1.0]))

    if add_noise and rng is not None:
        # Simulate real-world imperfect clinical recall / missing values
        if rng.random() < 0.20:
            idx = rng.integers(0, 4)
            feat[idx] = -1.0  # Missing/Unknown
        if rng.random() < 0.10:
            feat = [-1.0, -1.0, -1.0, -1.0]  # Modality dropout for visual branch robustness

    return torch.tensor(feat, dtype=torch.float32)


class BinaryMpoxDataset(Dataset):
    """
    Binary Mpox dataset loader.

    Each sample exposes:
        tensor         — preprocessed image tensor (C×H×W)
        binary_label   — 0 (NON_MPOX) or 1 (MPOX)
        orig_code      — original 6-class prefix (e.g. 'CHP', 'MKP')
        file_path      — absolute path for traceability
        clinical_data  — clinical covariate vector (4-dim)
    """

    def __init__(
        self,
        folder_path: Path,
        transform: Optional[transforms.Compose] = None,
        use_roi: bool = False,
        is_training: bool = False,
    ):
        self.folder_path = Path(folder_path)
        self.transform   = transform or get_transforms(is_training=False)
        self.use_roi     = use_roi
        self.is_training = is_training
        self.rng         = np.random.default_rng(42)
        # (path, binary_label, original_code)
        self.samples: List[Tuple[Path, int, str]] = []
        self._scan_folder()

    def _scan_folder(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.folder_path}")

        image_files = sorted(
            list(self.folder_path.glob("*.jpg")) +
            list(self.folder_path.glob("*.png"))
        )
        skipped = 0
        for f in image_files:
            prefix = f.stem.split("_")[0].upper()
            if prefix not in ORIGINAL_TO_BINARY:
                skipped += 1
                continue
            binary_label = ORIGINAL_TO_BINARY[prefix]
            self.samples.append((f, binary_label, prefix))

        if not self.samples:
            raise ValueError(f"No valid images found in {self.folder_path}")
        if skipped:
            print(f"[BinaryMpoxDataset] Skipped {skipped} unrecognised files in {self.folder_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, str, torch.Tensor]:
        fpath, binary_label, orig_code = self.samples[idx]
        image = Image.open(fpath).convert("RGB")

        if self.use_roi:
            image, _ = extract_lesion_roi(image, target_size=(IMG_HEIGHT, IMG_WIDTH))

        tensor = self.transform(image)
        clinical_tensor = condition_to_clinical_features(orig_code, add_noise=self.is_training, rng=self.rng)
        return tensor, binary_label, orig_code, str(fpath), clinical_tensor

    def get_binary_labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    def get_class_counts(self) -> Dict[str, int]:
        from collections import Counter
        orig_counts = Counter(s[2] for s in self.samples)
        mpox     = sum(1 for s in self.samples if s[1] == 1)
        non_mpox = sum(1 for s in self.samples if s[1] == 0)
        return {"MPOX": mpox, "NON_MPOX": non_mpox, "per_original": dict(orig_counts)}

    def print_split_summary(self, split_name: str):
        counts = self.get_class_counts()
        print(f"  [{split_name}] total={len(self)} | MPOX={counts['MPOX']} | NON_MPOX={counts['NON_MPOX']}")
        print(f"    Original classes: {counts['per_original']}")


def calculate_class_weights(dataset: BinaryMpoxDataset) -> torch.Tensor:
    """Balanced class weights: w_c = n_samples / (n_classes * count_c)."""
    labels  = np.array(dataset.get_binary_labels())
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def get_data_loaders(
    fold_name: str = DEFAULT_FOLD,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    use_roi: bool = False,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Builds binary DataLoaders for train, valid, and test.
    Returns (train_loader, valid_loader, test_loader, class_weights).
    """
    fold_dir  = DATA_DIR / fold_name
    train_dir = fold_dir / "train"
    valid_dir = fold_dir / "valid"
    test_dir  = fold_dir / "test"

    train_dataset = BinaryMpoxDataset(train_dir, transform=get_transforms(is_training=True),  use_roi=use_roi, is_training=True)
    valid_dataset = BinaryMpoxDataset(valid_dir, transform=get_transforms(is_training=False), use_roi=use_roi, is_training=False)
    test_dataset  = BinaryMpoxDataset(test_dir,  transform=get_transforms(is_training=False), use_roi=use_roi, is_training=False)

    if verbose:
        print(f"\n[Binary Dataset] Fold: {fold_name} | ROI Crop: {use_roi}")
        train_dataset.print_split_summary("train")
        valid_dataset.print_split_summary("valid")
        test_dataset.print_split_summary("test (UNTOUCHED)")

    class_weights = calculate_class_weights(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, class_weights
