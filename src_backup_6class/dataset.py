import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME,
    NUM_CLASSES,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_FOLD
)

# Standard ImageNet normalization parameters (compatible with VGG19)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    Training includes mild data augmentations suitable for dermatological images.
    Validation/Test applies standard resize and ImageNet normalization.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class MpoxDataset(Dataset):
    """
    Dataset loader for Mpox Skin Lesion images organized by filename prefix:
    {CLASS_PREFIX}_{CASE_ID}_{IMAGE_ID}.jpg
    """
    def __init__(
        self,
        folder_path: Path,
        transform: Optional[transforms.Compose] = None
    ):
        self.folder_path = Path(folder_path)
        self.transform = transform or get_transforms(is_training=False)
        self.samples: List[Tuple[Path, int, str]] = []
        self.case_ids: List[str] = []

        self._scan_folder()

    def _scan_folder(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.folder_path}")

        image_files = sorted(list(self.folder_path.glob("*.jpg")) + list(self.folder_path.glob("*.png")))
        for f in image_files:
            prefix = f.stem.split("_")[0]
            if prefix in CLASS_CODES:
                class_idx = CLASS_CODES.index(prefix)
                # Parse patient/case identifier if present
                parts = f.stem.split("_")
                case_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else prefix
                self.samples.append((f, class_idx, prefix))
                self.case_ids.append(case_id)

        if not self.samples:
            raise ValueError(f"No valid images found in {self.folder_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        file_path, class_idx, prefix = self.samples[idx]
        image = Image.open(file_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, class_idx, str(file_path)

    def get_labels(self) -> List[int]:
        return [sample[1] for sample in self.samples]

    def get_class_counts(self) -> Dict[str, int]:
        counts = {code: 0 for code in CLASS_CODES}
        for _, _, prefix in self.samples:
            counts[prefix] += 1
        return counts


def calculate_class_weights(dataset: MpoxDataset) -> torch.Tensor:
    """
    Computes balanced class weights using scikit-learn formula:
    n_samples / (n_classes * np.bincount(y))
    Transparently handles class imbalance during training.
    """
    labels = np.array(dataset.get_labels())
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def get_data_loaders(
    fold_name: str = DEFAULT_FOLD,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Builds DataLoader objects for train, valid, and test sets of a specific fold.
    Returns (train_loader, valid_loader, test_loader, class_weights_tensor).
    """
    fold_dir = DATA_DIR / fold_name
    train_dir = fold_dir / "train"
    valid_dir = fold_dir / "valid"
    test_dir = fold_dir / "test"

    train_dataset = MpoxDataset(train_dir, transform=get_transforms(is_training=True))
    valid_dataset = MpoxDataset(valid_dir, transform=get_transforms(is_training=False))
    test_dataset = MpoxDataset(test_dir, transform=get_transforms(is_training=False))

    class_weights = calculate_class_weights(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, valid_loader, test_loader, class_weights
