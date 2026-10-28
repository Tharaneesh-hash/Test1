"""
src/model.py — Advanced Deep Learning Architectures for Binary Mpox Detection.

Supported backbones and modules:
    1. VGG19                    (Baseline)
    2. ResNet34                 (Residual connections)
    3. EfficientNet-B0          (Fast mobile architecture)
    4. ConvNeXt-V2 / ConvNeXt-T (Modern pure-conv architecture)
    5. Swin Transformer V2 / T  (Hierarchical vision transformer)
    6. ArcFaceHead              (Deep Metric Learning for look-alike separation)
    7. MultimodalMpoxClassifier (Image visual embedding + clinical metadata fusion)
    8. BinaryEnsemble           (Soft-voting ensemble)
    9. TemperatureScaling       (Calibrated confidence estimation)
"""

import math
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import (
    NUM_BINARY_CLASSES,
    BINARY_VGG19_PATH,
    BINARY_RESNET34_PATH,
    BINARY_EFFNET_PATH,
    BINARY_CONVNEXT_PATH,
    BINARY_SWIN_PATH,
    BINARY_MULTIMODAL_PATH,
    BINARY_ARCFACE_PATH,
    BEST_BINARY_MODEL_PATH,
    CLASS_MAPPING_PATH,
    MODEL_CONFIG_PATH,
    BINARY_CLASS_NAMES,
    ORIGINAL_CODE_TO_NAME,
    CLINICAL_FEATURE_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. VGG19 Binary Classifier
# ─────────────────────────────────────────────────────────────────────────────
class BinaryVGG19(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        base = models.vgg19(weights=weights)
        self.features = base.features

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, NUM_BINARY_CLASSES),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # If backbone is completely frozen, skip autograd graph allocation for speed
        if not any(p.requires_grad for p in self.features.parameters()):
            with torch.no_grad():
                feat = self.features(x)
                feat = self.global_pool(feat)
        else:
            feat = self.features(x)
            feat = self.global_pool(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier_head[1:](feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.features[34]

    def unfreeze_block5(self):
        for i, layer in enumerate(self.features):
            if i >= 28:
                for p in layer.parameters():
                    p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 2. ResNet34 Binary Classifier
# ─────────────────────────────────────────────────────────────────────────────
class BinaryResNet34(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        if freeze_backbone:
            for p in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for param in p.parameters():
                    param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, NUM_BINARY_CLASSES),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.fc(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.layer4[-1].conv2

    def unfreeze_layer3(self):
        for p in self.layer3.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. EfficientNet-B0 Binary Classifier
# ─────────────────────────────────────────────────────────────────────────────
class BinaryEfficientNetB0(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        self.features = base.features
        self.avgpool  = base.avgpool

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, NUM_BINARY_CLASSES),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.features[8][0]

    def unfreeze_last_block(self):
        for i in [6, 7, 8]:
            for p in self.features[i].parameters():
                p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 4. ConvNeXt-Tiny Binary Classifier (Modern Pure-Conv Vision Model)
# ─────────────────────────────────────────────────────────────────────────────
class BinaryConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny backbone fine-tuned for high-accuracy binary Mpox recognition.
    Features 7x7 depthwise convolutions and inverted bottleneck design.
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = models.convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool  = base.avgpool

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        in_features = base.classifier[2].in_features  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, NUM_BINARY_CLASSES),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        # Final block in stage 7 of ConvNeXt-Tiny
        return self.features[-1][-1].block[0]

    def unfreeze_last_stage(self):
        for p in self.features[-1].parameters():
            p.requires_grad = True
        for p in self.features[-2].parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Swin Transformer V2 / Tiny Binary Classifier
# ─────────────────────────────────────────────────────────────────────────────
class BinarySwinTransformer(nn.Module):
    """
    Swin Transformer Tiny backbone with shifted window self-attention.
    Captures both localized lesion textures and macro cutaneous clustering.
    """
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        base = models.swin_t(weights=weights)
        self.features = base.features
        self.norm     = base.norm
        self.permute  = base.permute
        self.avgpool  = base.avgpool

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        in_features = base.head.in_features  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, NUM_BINARY_CLASSES),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.features[-1]

    def unfreeze_last_stage(self):
        for p in self.features[-1].parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Deep Metric Learning: ArcFace Head (Algorithm 4)
# ─────────────────────────────────────────────────────────────────────────────
class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin (ArcFace) Head for metric separation of Mpox
    from close morphological look-alikes (Chickenpox, Cowpox).
    """
    def __init__(self, in_features: int = 512, num_classes: int = NUM_BINARY_CLASSES,
                 scale: float = 16.0, margin: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, features: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize features and weights to hypersphere
        norm_x = F.normalize(features, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(norm_x, norm_w)

        if labels is None or not self.training:
            # During inference, return scaled cosine logits
            return self.scale * cosine

        # Angular margin calculation for training
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return self.scale * output


# ─────────────────────────────────────────────────────────────────────────────
# 7. Multimodal Clinical Metadata Fusion (Algorithm 5)
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalMpoxClassifier(nn.Module):
    """
    Combines deep image representation (from ResNet34, ConvNeXt, or VGG19)
    with structured patient clinical metadata:
      1. Swollen lymph nodes (yes=1, no=0, unknown=-1)
      2. Fever relative to rash (prodromal=1, concurrent=0, unknown=-1)
      3. Rash distribution (centrifugal=1, centripetal=0, unknown=-1)
      4. Lesion stage (vesicle/pustule=1, maculopapular=0, scab=2, unknown=-1)

    Handles missing/unknown metadata gracefully with a robust tabular embedding layer.
    """
    def __init__(self, visual_backbone: Optional[nn.Module] = None, feature_dim: int = 512):
        super().__init__()
        self.visual_backbone = visual_backbone or BinaryResNet34(pretrained=True, freeze_backbone=True)
        self.feature_dim = feature_dim

        # Tabular Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(CLINICAL_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Cross-Modal Fusion Head
        fused_dim = feature_dim + 64
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, NUM_BINARY_CLASSES),
        )

    def forward(self, images: torch.Tensor, clinical_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        vis_features = self.visual_backbone.extract_features(images)

        batch_size = images.size(0)
        device = images.device

        if clinical_data is None:
            # Default to neutral / unknown vector [-1, -1, -1, -1]
            clinical_data = torch.full((batch_size, CLINICAL_FEATURE_DIM), -1.0, device=device)
        else:
            clinical_data = clinical_data.to(device)

        clin_features = self.clinical_encoder(clinical_data)
        fused = torch.cat([vis_features, clin_features], dim=1)
        return self.fusion_head(fused)

    def predict_proba(self, images: torch.Tensor, clinical_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(images, clinical_data)
        return F.softmax(logits, dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.visual_backbone.get_final_conv_layer()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Ensemble Model
# ─────────────────────────────────────────────────────────────────────────────
class BinaryEnsemble(nn.Module):
    def __init__(self, model_a: nn.Module, model_b: nn.Module,
                 weight_a: float = 0.5, weight_b: float = 0.5):
        super().__init__()
        self.model_a  = model_a
        self.model_b  = model_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        pa = self.model_a.predict_proba(x)
        pb = self.model_b.predict_proba(x)
        return self.weight_a * pa + self.weight_b * pb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.log(probs + 1e-8)

    def get_final_conv_layer(self) -> nn.Module:
        return self.model_a.get_final_conv_layer()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Temperature Scaling for Calibrated Probability Estimation (Algorithm 6)
# ─────────────────────────────────────────────────────────────────────────────
class ModelWithTemperature(nn.Module):
    """
    Applies learned temperature scaling to model logits:
        logits' = logits / T
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        logits = self.model(*args, **kwargs)
        return logits / self.temperature

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        scaled_logits = self.forward(*args, **kwargs)
        return F.softmax(scaled_logits, dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.model.get_final_conv_layer()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture Registry & Checkpoint Loaders
# ─────────────────────────────────────────────────────────────────────────────
_ARCH_REGISTRY = {
    "BinaryVGG19":              BinaryVGG19,
    "BinaryResNet34":           BinaryResNet34,
    "BinaryEfficientNetB0":     BinaryEfficientNetB0,
    "BinaryConvNeXt":           BinaryConvNeXt,
    "BinarySwinTransformer":    BinarySwinTransformer,
    "MultimodalMpoxClassifier": MultimodalMpoxClassifier,
}


def save_binary_model(model: nn.Module, save_path: Path, arch_name: str):
    """Saves model checkpoint with metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch_name, "state_dict": model.state_dict()}, save_path)

    mapping = {
        "binary_class_names": BINARY_CLASS_NAMES,
        "mpox_index": 1,
        "non_mpox_index": 0,
        "original_code_to_name": ORIGINAL_CODE_TO_NAME,
    }
    with open(CLASS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    cfg = {"architecture": arch_name, "num_classes": NUM_BINARY_CLASSES, "input_shape": [224, 224, 3]}
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


def load_binary_model(checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Loads a binary model checkpoint into the appropriate architecture."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training first.")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "arch" in ckpt:
        arch_name = ckpt["arch"]
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
        name = checkpoint_path.stem.lower()
        if "convnext" in name:
            arch_name = "BinaryConvNeXt"
        elif "swin" in name:
            arch_name = "BinarySwinTransformer"
        elif "vgg" in name:
            arch_name = "BinaryVGG19"
        elif "resnet" in name:
            arch_name = "BinaryResNet34"
        elif "multimodal" in name:
            arch_name = "MultimodalMpoxClassifier"
        elif "effic" in name:
            arch_name = "BinaryEfficientNetB0"
        else:
            raise ValueError(f"Cannot infer architecture from path: {checkpoint_path}")

    if arch_name not in _ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch_name}")

    model = _ARCH_REGISTRY[arch_name](pretrained=False, freeze_backbone=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_best_model(device: torch.device = torch.device("cpu")) -> nn.Module:
    """Loads the canonical best model selected on validation data."""
    return load_binary_model(BEST_BINARY_MODEL_PATH, device=device)
