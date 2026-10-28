import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import (
    NUM_CLASSES,
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME,
    DENSE_UNITS,
    DROPOUT_RATE,
    BEST_MODEL_PATH,
    RESNET_MODEL_PATH,
    CLASS_MAPPING_PATH,
    MODEL_CONFIG_PATH
)


class VGG19MpoxClassifier(nn.Module):
    """
    VGG19 Transfer Learning Architecture for Mpox & Differential Skin Lesion Recognition.
    Architecture:
    1. Pretrained VGG19 convolutional backbone (features)
    2. Global Average Pooling (AdaptiveAvgPool2d to 1x1)
    3. Fully connected Dense layer (512 -> 256) with ReLU
    4. Dropout layer for regularization (rate = 0.4)
    5. Softmax-ready output projection (256 -> 6 classes)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dense_units: int = DENSE_UNITS,
        dropout_rate: float = DROPOUT_RATE,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # 1. Pretrained VGG19 backbone
        if pretrained:
            weights = models.VGG19_Weights.DEFAULT
            base_vgg = models.vgg19(weights=weights)
        else:
            base_vgg = models.vgg19(weights=None)

        self.features = base_vgg.features

        # 2. Freeze convolutional backbone initially
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # 3. Custom Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw class logits (unscaled scores).
        """
        feats = self.features(x)
        pooled = self.global_pool(feats)
        logits = self.classifier_head(pooled)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized class probabilities via Softmax.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        """
        Returns the final convolutional layer of VGG19 (features[34] / block5_conv4).
        Used by the Grad-CAM explainability module.
        """
        return self.features[34]

    def unfreeze_block5(self):
        """
        Unfreezes the final convolutional block (block 5: layers 28 to 36)
        for stage-2 transfer learning fine-tuning.
        """
        for i, layer in enumerate(self.features):
            if i >= 28:
                for param in layer.parameters():
                    param.requires_grad = True


class ResNetMpoxClassifier(nn.Module):
    """
    ResNet34 Architecture with Residual Connections for High Mpox Sensitivity (85.19%).
    Uses Layer 3 + Layer 4 fine-tuning and BatchNorm classification head.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_final_conv_layer(self) -> nn.Module:
        return self.layer4[-1].conv2


class EnsembleMpoxClassifier(nn.Module):
    """
    Weighted Soft-Voting Ensemble Algorithm fusing VGG19 and ResNet34.
    Test Accuracy: 78.38% | Macro F1: 77.93% | Mpox Sensitivity: 77.78%
    """
    def __init__(
        self,
        vgg_model: VGG19MpoxClassifier,
        resnet_model: ResNetMpoxClassifier,
        vgg_weight: float = 0.7,
        resnet_weight: float = 0.3
    ):
        super().__init__()
        self.vgg = vgg_model
        self.resnet = resnet_model
        self.vgg_weight = vgg_weight
        self.resnet_weight = resnet_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_vgg = F.softmax(self.vgg(x), dim=1)
        p_res = F.softmax(self.resnet(x), dim=1)
        blended = self.vgg_weight * p_vgg + self.resnet_weight * p_res
        return torch.log(blended + 1e-8)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        p_vgg = F.softmax(self.vgg(x), dim=1)
        p_res = F.softmax(self.resnet(x), dim=1)
        return self.vgg_weight * p_vgg + self.resnet_weight * p_res

    def get_final_conv_layer(self) -> nn.Module:
        return self.vgg.get_final_conv_layer()


def save_model_artifacts(
    model: VGG19MpoxClassifier,
    save_path: Path = BEST_MODEL_PATH,
    config_path: Path = MODEL_CONFIG_PATH,
    mapping_path: Path = CLASS_MAPPING_PATH
):
    """
    Saves model checkpoint along with class mapping and configuration metadata.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model weights & state dictionary
    torch.save(model.state_dict(), save_path)

    # Save class mapping
    mapping = {
        "class_codes": CLASS_CODES,
        "class_names": CLASS_NAMES,
        "code_to_name": CODE_TO_NAME,
        "num_classes": NUM_CLASSES,
        "class_to_idx": {code: i for i, code in enumerate(CLASS_CODES)},
        "idx_to_class": {i: code for i, code in enumerate(CLASS_CODES)},
        "idx_to_name": {i: CODE_TO_NAME[code] for i, code in enumerate(CLASS_CODES)}
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)

    # Save architecture configuration
    config = {
        "architecture": "VGG19_TransferLearning",
        "backbone": "vgg19_imagenet",
        "num_classes": model.num_classes,
        "dense_units": model.dense_units,
        "dropout_rate": model.dropout_rate,
        "input_shape": [224, 224, 3],
        "final_conv_layer": "features[34] (block5_conv4)",
        "framework": "PyTorch",
        "version": torch.__version__
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def load_model(
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: torch.device = torch.device("cpu")
) -> VGG19MpoxClassifier:
    """
    Loads model architecture and instantiates weights from checkpoint.
    Dynamically aligns dense units and num classes with saved weights.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Please train the model first by running src/train.py"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Infer architecture dimensions from checkpoint
    if "classifier_head.1.weight" in state_dict:
        dense_units = state_dict["classifier_head.1.weight"].shape[0]
    else:
        dense_units = DENSE_UNITS

    if "classifier_head.4.weight" in state_dict:
        num_classes = state_dict["classifier_head.4.weight"].shape[0]
    else:
        num_classes = NUM_CLASSES

    model = VGG19MpoxClassifier(
        num_classes=num_classes,
        dense_units=dense_units,
        dropout_rate=DROPOUT_RATE,
        pretrained=False,
        freeze_backbone=False
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_resnet_model(
    checkpoint_path: Path = RESNET_MODEL_PATH,
    device: torch.device = torch.device("cpu")
) -> ResNetMpoxClassifier:
    """
    Loads pretrained fine-tuned ResNet34 model.
    """
    model = ResNetMpoxClassifier(num_classes=NUM_CLASSES, pretrained=False)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_ensemble_model(
    vgg_path: Path = BEST_MODEL_PATH,
    resnet_path: Path = RESNET_MODEL_PATH,
    device: torch.device = torch.device("cpu"),
    vgg_weight: float = 0.7,
    resnet_weight: float = 0.3
) -> EnsembleMpoxClassifier:
    """
    Loads weighted soft-voting Ensemble combining VGG19 and ResNet34.
    """
    vgg = load_model(vgg_path, device=device)
    resnet = load_resnet_model(resnet_path, device=device)
    ensemble = EnsembleMpoxClassifier(vgg, resnet, vgg_weight=vgg_weight, resnet_weight=resnet_weight)
    ensemble.to(device)
    ensemble.eval()
    return ensemble
