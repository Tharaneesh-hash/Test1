from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from src.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME
)
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for VGG19.
    Visualizes influential spatial regions in skin lesion images for a chosen diagnostic class.
    
    Medical Context:
    Grad-CAM highlights regions of the input that most strongly contributed to the model's 
    class activation score. It represents feature attribution, NOT clinical validation.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Target layer defaults to VGG19's final conv layer (block5_conv4 / features[34])
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_final_conv_layer"):
            self.target_layer = model.get_final_conv_layer()
        elif hasattr(model, "features"):
            self.target_layer = model.features[34]
        else:
            raise ValueError("Target convolutional layer could not be determined.")

        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            # Ensure activation tensor tracks gradient even if backbone conv weights are frozen
            if not output.requires_grad:
                output.requires_grad_(True)
            output.retain_grad()

        h = self.target_layer.register_forward_hook(forward_hook)
        self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates raw Grad-CAM activation heatmap for a given input tensor.
        Returns: (heatmap_2d, target_class_idx, class_score)
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        class_prob = float(probs[0, target_class_idx].item())

        # Backward pass for gradients of selected class score w.r.t. feature maps
        score.backward(retain_graph=True)

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Failed to capture gradients or activations on target conv layer.")

        gradients = self.activations.grad

        # Channel-wise Global Average Pooling of gradients
        # alpha_k = (1/Z) * sum_i sum_j (d y_c / d A^k_ij)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weighted combination of feature maps
        activations = self.activations[0].detach()
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, weight in enumerate(pooled_gradients):
            cam += weight * activations[i]

        # Apply ReLU to retain only features with a positive influence on the target class
        cam = torch.relu(cam)

        # Heatmap normalization to [0, 1]
        cam_np = cam.cpu().numpy()
        cam_min = np.min(cam_np)
        cam_max = np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class_idx, class_prob

    def overlay_heatmap(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap_name: str = "jet"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resizes heatmap to original image dimensions and creates visual overlay.
        Returns: (original_rgb_uint8, colored_heatmap_uint8, overlay_rgb_uint8)
        """
        orig_w, orig_h = pil_image.size
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
        heatmap_norm = np.array(heatmap_resized).astype(np.float32) / 255.0

        cmap = plt.get_cmap(colormap_name)
        colored_heatmap = cmap(heatmap_norm)[:, :, :3]
        colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

        orig_array = np.array(pil_image.convert("RGB"))
        overlay = (1.0 - alpha) * orig_array + alpha * colored_heatmap_uint8
        overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)

        return orig_array, colored_heatmap_uint8, overlay_uint8

    def explain_image(
        self,
        image_input: Union[str, Path, Image.Image],
        target_class_idx: Optional[int] = None,
        alpha: float = 0.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        """
        End-to-end explainability pipeline:
        Takes image path or PIL Image -> preprocesses -> generates CAM -> generates overlay.
        Returns: (original_image, colored_heatmap, overlay, target_class_idx, class_prob)
        """
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        preprocess = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        tensor = preprocess(pil_image).unsqueeze(0)

        heatmap_2d, target_idx, prob = self.generate_cam(tensor, target_class_idx=target_class_idx)
        orig_img, heat_img, overlay_img = self.overlay_heatmap(pil_image, heatmap_2d, alpha=alpha)

        return orig_img, heat_img, overlay_img, target_idx, prob
