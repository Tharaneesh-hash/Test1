"""
Binary Grad-CAM — compatible with BinaryVGG19, BinaryResNet34, BinaryEfficientNetB0.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from src.config import IMG_HEIGHT, IMG_WIDTH, MPOX_CLASS_INDEX
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for binary Mpox detection.
    Architecture-agnostic: uses get_final_conv_layer() from each model.
    """
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model  = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        if not hasattr(model, "get_final_conv_layer"):
            raise AttributeError("Model must implement get_final_conv_layer().")

        self.target_layer: nn.Module = model.get_final_conv_layer()
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
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
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates raw 2-D Grad-CAM heatmap.
        Returns: (heatmap_normalised, target_class_idx, class_prob)
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device)
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probs  = torch.softmax(logits, dim=1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        class_prob = float(probs[0, target_class_idx].item())
        score      = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Failed to capture gradients on target conv layer.")

        gradients         = self.activations.grad
        pooled_gradients  = torch.mean(gradients, dim=[0, 2, 3])
        activations       = self.activations[0].detach()

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(pooled_gradients):
            cam += w * activations[i]

        cam    = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        vmin, vmax = cam_np.min(), cam_np.max()
        if vmax - vmin > 1e-8:
            cam_np = (cam_np - vmin) / (vmax - vmin)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class_idx, class_prob

    def overlay_heatmap(
        self,
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap_name: str = "jet",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = pil_image.size
        hm_pil     = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        hm_norm    = np.array(hm_pil).astype(np.float32) / 255.0
        cmap       = plt.get_cmap(colormap_name)
        colored    = (cmap(hm_norm)[:, :, :3] * 255).astype(np.uint8)
        orig_arr   = np.array(pil_image.convert("RGB"))
        overlay    = np.clip((1 - alpha) * orig_arr + alpha * colored, 0, 255).astype(np.uint8)
        return orig_arr, colored, overlay

    def explain_image(
        self,
        image_input: Union[str, Path, Image.Image],
        target_class_idx: Optional[int] = None,
        alpha: float = 0.4,
        colormap_name: str = "jet",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        preprocess = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        tensor     = preprocess(pil_image).unsqueeze(0)
        heatmap_2d, tgt_idx, prob = self.generate_cam(tensor, target_class_idx)
        orig, heat, overlay = self.overlay_heatmap(pil_image, heatmap_2d, alpha, colormap_name)
        return orig, heat, overlay, tgt_idx, prob
