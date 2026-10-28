"""
tests/test_all_phases.py — Comprehensive Test Suite for High-Accuracy Mpox Detection.

Covers:
  1. Binary label mapping
  2. Image preprocessing
  3. EfficientNet-B0 output dimensions
  4. VGG19 binary output dimensions
  5. ResNet34 binary output dimensions
  6. Valid binary probability bounds & sums
  7. Grad-CAM attribution generation
  8. Uncertainty — high confidence MPOX
  9. Uncertainty — high confidence NON_MPOX
  10. Uncertainty — low confidence -> INCONCLUSIVE_REFER_FOR_PCR
  11. Uncertainty — invalid / corrupted input handling
  12. Checkpoint save/load round-trip
  13. Missing checkpoint error handling
  14. ConvNeXt-Tiny output shape & feature embedding
  15. Swin Transformer output shape
  16. ArcFace angular metric loss & margin calculation
  17. Multimodal clinical metadata fusion & missing covariate handling
  18. Saliency Lesion ROI localization & bounding box extraction
  19. Conformal prediction set calibration & selective classification
"""

import json
import math
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.config import (
    NUM_BINARY_CLASSES,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    BINARY_CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    STATE_MPOX_SUSPECTED,
    STATE_MPOX_NOT_DETECTED,
    STATE_INCONCLUSIVE,
    STATE_INVALID_IMAGE,
    ORIGINAL_TO_BINARY,
)
from src.dataset import get_transforms, BinaryMpoxDataset
from src.model import (
    BinaryVGG19,
    BinaryResNet34,
    BinaryEfficientNetB0,
    BinaryConvNeXt,
    BinarySwinTransformer,
    ArcFaceHead,
    MultimodalMpoxClassifier,
    save_binary_model,
    load_binary_model,
)
from src.gradcam import GradCAM
from src.uncertainty import assess_prediction_uncertainty
from src.roi import detect_lesion_bbox, extract_lesion_roi
from src.conformal import ConformalPredictor, SelectiveClassifier


class TestBinaryMpoxProject:
    """Full Automated Test Suite for Mpox Screening System."""

    @pytest.fixture(scope="class")
    def sample_image(self, tmp_path_factory):
        fn = tmp_path_factory.mktemp("data") / "MKP_01_01.jpg"
        img = Image.new("RGB", (300, 300), color=(180, 100, 90))
        img.save(fn)
        return fn

    @pytest.fixture(scope="class")
    def effnet_model(self):
        m = BinaryEfficientNetB0(pretrained=False, freeze_backbone=True)
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def vgg_model(self):
        m = BinaryVGG19(pretrained=False, freeze_backbone=True)
        m.eval()
        return m

    # ── 1. Binary label mapping ───────────────────────────────────────────────
    def test_01_binary_label_mapping(self):
        assert ORIGINAL_TO_BINARY["MKP"]     == 1, "MPOX must map to 1"
        assert ORIGINAL_TO_BINARY["CHP"]     == 0
        assert ORIGINAL_TO_BINARY["CWP"]     == 0
        assert ORIGINAL_TO_BINARY["HEALTHY"] == 0
        assert ORIGINAL_TO_BINARY["HFMD"]    == 0
        assert ORIGINAL_TO_BINARY["MSL"]     == 0
        assert NUM_BINARY_CLASSES == 2
        assert MPOX_CLASS_INDEX    == 1
        assert NON_MPOX_CLASS_INDEX == 0

    # ── 2. Image preprocessing ────────────────────────────────────────────────
    def test_02_image_preprocessing(self, sample_image):
        tfm    = get_transforms(is_training=False)
        pil    = Image.open(sample_image).convert("RGB")
        tensor = tfm(pil)
        assert tensor.shape  == (3, IMG_HEIGHT, IMG_WIDTH)
        assert tensor.dtype  == torch.float32
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()

    # ── 3. EfficientNet-B0 output shape ──────────────────────────────────────
    def test_03_efficientnet_output_shape(self, effnet_model):
        x = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            out = effnet_model(x)
        assert out.shape == (2, NUM_BINARY_CLASSES)

    # ── 4. VGG19 binary output shape ─────────────────────────────────────────
    def test_04_vgg19_output_shape(self, vgg_model):
        x = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            out = vgg_model(x)
        assert out.shape == (2, NUM_BINARY_CLASSES)

    # ── 5. ResNet34 binary output shape ──────────────────────────────────────
    def test_05_resnet34_output_shape(self):
        m = BinaryResNet34(pretrained=False, freeze_backbone=True)
        m.eval()
        x = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, NUM_BINARY_CLASSES)

    # ── 6. Probability validity ───────────────────────────────────────────────
    def test_06_valid_binary_probabilities(self, effnet_model):
        x    = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
        prbs = effnet_model.predict_proba(x).detach().numpy()[0]
        assert len(prbs) == NUM_BINARY_CLASSES
        assert np.all(prbs >= 0.0)
        assert np.all(prbs <= 1.0 + 1e-5)
        assert abs(prbs.sum() - 1.0) < 1e-4

    # ── 7. Grad-CAM generation ────────────────────────────────────────────────
    def test_07_gradcam_generation(self, effnet_model, sample_image):
        gcam = GradCAM(effnet_model)
        orig, heat, overlay, pred_idx, prob = gcam.explain_image(sample_image, target_class_idx=MPOX_CLASS_INDEX)
        assert orig.shape[2]    == 3
        assert heat.shape[2]    == 3
        assert overlay.shape[2] == 3
        assert orig.shape[:2]   == heat.shape[:2] == overlay.shape[:2]
        assert not np.isnan(heat).any()
        assert pred_idx in [0, 1]
        gcam.remove_hooks()

    # ── 8. Uncertainty — high confidence MPOX ────────────────────────────────
    def test_08_uncertainty_mpox_prediction(self):
        probs = [0.05, 0.95]   # NON_MPOX=5%, MPOX=95%
        res   = assess_prediction_uncertainty(probs, threshold=0.70)
        assert res["decision_state"]     == STATE_MPOX_SUSPECTED
        assert res["predicted_class"]    == "MPOX"
        assert res["mpox_probability"]   == pytest.approx(0.95, abs=1e-4)
        assert res["review_recommended"] == False

    # ── 9. Uncertainty — high confidence NON_MPOX ────────────────────────────
    def test_09_uncertainty_non_mpox_prediction(self):
        probs = [0.92, 0.08]
        res   = assess_prediction_uncertainty(probs, threshold=0.70)
        assert res["decision_state"]  == STATE_MPOX_NOT_DETECTED
        assert res["predicted_class"] == "NON_MPOX"
        assert res["review_recommended"] == False

    # ── 10. Uncertainty — low confidence -> INCONCLUSIVE ─────────────────────
    def test_10_uncertainty_review_required(self):
        probs = [0.52, 0.48]
        res   = assess_prediction_uncertainty(probs, threshold=0.70)
        assert res["decision_state"]     == STATE_INCONCLUSIVE
        assert res["review_recommended"] == True

    # ── 11. Uncertainty — invalid input ──────────────────────────────────────
    def test_11_invalid_input_handling(self):
        res_none = assess_prediction_uncertainty(None, is_valid_input=False)
        assert res_none["decision_state"] == STATE_INVALID_IMAGE

        res_nan = assess_prediction_uncertainty([float("nan"), 0.5])
        assert res_nan["decision_state"]  == STATE_INVALID_IMAGE

    # ── 12. Checkpoint save/load round-trip ──────────────────────────────────
    def test_12_checkpoint_save_load(self, effnet_model, tmp_path):
        ckpt_file = tmp_path / "test_effnet.pt"
        save_binary_model(effnet_model, ckpt_file, "BinaryEfficientNetB0")
        assert ckpt_file.exists()

        loaded = load_binary_model(ckpt_file)
        assert isinstance(loaded, BinaryEfficientNetB0)
        x = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            out = loaded.predict_proba(x)
        assert out.shape == (1, NUM_BINARY_CLASSES)

    # ── 13. Missing checkpoint raises FileNotFoundError ───────────────────────
    def test_13_missing_checkpoint_error(self, tmp_path):
        fake = tmp_path / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            load_binary_model(fake)

    # ── 14. ConvNeXt-Tiny output shape & feature extraction ──────────────────
    def test_14_convnext_output_shape(self):
        m = BinaryConvNeXt(pretrained=False, freeze_backbone=True)
        m.eval()
        x = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            out = m(x)
            feat = m.extract_features(x)
        assert out.shape == (2, NUM_BINARY_CLASSES)
        assert feat.shape == (2, 768)

    # ── 15. Swin Transformer output shape ────────────────────────────────────
    def test_15_swin_transformer_output_shape(self):
        m = BinarySwinTransformer(pretrained=False, freeze_backbone=True)
        m.eval()
        x = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            out = m(x)
            feat = m.extract_features(x)
        assert out.shape == (1, NUM_BINARY_CLASSES)
        assert feat.shape == (1, 768)

    # ── 16. ArcFace Metric Learning Loss Calculation ─────────────────────────
    def test_16_arcface_head(self):
        head = ArcFaceHead(in_features=256, num_classes=2, scale=16.0, margin=0.35)
        head.train()
        feat = torch.randn(4, 256)
        labels = torch.tensor([0, 1, 0, 1])
        logits = head(feat, labels)
        assert logits.shape == (4, 2)
        assert not torch.isnan(logits).any()

        # Inference mode
        head.eval()
        with torch.no_grad():
            eval_logits = head(feat)
        assert eval_logits.shape == (4, 2)

    # ── 17. Multimodal Clinical Metadata Fusion ──────────────────────────────
    def test_17_multimodal_clinical_fusion(self):
        model = MultimodalMpoxClassifier(visual_backbone=BinaryResNet34(pretrained=False, freeze_backbone=True))
        model.eval()

        img = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH)
        clin_complete = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

        with torch.no_grad():
            out_with_meta = model(img, clin_complete)
            # Test missing metadata handling
            out_missing = model(img, None)

        assert out_with_meta.shape == (2, NUM_BINARY_CLASSES)
        assert out_missing.shape == (2, NUM_BINARY_CLASSES)
        assert not torch.isnan(out_with_meta).any()
        assert not torch.isnan(out_missing).any()

    # ── 18. Lesion Saliency ROI Localization ─────────────────────────────────
    def test_18_lesion_roi_localization(self, sample_image):
        pil = Image.open(sample_image).convert("RGB")
        bbox = detect_lesion_bbox(pil)
        assert len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 <= pil.width
        assert 0 <= y1 < y2 <= pil.height

        cropped, crop_bbox = extract_lesion_roi(pil, target_size=(224, 224))
        assert cropped.size == (224, 224)
        assert crop_bbox == bbox

    # ── 19. Conformal Prediction Set & Selective Classifier ──────────────────
    def test_19_conformal_selective_classification(self):
        val_probs = np.array([
            [0.95, 0.05],
            [0.10, 0.90],
            [0.85, 0.15],
            [0.05, 0.95],
            [0.70, 0.30],
        ])
        val_labels = np.array([0, 1, 0, 1, 0])

        cp = ConformalPredictor(alpha=0.10)
        cp.calibrate(val_probs, val_labels)
        assert cp.calibrated
        assert 0.0 <= cp.q_val <= 1.0

        sc = SelectiveClassifier(confidence_threshold=0.70, conformal_predictor=cp)
        res_suspected = sc.classify([0.05, 0.95])
        assert res_suspected["decision_state"] == STATE_MPOX_SUSPECTED

        res_notdetected = sc.classify([0.92, 0.08])
        assert res_notdetected["decision_state"] == STATE_MPOX_NOT_DETECTED

        res_inconclusive = sc.classify([0.52, 0.48])
        assert res_inconclusive["decision_state"] == STATE_INCONCLUSIVE
        assert res_inconclusive["abstain_recommended"] == True
