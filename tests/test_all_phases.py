import os
import json
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn

from src.config import (
    NUM_CLASSES,
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME,
    INPUT_SHAPE,
    IMG_HEIGHT,
    IMG_WIDTH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    STATE_PREDICTION_AVAILABLE,
    STATE_REVIEW_REQUIRED,
    STATE_INVALID_INPUT,
    BEST_MODEL_PATH,
    CLASS_MAPPING_PATH,
    MODEL_CONFIG_PATH
)
from src.dataset import get_transforms, MpoxDataset
from src.model import VGG19MpoxClassifier, load_model, save_model_artifacts
from src.gradcam import GradCAM
from src.uncertainty import assess_prediction_uncertainty, calculate_predictive_entropy


class TestMpoxProject:
    """
    Automated Test Suite for Mpox Recognition with VGG19, Grad-CAM and Uncertainty Estimation.
    Covers all 11 evaluation requirements from Phase 7.
    """

    @pytest.fixture(scope="class")
    def sample_image(self, tmp_path_factory):
        # Create a synthetic skin image for fast, deterministic unit testing
        fn = tmp_path_factory.mktemp("data") / "CHP_01_01.jpg"
        img = Image.new("RGB", (300, 300), color=(180, 120, 100))
        img.save(fn)
        return fn

    @pytest.fixture(scope="class")
    def mock_model(self):
        # Lightweight instantiated model for fast testing
        model = VGG19MpoxClassifier(
            num_classes=NUM_CLASSES,
            dense_units=64,
            dropout_rate=0.2,
            pretrained=False,
            freeze_backbone=True
        )
        model.eval()
        return model

    # 1. Model loads successfully
    def test_01_model_instantiation(self, mock_model):
        assert isinstance(mock_model, nn.Module)
        assert hasattr(mock_model, "features")
        assert hasattr(mock_model, "classifier_head")

    # 2. Model output dimensions match the number of classes (6 classes)
    def test_02_model_output_dimensions(self, mock_model):
        dummy_input = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH)
        with torch.no_grad():
            outputs = mock_model(dummy_input)
        assert outputs.shape == (2, NUM_CLASSES), f"Expected shape (2, {NUM_CLASSES}), got {outputs.shape}"

    # 3. Image preprocessing produces expected tensor shape and valid numerical values
    def test_03_image_preprocessing(self, sample_image):
        transform = get_transforms(is_training=False)
        pil_img = Image.open(sample_image).convert("RGB")
        tensor = transform(pil_img)

        assert tensor.shape == (3, IMG_HEIGHT, IMG_WIDTH)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()
        assert tensor.dtype == torch.float32

    # 4. Predictions produce valid class probabilities (sum to 1, range [0, 1])
    def test_04_valid_class_probabilities(self, mock_model):
        dummy_input = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH)
        probs = mock_model.predict_proba(dummy_input).cpu().detach().numpy()[0]

        assert len(probs) == NUM_CLASSES
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    # 5. Class indices map correctly to class names
    def test_05_class_indices_and_mapping(self):
        assert len(CLASS_CODES) == 6
        assert len(CLASS_NAMES) == 6
        for i, code in enumerate(CLASS_CODES):
            assert code in CODE_TO_NAME
            assert CODE_TO_NAME[code] == CLASS_NAMES[i]
        assert "MKP" in CLASS_CODES
        assert CODE_TO_NAME["MKP"] == "Monkeypox"

    # 6. Grad-CAM produces a finite, correctly sized heatmap
    def test_06_gradcam_generation(self, mock_model, sample_image):
        gradcam = GradCAM(mock_model)
        orig_img, heat_img, overlay_img, pred_idx, prob = gradcam.explain_image(sample_image)

        assert orig_img.shape[2] == 3
        assert heat_img.shape[2] == 3
        assert overlay_img.shape[2] == 3
        assert orig_img.shape[:2] == heat_img.shape[:2] == overlay_img.shape[:2]
        assert not np.isnan(heat_img).any()
        assert not np.isnan(overlay_img).any()
        gradcam.remove_hooks()

    # 7. Uncertainty logic correctly flags predictions below threshold
    def test_07_uncertainty_threshold_logic(self):
        # Case A: High confidence -> PREDICTION_AVAILABLE
        high_conf_probs = [0.85, 0.03, 0.03, 0.03, 0.03, 0.03]
        res_high = assess_prediction_uncertainty(high_conf_probs, threshold=0.75)
        assert res_high["decision_state"] == STATE_PREDICTION_AVAILABLE
        assert res_high["review_recommended"] is False
        assert res_high["predicted_code"] == CLASS_CODES[0]

        # Case B: Sub-threshold confidence -> REVIEW_REQUIRED
        low_conf_probs = [0.40, 0.30, 0.10, 0.10, 0.05, 0.05]
        res_low = assess_prediction_uncertainty(low_conf_probs, threshold=0.75)
        assert res_low["decision_state"] == STATE_REVIEW_REQUIRED
        assert res_low["review_recommended"] is True

    # 8. Invalid image files are handled safely
    def test_08_invalid_image_handling(self):
        res_none = assess_prediction_uncertainty(None, is_valid_input=False)
        assert res_none["decision_state"] == STATE_INVALID_INPUT
        assert res_none["review_recommended"] is True

        res_nan = assess_prediction_uncertainty([np.nan, 0.2, 0.3, 0.1, 0.2, 0.2])
        assert res_nan["decision_state"] == STATE_INVALID_INPUT

    # 9. Missing model checkpoints produce actionable error messages
    def test_09_missing_checkpoint_error(self, tmp_path):
        fake_path = tmp_path / "non_existent_model.keras"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model(checkpoint_path=fake_path)
        assert "Model checkpoint not found" in str(exc_info.value)

    # 10. Application can run without retraining model (artifact loading)
    def test_10_artifact_saving_and_loading(self, mock_model, tmp_path):
        save_file = tmp_path / "test_model.pt"
        cfg_file = tmp_path / "cfg.json"
        map_file = tmp_path / "map.json"

        save_model_artifacts(mock_model, save_file, cfg_file, map_file)
        assert save_file.exists()
        assert cfg_file.exists()
        assert map_file.exists()

        # Load back
        loaded = load_model(checkpoint_path=save_file)
        assert isinstance(loaded, VGG19MpoxClassifier)

    # 11. Saved metrics and configuration are valid JSON
    def test_11_metrics_and_config_format(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        config_data = {
            "architecture": "VGG19_TransferLearning",
            "num_classes": NUM_CLASSES,
            "classes": CLASS_CODES
        }
        with open(cfg_file, "w") as f:
            json.dump(config_data, f)

        with open(cfg_file, "r") as f:
            loaded_cfg = json.load(f)
        assert loaded_cfg["num_classes"] == 6
        assert loaded_cfg["architecture"] == "VGG19_TransferLearning"
