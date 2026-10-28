import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import torch

from src.config import (
    BEST_MODEL_PATH,
    RESNET_MODEL_PATH,
    CLASS_MAPPING_PATH,
    MODEL_CONFIG_PATH,
    TEST_METRICS_PATH,
    ENSEMBLE_METRICS_PATH,
    UNCERTAINTY_METRICS_PATH,
    CONFUSION_MATRIX_PLOT,
    ENSEMBLE_CONFUSION_MATRIX_PLOT,
    TRAINING_CURVES_PLOT,
    RELIABILITY_PLOT,
    CLASS_CODES,
    CLASS_NAMES,
    CODE_TO_NAME,
    NUM_CLASSES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    STATE_PREDICTION_AVAILABLE,
    STATE_REVIEW_REQUIRED,
    STATE_INVALID_INPUT,
    TARGET_CODE
)
from src.model import (
    load_model,
    load_resnet_model,
    load_ensemble_model,
    VGG19MpoxClassifier,
    ResNetMpoxClassifier,
    EnsembleMpoxClassifier
)
from src.gradcam import GradCAM
from src.uncertainty import assess_prediction_uncertainty

# Streamlit Page Setup
st.set_page_config(
    page_title="Mpox Recognition & Explainable AI — Ensemble VGG19+ResNet34",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-ok {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-review {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-invalid {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        border-radius: 4px;
        color: #7F1D1D;
        font-size: 0.9rem;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .ensemble-badge {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model(model_choice: str):
    """
    Loads and caches the selected model once per session.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if model_choice == "Ensemble (VGG19 + ResNet34)":
            if not BEST_MODEL_PATH.exists():
                return None, device, f"VGG19 checkpoint not found at {BEST_MODEL_PATH}"
            if not RESNET_MODEL_PATH.exists():
                return None, device, f"ResNet34 checkpoint not found at {RESNET_MODEL_PATH}"
            model = load_ensemble_model(BEST_MODEL_PATH, RESNET_MODEL_PATH, device=device)
        elif model_choice == "ResNet34 (High Sensitivity)":
            if not RESNET_MODEL_PATH.exists():
                return None, device, f"ResNet34 checkpoint not found at {RESNET_MODEL_PATH}"
            model = load_resnet_model(RESNET_MODEL_PATH, device=device)
        else:  # VGG19 baseline
            if not BEST_MODEL_PATH.exists():
                return None, device, f"VGG19 checkpoint not found at {BEST_MODEL_PATH}"
            model = load_model(BEST_MODEL_PATH, device=device)
        return model, device, None
    except Exception as e:
        return None, device, str(e)


@st.cache_resource
def get_gradcam_engine(_model, device):
    """
    Caches Grad-CAM explainer. Ensemble uses VGG19's final conv layer.
    """
    if _model is None:
        return None
    return GradCAM(_model, device=device)


@st.cache_data
def get_saved_metrics(model_choice: str):
    """
    Loads pre-computed test set metrics for the selected model.
    """
    if model_choice == "Ensemble (VGG19 + ResNet34)":
        path = ENSEMBLE_METRICS_PATH
    elif model_choice == "ResNet34 (High Sensitivity)":
        path = Path(str(ENSEMBLE_METRICS_PATH).replace("ensemble", "resnet34"))
    else:
        path = TEST_METRICS_PATH

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    if TEST_METRICS_PATH.exists():
        try:
            with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# Sidebar Configuration
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
st.sidebar.title("System Controls")

# Model Selection
st.sidebar.markdown("### 🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Inference Model",
    options=[
        "Ensemble (VGG19 + ResNet34)",
        "VGG19 (Baseline)",
        "ResNet34 (High Sensitivity)",
    ],
    index=0,
    help="Ensemble: Best overall accuracy (78.38%). ResNet34: Best Mpox sensitivity (85.19%)."
)

# Model info badges
model_info = {
    "Ensemble (VGG19 + ResNet34)": ("78.38%", "77.93%", "77.78%", "85.11%"),
    "VGG19 (Baseline)":            ("77.03%", "76.44%", "70.37%", "93.62%"),
    "ResNet34 (High Sensitivity)": ("74.32%", "69.47%", "85.19%", "76.60%"),
}
acc, f1, sens, spec = model_info[model_choice]
st.sidebar.markdown(f"""
<div style="background:#F0F9FF; border-radius:8px; padding:10px; font-size:0.87rem; margin-bottom:8px;">
<strong>Selected Model Performance</strong><br>
🎯 Test Acc: <strong>{acc}</strong><br>
📊 Macro F1: <strong>{f1}</strong><br>
🔴 MKP Sensitivity: <strong>{sens}</strong><br>
🟢 MKP Specificity: <strong>{spec}</strong>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Uncertainty Policy")
threshold = st.sidebar.slider(
    "Abstention Confidence Threshold",
    min_value=0.50,
    max_value=0.95,
    value=float(DEFAULT_CONFIDENCE_THRESHOLD),
    step=0.01,
    help="Predictions below this confidence threshold will be flagged as REVIEW_REQUIRED."
)

st.sidebar.markdown("### Explainability (Grad-CAM)")
cam_alpha = st.sidebar.slider("Heatmap Overlay Opacity", min_value=0.1, max_value=0.8, value=0.4, step=0.05)
colormap_name = st.sidebar.selectbox("Colormap", options=["jet", "turbo", "viridis", "inferno"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Architecture Info")
st.sidebar.markdown(f"""
- **Active Model:** {model_choice}
- **Backbone(s):** VGG19 + ResNet34 (Ensemble)
- **Conv Grad-CAM Layer:** `block5_conv4`
- **Output Classes:** 6
- **Device:** CPU
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="disclaimer-box">
<strong>Research Prototype Disclaimer:</strong><br>
This application is strictly for scientific research and educational demonstration. It is not an FDA-cleared or CE-marked medical diagnostic system. All outputs require clinical validation by certified dermatologists.
</div>
""", unsafe_allow_html=True)


# Main Content Area
st.markdown('<div class="main-title">Transfer Learning Enabled Mpox Recognition</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Ensemble AI (VGG19 + ResNet34) &nbsp;'
    f'<span class="ensemble-badge">🚀 Upgraded: 78.38% Accuracy</span>&nbsp; '
    f'| Explainable AI (Grad-CAM) &amp; Uncertainty-Aware Decision Support</div>',
    unsafe_allow_html=True
)

# Load Model
model, device, load_err = get_model(model_choice)

if model is None:
    st.warning(f"⚠️ Model is currently initializing or training: {load_err}")
    st.info("You can inspect the benchmark evaluation and project documentation below while the model is training.")

# Navigation Tabs
tab_predict, tab_explain, tab_eval, tab_compare, tab_info = st.tabs([
    "🔬 Diagnostic Inference",
    "🧠 Explainability (Grad-CAM)",
    "📊 Benchmark Evaluation",
    "📈 Model Comparison",
    "ℹ️ Methodology & Disclaimers"
])

uploaded_image_state = None

with tab_predict:
    st.subheader("1. Patient Skin Lesion Upload & Inference")
    col_upload, col_preview = st.columns([1.2, 1.0])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Dermatological Lesion Image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select an image file containing a skin lesion to analyze."
        )
        sample_btn = st.button("Load Representative Sample Test Image")

    pil_img = None
    if sample_btn:
        from src.config import DATA_DIR
        test_samples = list((DATA_DIR / "fold1" / "test").glob("*.jpg"))
        if test_samples:
            pil_img = Image.open(test_samples[0]).convert("RGB")
            st.success(f"Loaded sample test image: `{test_samples[0].name}`")

    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Invalid image file: {e}")
            pil_img = None

    if pil_img is not None:
        uploaded_image_state = pil_img
        with col_preview:
            st.image(pil_img, caption="Patient Lesion Image (Original)", use_container_width=True)

        if model is not None:
            from torchvision import transforms
            from src.dataset import IMAGENET_MEAN, IMAGENET_STD

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            tensor = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = model.predict_proba(tensor).cpu().numpy()[0]

            # Uncertainty Assessment
            assessment = assess_prediction_uncertainty(probs, threshold=threshold)

            st.markdown("---")
            st.subheader("2. Uncertainty-Aware Diagnostic Result")

            col_res1, col_res2 = st.columns([1.1, 1.2])

            with col_res1:
                pred_code = assessment["predicted_code"]
                pred_name = assessment["predicted_name"]
                conf = assessment["confidence"]
                state = assessment["decision_state"]

                is_mpox = (pred_code == TARGET_CODE)
                header_color = "#DC2626" if is_mpox else "#1E3A8A"

                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {header_color};">
                    <h4 style="margin-top:0; color:{header_color};">Predicted Classification</h4>
                    <h2 style="margin:4px 0;">{pred_name} <span style="font-size:1.1rem; color:#6B7280;">({pred_code})</span></h2>
                    <p style="font-size:1.1rem; margin-bottom:8px;">Model Confidence: <strong>{conf*100:.2f}%</strong></p>
                    <p style="font-size:0.82rem; color:#6B7280; margin-bottom:0;">Inference by: <strong>{model_choice}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                if state == STATE_PREDICTION_AVAILABLE:
                    st.markdown("""<div class="status-badge-ok">✔ PREDICTION AVAILABLE (Confidence ≥ Threshold)</div>""", unsafe_allow_html=True)
                    st.success(assessment["recommendation_reason"])
                elif state == STATE_REVIEW_REQUIRED:
                    st.markdown("""<div class="status-badge-review">⚠️ REVIEW REQUIRED (Sub-threshold Confidence)</div>""", unsafe_allow_html=True)
                    st.warning(assessment["recommendation_reason"])
                else:
                    st.markdown("""<div class="status-badge-invalid">✖ INVALID INPUT</div>""", unsafe_allow_html=True)
                    st.error(assessment["recommendation_reason"])

                st.markdown(f"""
                <div style="margin-top:12px; font-size:0.9rem; color:#4B5563;">
                    <strong>Shannon Predictive Entropy:</strong> {assessment['predictive_entropy']:.4f} nats<br>
                    <strong>Normalized Uncertainty:</strong> {assessment['normalized_entropy']*100:.2f}%
                </div>
                """, unsafe_allow_html=True)

            with col_res2:
                st.markdown("#### Differential Diagnosis Probability Breakdown")
                df_probs = pd.DataFrame([
                    {
                        "Class Code": c,
                        "Diagnosis": CODE_TO_NAME[c],
                        "Probability (%)": round(probs[i] * 100, 2)
                    }
                    for i, c in enumerate(CLASS_CODES)
                ]).sort_values(by="Probability (%)", ascending=False)

                st.bar_chart(
                    data=df_probs.set_index("Diagnosis")["Probability (%)"],
                    color="#2563EB",
                    horizontal=True
                )
                st.dataframe(df_probs, hide_index=True, use_container_width=True)
        else:
            st.info("Model checkpoint is loading or being trained.")
    else:
        st.info("Upload a skin lesion image above or click 'Load Representative Sample Test Image' to begin.")


with tab_explain:
    st.subheader("Explainable AI via Grad-CAM (Target Layer: `block5_conv4`)")
    st.write(
        "Grad-CAM computes the gradients of the selected diagnostic score with respect to VGG19's "
        "final convolutional feature maps to identify the visual features that contributed most to the prediction."
    )

    if uploaded_image_state is not None and model is not None:
        gradcam = get_gradcam_engine(model, device)
        if gradcam is not None:
            selected_class_code = st.selectbox(
                "Select Diagnostic Class for Grad-CAM Attribution Map:",
                options=CLASS_CODES,
                index=int(np.argmax(probs)) if 'probs' in locals() else 0,
                format_func=lambda c: f"{c} - {CODE_TO_NAME[c]}"
            )
            target_class_idx = CLASS_CODES.index(selected_class_code)

            orig_img, heat_img, overlay_img, pred_idx, prob = gradcam.explain_image(
                uploaded_image_state,
                target_class_idx=target_class_idx,
                alpha=cam_alpha
            )

            col_cam1, col_cam2, col_cam3 = st.columns(3)
            with col_cam1:
                st.image(orig_img, caption="1. Original Input Lesion", use_container_width=True)
            with col_cam2:
                st.image(heat_img, caption=f"2. Grad-CAM Heatmap ({selected_class_code})", use_container_width=True)
            with col_cam3:
                st.image(overlay_img, caption=f"3. Heatmap Overlay ({selected_class_code})", use_container_width=True)

            st.markdown("""
            > **Clinical Interpretation Guidance:**
            > Warm colors (red, yellow) highlight areas of the lesion that exerted the strongest positive influence 
            > on the model's classification score. Cool colors (blue, cyan) indicate low activation. 
            > This visualization reflects feature attribution and does not establish medical truth or lesion boundaries.
            """)
    else:
        st.info("Please upload an image in the 'Diagnostic Inference' tab to view its Grad-CAM visual explanation.")


with tab_eval:
    st.subheader("Benchmark Test Set Performance (Untouched Evaluation)")
    metrics = get_saved_metrics(model_choice)

    if metrics is not None:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col_m2.metric("Macro F1-Score", f"{metrics['macro_f1']*100:.2f}%")

        mpox_eval = metrics.get("mpox_target_evaluation", {})
        sens = mpox_eval.get("sensitivity_recall", 0.0)
        spec = mpox_eval.get("specificity", 0.0)
        col_m3.metric("Mpox Sensitivity (Recall)", f"{sens*100:.2f}%")
        col_m4.metric("Mpox Specificity", f"{spec*100:.2f}%")

        st.markdown("---")
        col_cm, col_curves = st.columns(2)

        with col_cm:
            st.markdown("#### Test Set Confusion Matrix")
            cm_plot = ENSEMBLE_CONFUSION_MATRIX_PLOT if model_choice == "Ensemble (VGG19 + ResNet34)" else CONFUSION_MATRIX_PLOT
            if cm_plot.exists():
                st.image(str(cm_plot), use_container_width=True)
            elif CONFUSION_MATRIX_PLOT.exists():
                st.image(str(CONFUSION_MATRIX_PLOT), use_container_width=True)
            else:
                st.write(pd.DataFrame(metrics["confusion_matrix"], index=CLASS_CODES, columns=CLASS_CODES))

        with col_curves:
            st.markdown("#### Training and Validation Loss / Accuracy Curves")
            if TRAINING_CURVES_PLOT.exists():
                st.image(str(TRAINING_CURVES_PLOT), use_container_width=True)
            elif RELIABILITY_PLOT.exists():
                st.image(str(RELIABILITY_PLOT), use_container_width=True)
            else:
                st.info("Training curves will appear after model training completes.")

        st.markdown("---")
        st.markdown("#### Per-Class Performance Breakdown")
        per_class = metrics.get("per_class_metrics", {})
        if per_class:
            df_pc = pd.DataFrame([
                {
                    "Class Code": k,
                    "Diagnosis": v["name"],
                    "Precision": f"{v['precision']:.4f}",
                    "Recall (Sensitivity)": f"{v['recall']:.4f}",
                    "F1-Score": f"{v['f1_score']:.4f}",
                    "Test Samples (Support)": v["support"]
                }
                for k, v in per_class.items()
            ])
            st.dataframe(df_pc, hide_index=True, use_container_width=True)

        if "classification_report_text" in metrics:
            with st.expander("Detailed Scikit-Learn Classification Report"):
                st.text(metrics["classification_report_text"])
    else:
        st.info("No saved test evaluation metrics found yet. Run src/evaluate_ensemble.py to generate them.")


with tab_compare:
    st.subheader("📈 Model Comparison: VGG19 vs ResNet34 vs Ensemble")
    st.markdown("All metrics evaluated on the same **74 untouched test images** from `fold1/test`.")

    comparison_data = {
        "Model": ["VGG19 (Baseline)", "ResNet34 (High Sensitivity)", "Ensemble VGG19+ResNet34"],
        "Test Accuracy": ["77.03%", "74.32%", "78.38% ✅"],
        "Macro F1": ["76.44%", "69.47%", "77.93% ✅"],
        "Macro Precision": ["75.03%", "—", "—"],
        "MKP Sensitivity": ["70.37%", "85.19% ✅", "77.78%"],
        "MKP Specificity": ["93.62% ✅", "76.60%", "85.11%"],
        "Algorithm Highlights": [
            "VGG19 + 2-Stage Fine-Tuning + Class Weights",
            "ResNet34 + Layer3/4 Fine-Tuning + BatchNorm Head + Label Smoothing + Cosine LR",
            "Soft-Voting Ensemble (VGG19×0.7 + ResNet34×0.3) — Accuracy Optimal Blend"
        ]
    }

    df_compare = pd.DataFrame(comparison_data)
    st.dataframe(df_compare, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Key Algorithm Upgrades Applied")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
**🏗️ ResNet34 Architecture**
- Residual skip connections prevent vanishing gradients
- BatchNorm in classification head for stable training
- Layer3 + Layer4 progressive fine-tuning
- **Effect:** +14.82% Mpox Sensitivity vs VGG19
""")

    with col2:
        st.markdown("""
**🎛️ Advanced Optimization**
- **Label Smoothing** (ε=0.1): Prevents overconfident predictions
- **AdamW + Weight Decay** (1e-3): Better generalization
- **Cosine Annealing LR**: Smooth convergence, escape local minima
- **Balanced Cross-Entropy**: Handles class imbalance
""")

    with col3:
        st.markdown("""
**🤝 Soft-Voting Ensemble**
- Grid-searched optimal blend: VGG19×0.7 + ResNet34×0.3
- Averages class probability distributions from both models
- Reduces individual model variance and errors
- **Effect:** +1.35% overall accuracy vs VGG19 baseline
""")

    st.info(
        "💡 **Clinical Use Case Guidance:** Use **Ensemble** for highest overall accuracy. "
        "Use **ResNet34** when minimizing false negatives for Mpox is the priority (emergency triage). "
        "Use **VGG19** when maximum Mpox specificity (fewer false positives) is needed."
    )


with tab_info:
    st.subheader("Methodology & Clinical Scope")
    st.markdown("""
    ### Project Overview
    - **Title:** Transfer Learning Enabled Mpox Recognition Using VGG19 + ResNet34 Ensemble for Accurate Medical Image Analysis
    - **Objectives:** 
      1. Differential classification of Mpox against similar dermatological presentations (Chickenpox, Cowpox, HFMD, Measles, and Healthy skin).
      2. Explainable AI through Grad-CAM to visualize model feature attribution on dermatological lesions.
      3. Uncertainty-aware predictions with human-in-the-loop abstention when diagnostic ambiguity is elevated.

    ### Accuracy Enhancement Algorithms Applied
    - **ResNet34 Transfer Learning:** Residual connections + Layer3/Layer4 fine-tuning + BatchNorm head
    - **Label Smoothing Loss:** Prevents overconfidence, improves calibration (ε=0.1)
    - **AdamW Optimizer + Weight Decay:** Stronger generalization vs plain Adam
    - **Cosine Annealing LR Schedule:** Better convergence, fewer local minima
    - **Weighted Cross-Entropy:** Handles class imbalance (Monkeypox over-represented)
    - **Soft-Voting Ensemble:** Grid-searched optimal blend of VGG19 and ResNet34 probability outputs

    ### Transfer Learning Architecture
    - **VGG19:** ImageNet pretrained → 2-stage fine-tuning (head + Block5) → 77.03% accuracy
    - **ResNet34:** ImageNet pretrained → Layer3+4 fine-tuning + BatchNorm head → 85.19% MKP sensitivity
    - **Ensemble:** Soft-vote blend (VGG19×0.7 + ResNet34×0.3) → **78.38% accuracy**

    ### Explainable AI (Grad-CAM)
    - Computes gradients of the target class score with respect to feature maps of VGG19's final conv layer (`block5_conv4`).
    - Produces a normalized coarse 2D localization map highlighting discriminating lesion patterns.

    ### Uncertainty & Abstention Policy
    - Based on maximum predicted probability and predictive entropy.
    - An abstention threshold is selected using validation data only (no test set leakage).
    - If confidence falls below the threshold, the system flags `REVIEW_REQUIRED` to recommend clinical assessment.

    ### Ethical & Medical Research Disclaimer
    This software is an experimental prototype intended for academic and research investigation only. It does NOT constitute medical advice, diagnosis, or treatment. It is not intended to replace clinical judgment, physical examination, or PCR laboratory testing.
    """)
