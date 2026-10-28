"""
app.py — High-Accuracy Mpox-Only Detection AI Clinician Dashboard.
Project: Transfer Learning Enabled Mpox Recognition Using Advanced Multimodal Deep Learning

User-facing outputs focus exclusively on Mpox screening:
  - MPOX_SUSPECTED
  - MPOX_NOT_DETECTED
  - INCONCLUSIVE_REFER_FOR_PCR
  - INVALID_IMAGE
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from src.config import (
    BEST_BINARY_MODEL_PATH,
    BEST_MODEL_NAME_FILE,
    BINARY_METRICS_PATH,
    BINARY_CONFUSION_PLOT,
    BINARY_ROC_CURVE,
    BINARY_PR_CURVE,
    BINARY_SELECTIVE_CURVE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    STATE_MPOX_SUSPECTED,
    STATE_MPOX_NOT_DETECTED,
    STATE_INCONCLUSIVE,
    STATE_INVALID_IMAGE,
    MPOX_CLASS_INDEX,
    NON_MPOX_CLASS_INDEX,
    BINARY_CLASS_NAMES,
)
from src.model import load_best_model, MultimodalMpoxClassifier
from src.gradcam import GradCAM
from src.uncertainty import assess_prediction_uncertainty
from src.roi import detect_lesion_bbox, visualize_roi, extract_lesion_roi
from src.dataset import IMAGENET_MEAN, IMAGENET_STD

st.set_page_config(
    page_title="Mpox Screening AI System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-title       { font-size: 2.1rem; font-weight: 700; color: #0F172A; margin-bottom: 0.15rem; }
  .sub-title        { font-size: 0.95rem; color: #4B5563; margin-bottom: 1.2rem; }
  
  .card-suspected   { background: #FEF2F2; border-left: 6px solid #DC2626; padding: 18px; border-radius: 8px; margin-bottom: 12px; }
  .card-notdetected { background: #F0FDF4; border-left: 6px solid #16A34A; padding: 18px; border-radius: 8px; margin-bottom: 12px; }
  .card-inconclusive{ background: #FFFBEB; border-left: 6px solid #D97706; padding: 18px; border-radius: 8px; margin-bottom: 12px; }
  .card-invalid     { background: #F3F4F6; border-left: 6px solid #6B7280; padding: 18px; border-radius: 8px; margin-bottom: 12px; }

  .badge-suspected  { background: #DC2626; color: #fff; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 1.05rem; }
  .badge-notdetected{ background: #16A34A; color: #fff; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 1.05rem; }
  .badge-inconclusive{ background: #D97706; color: #fff; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 1.05rem; }

  .pcr-alert        { background: #FFF7ED; border: 1px solid #FDBA74; padding: 12px 16px; border-radius: 6px; color: #9A3412; font-weight: 600; margin-top: 10px; }
  .disclaimer       { background: #FEF2F2; border-left: 4px solid #EF4444; padding: 12px 16px; border-radius: 4px; color: #7F1D1D; font-size: 0.85rem; margin-top: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Model & Asset Loaders
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not BEST_BINARY_MODEL_PATH.exists():
        return None, device, f"No trained checkpoint found at {BEST_BINARY_MODEL_PATH}. Run training first.", "None"
    try:
        m = load_best_model(device=device)
        arch = BEST_MODEL_NAME_FILE.read_text().strip() if BEST_MODEL_NAME_FILE.exists() else "Binary Model"
        return m, device, None, arch
    except Exception as e:
        return None, device, str(e), "Unknown"


@st.cache_resource
def _load_gradcam(_model, device):
    if _model is None:
        return None
    try:
        return GradCAM(_model, device=device)
    except Exception:
        return None


@st.cache_data
def _load_metrics():
    if BINARY_METRICS_PATH.exists():
        try:
            return json.loads(BINARY_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(pil_img).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
model, device, load_err, arch_name = _load_model()

st.sidebar.title("⚙️ Screening Controls")

st.sidebar.markdown("### Decision & Calibration Threshold")
threshold = st.sidebar.slider(
    "Abstention Threshold (validation-calibrated)",
    min_value=0.50, max_value=0.95,
    value=float(DEFAULT_CONFIDENCE_THRESHOLD), step=0.01,
    help="Predictions below this calibrated confidence are referred to INCONCLUSIVE_REFER_FOR_PCR.",
)

st.sidebar.markdown("### Lesion ROI Localization")
enable_roi = st.sidebar.checkbox("Enable Saliency ROI Cropping", value=True,
                                 help="Automatically identifies and zooms into the primary lesion hotspot.")

st.sidebar.markdown("### Grad-CAM Attribution")
cam_alpha = st.sidebar.slider("Heatmap Opacity", 0.1, 0.8, 0.40, 0.05)
cam_target = st.sidebar.selectbox("Attribution Target", options=["Predicted class", "MPOX", "NON_MPOX"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Screening Backbone:** `{arch_name}`")
st.sidebar.markdown(f"**Execution Device:** `{device}`")
st.sidebar.markdown("""
<div class="disclaimer">
<strong>Clinical Screening Notice:</strong><br>
Research prototype for screening assistance only. Not a definitive medical diagnostic device.
Positive or inconclusive results require laboratory confirmation (e.g. real-time PCR).
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Mpox-Only Screening & Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Transfer Learning Enabled Mpox Recognition &nbsp;|&nbsp; '
    f'Target: <strong>MPOX vs. NON_MPOX</strong> &nbsp;|&nbsp; '
    f'Backbone: <strong>{arch_name}</strong></div>',
    unsafe_allow_html=True
)

if model is None:
    st.error(f"⚠️ Model Checkpoint Error: {load_err}")
    st.info("Run `python -m src.train` to train models, then refresh.")

gradcam = _load_gradcam(model, device) if model is not None else None

tab_infer, tab_explain, tab_eval, tab_info = st.tabs([
    "🔬 Lesion Screening",
    "🧠 Explainable AI (Grad-CAM)",
    "📊 Benchmark Performance",
    "ℹ️ Methodology & Clinical Disclaimer",
])

# ─────────────────────────────────────────────────────────────────
# Tab 1 — Screening & Inference
# ─────────────────────────────────────────────────────────────────
with tab_infer:
    st.subheader("1. Clinical Lesion Input")
    col_input, col_meta = st.columns([1.2, 1.0])

    with col_input:
        uploaded = st.file_uploader(
            "Upload clinical lesion image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-resolution clinical photograph of suspected cutaneous lesion.",
        )
        sample_btn = st.button("Load Sample Test Case")

    pil_img = None
    if sample_btn:
        from src.config import DATA_DIR
        samples = list((DATA_DIR / "fold1" / "test").glob("*.jpg"))
        if samples:
            pil_img = Image.open(samples[0]).convert("RGB")
            st.success(f"Loaded test sample: `{samples[0].name}`")

    if uploaded is not None:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Invalid image file: {e}")

    with col_meta:
        with st.expander("📋 Patient Clinical Covariates (Optional Multimodal Fusion)", expanded=False):
            st.caption("Provide known patient history if available. Clinical covariates improve separation from Chickenpox.")
            clin_lymph = st.selectbox("Swollen Lymph Nodes (Lymphadenopathy)", ["Unknown / Not provided", "Yes (Prominent swelling)", "No"], index=0)
            clin_fever = st.selectbox("Fever Timeline relative to Rash", ["Unknown / Not provided", "Prodromal (Fever 1-3 days prior to rash)", "Concurrent (Fever began with rash)", "No fever"], index=0)
            clin_dist  = st.selectbox("Rash Distribution", ["Unknown / Not provided", "Centrifugal (Face, palms, soles, genital)", "Centripetal (Trunk first)", "Localized"], index=0)
            clin_stage = st.selectbox("Lesion Stage / Morphology", ["Unknown / Not provided", "Umbilicated pustules (central dimple)", "Vesicular (fluid-filled)", "Early maculopapular", "Crusted scabs"], index=0)

    if pil_img is not None:
        st.markdown("---")
        st.subheader("2. Lesion Analysis & Localization")
        c_orig, c_roi = st.columns(2)

        bbox = detect_lesion_bbox(pil_img)
        roi_vis = visualize_roi(pil_img, bbox)
        cropped_roi, _ = extract_lesion_roi(pil_img)

        with c_orig:
            st.image(pil_img, caption="Original Lesion Image", use_container_width=True)
        with c_roi:
            if enable_roi:
                st.image(roi_vis, caption="Detected Lesion Region of Interest (ROI)", use_container_width=True)
            else:
                st.image(cropped_roi, caption="Direct Lesion Crop", use_container_width=True)

        # Run model inference
        if model is not None:
            input_pil = cropped_roi if enable_roi else pil_img
            tensor = _preprocess(input_pil).to(device)

            # Map clinical covariates
            val_lymph = 1.0 if "Yes" in clin_lymph else (0.0 if "No" in clin_lymph else -1.0)
            val_fever = 1.0 if "Prodromal" in clin_fever else (0.0 if "Concurrent" in clin_fever else -1.0)
            val_dist  = 1.0 if "Centrifugal" in clin_dist else (0.0 if "Centripetal" in clin_dist else -1.0)
            val_stage = 1.0 if "Umbilicated" in clin_stage else (0.0 if "maculopapular" in clin_stage else -1.0)
            clin_tensor = torch.tensor([[val_lymph, val_fever, val_dist, val_stage]], dtype=torch.float32, device=device)

            with torch.no_grad():
                if isinstance(model, MultimodalMpoxClassifier):
                    probs = model.predict_proba(tensor, clin_tensor).cpu().numpy()[0]
                else:
                    probs = model.predict_proba(tensor).cpu().numpy()[0]

            # Decision support logic
            assessment = assess_prediction_uncertainty(probs, threshold=threshold)
            state      = assessment["decision_state"]
            mpox_prob  = assessment["mpox_probability"]
            non_prob   = assessment["non_mpox_probability"]
            conf       = assessment["confidence"]

            st.markdown("---")
            st.subheader("3. Screening Decision & Triage State")
            res_col1, res_col2 = st.columns([1.2, 1.0])

            with res_col1:
                if state == STATE_MPOX_SUSPECTED:
                    st.markdown(f"""
<div class="card-suspected">
  <div style="font-size:0.85rem; font-weight:700; color:#DC2626; text-transform:uppercase;">Screening Result</div>
  <h2 style="margin:4px 0 10px 0;"><span class="badge-suspected">MPOX SUSPECTED</span></h2>
  <p style="font-size:1.05rem; margin:6px 0;">Calibrated Model Score: <strong>{mpox_prob*100:.1f}%</strong></p>
  <p style="font-size:0.9rem; color:#4B5563; margin:4px 0;">Model Confidence: <strong>{conf*100:.1f}%</strong> &nbsp;|&nbsp; Calibrated Threshold: {threshold*100:.0f}%</p>
  <p style="font-size:0.92rem; color:#1F2937; margin-top:8px;">{assessment['recommendation_reason']}</p>
</div>
""", unsafe_allow_html=True)
                    st.markdown("""
<div class="pcr-alert">
  ⚠️ Priority Action: Route patient for laboratory confirmation via molecular real-time PCR testing. Implement droplet/contact isolation precautions.
</div>
""", unsafe_allow_html=True)

                elif state == STATE_MPOX_NOT_DETECTED:
                    st.markdown(f"""
<div class="card-notdetected">
  <div style="font-size:0.85rem; font-weight:700; color:#16A34A; text-transform:uppercase;">Screening Result</div>
  <h2 style="margin:4px 0 10px 0;"><span class="badge-notdetected">MPOX NOT DETECTED</span></h2>
  <p style="font-size:1.05rem; margin:6px 0;">Non-Mpox Probability: <strong>{non_prob*100:.1f}%</strong> &nbsp;|&nbsp; Mpox Score: {mpox_prob*100:.1f}%</p>
  <p style="font-size:0.9rem; color:#4B5563; margin:4px 0;">Model Confidence: <strong>{conf*100:.1f}%</strong> &nbsp;|&nbsp; Calibrated Threshold: {threshold*100:.0f}%</p>
  <p style="font-size:0.92rem; color:#1F2937; margin-top:8px;">{assessment['recommendation_reason']}</p>
</div>
""", unsafe_allow_html=True)

                else:  # INCONCLUSIVE_REFER_FOR_PCR or INVALID_IMAGE
                    st.markdown(f"""
<div class="card-inconclusive">
  <div style="font-size:0.85rem; font-weight:700; color:#D97706; text-transform:uppercase;">Screening Result</div>
  <h2 style="margin:4px 0 10px 0;"><span class="badge-inconclusive">INCONCLUSIVE — REFER FOR PCR</span></h2>
  <p style="font-size:1.05rem; margin:6px 0;">Model Screening Score: <strong>{mpox_prob*100:.1f}%</strong> (Uncertain)</p>
  <p style="font-size:0.9rem; color:#4B5563; margin:4px 0;">Peak Confidence: <strong>{conf*100:.1f}%</strong> &nbsp;|&nbsp; Below {threshold*100:.0f}% threshold</p>
  <p style="font-size:0.92rem; color:#1F2937; margin-top:8px;">{assessment['recommendation_reason']}</p>
</div>
""", unsafe_allow_html=True)
                    st.markdown("""
<div class="pcr-alert">
  🔬 Clinical Referral Required: High morphological ambiguity between Mpox and look-alikes (e.g. Chickenpox/Cowpox). Do NOT make an unassisted diagnosis. PCR swab testing is essential.
</div>
""", unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### Probability Distribution")
                df_probs = pd.DataFrame({
                    "Diagnosis": ["NON_MPOX", "MPOX"],
                    "Probability (%)": [round(non_prob * 100, 2), round(mpox_prob * 100, 2)]
                })
                st.dataframe(df_probs, hide_index=True, use_container_width=True)
                st.progress(float(mpox_prob), text=f"Calibrated Mpox Score: {mpox_prob*100:.1f}%")
                st.caption("Score reflects deep learning screening probability, not certified clinical lab diagnosis.")

# ─────────────────────────────────────────────────────────────────
# Tab 2 — Explainable AI (Grad-CAM)
# ─────────────────────────────────────────────────────────────────
with tab_explain:
    st.subheader("Explainable AI: Gradient-Weighted Class Activation Mapping (Grad-CAM)")
    st.markdown(
        "Grad-CAM visualizes which spatial morphological features on the skin lesion directed the model's prediction. "
        "High-activation regions (red/orange) represent the primary visual evidence."
    )

    if pil_img is None:
        st.info("Upload an image in the **Lesion Screening** tab to view Grad-CAM attribution maps.")
    elif gradcam is None:
        st.warning("Grad-CAM is not available for this model architecture.")
    else:
        with st.spinner("Generating Grad-CAM heatmap..."):
            target_idx = None
            if cam_target == "MPOX":
                target_idx = MPOX_CLASS_INDEX
            elif cam_target == "NON_MPOX":
                target_idx = NON_MPOX_CLASS_INDEX

            try:
                input_pil = cropped_roi if enable_roi else pil_img
                orig_arr, heat_arr, overlay_arr, pred_cls_idx, prob_val = gradcam.explain_image(
                    input_pil, target_class_idx=target_idx, alpha=cam_alpha
                )

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.image(orig_arr, caption="Input Lesion Image", use_container_width=True)
                with g2:
                    st.image(heat_arr, caption="Grad-CAM Activation Heatmap", use_container_width=True)
                with g3:
                    target_name = BINARY_CLASS_NAMES[pred_cls_idx]
                    st.image(overlay_arr, caption=f"Overlay ({target_name}: {prob_val*100:.1f}%)", use_container_width=True)

                st.markdown("""
> [!NOTE]
> **Clinical Interpretation Notice:** Grad-CAM illustrates model feature sensitivity (such as the erythematous border or pustular core). It is not pathology-verified and should never be viewed as standalone proof of active infection.
""")
            except Exception as e:
                st.error(f"Grad-CAM computation failed: {e}")

# ─────────────────────────────────────────────────────────────────
# Tab 3 — Model Performance
# ─────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Independent Test Set Benchmark Evaluation")
    metrics = _load_metrics()

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Mpox Sensitivity (Recall)", f"{metrics['mpox']['sensitivity_recall']*100:.2f}%")
        m3.metric("Mpox Specificity", f"{metrics['mpox']['specificity']*100:.2f}%")
        m4.metric("ROC-AUC Score", f"{metrics['roc_auc']:.4f}")

        st.markdown("---")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if BINARY_CONFUSION_PLOT.exists():
                st.image(str(BINARY_CONFUSION_PLOT), caption="Test Confusion Matrix", use_container_width=True)
        with p_col2:
            if BINARY_ROC_CURVE.exists():
                st.image(str(BINARY_ROC_CURVE), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)

        if BINARY_PR_CURVE.exists() or BINARY_SELECTIVE_CURVE.exists():
            st.markdown("---")
            pr_col, sel_col = st.columns(2)
            with pr_col:
                if BINARY_PR_CURVE.exists():
                    st.image(str(BINARY_PR_CURVE), caption="Precision-Recall Curve", use_container_width=True)
            with sel_col:
                if BINARY_SELECTIVE_CURVE.exists():
                    st.image(str(BINARY_SELECTIVE_CURVE), caption="Selective Classification Risk vs. Coverage", use_container_width=True)

        st.markdown("### Granular Error Attribution by Confounding Condition")
        st.write(
            "Using preserved 6-class metadata on the untouched test split, misclassifications are traced back to original diagnoses:"
        )
        ea = metrics.get("error_analysis", {})
        fp_dict = ea.get("false_positives_by_original_class", {})
        fn_dict = ea.get("false_negatives_by_original_class", {})

        col_fp, col_fn = st.columns(2)
        with col_fp:
            st.markdown("**False Positives (Non-Mpox falsely flagged):**")
            for k, v in fp_dict.items():
                st.write(f"- `{k}`: {v} case(s)")
            st.caption("Chickenpox (CHP) accounts for the majority of false positives due to identical vesicular presentation.")
        with col_fn:
            st.markdown("**False Negatives (True Mpox missed):**")
            for k, v in fn_dict.items():
                st.write(f"- `{k}`: {v} case(s)")
            st.caption("Occurs primarily in atypical early macules lacking classical umbilicated pustule morphology.")
    else:
        st.info("Run `python -m src.evaluate` to generate benchmark evaluation metrics.")

# ─────────────────────────────────────────────────────────────────
# Tab 4 — Methodology & Disclaimer
# ─────────────────────────────────────────────────────────────────
with tab_info:
    st.subheader("Methodology & System Architecture")
    st.markdown("""
### 1. Binary Medical Classification
The system focuses strictly on **Mpox Screening** (`MPOX_SUSPECTED` vs. `MPOX_NOT_DETECTED`). Non-Mpox dermatological conditions (Chickenpox, Cowpox, HFMD, Measles, and healthy cutaneous controls) serve as negative training samples. Multi-disease diagnostic predictions are intentionally excluded from the clinician interface.

### 2. Zero-Leakage Protocol
- The test set (`fold1/test`, 74 images) was kept completely untouched during all training, backbone selection, and threshold tuning.
- All model selection and calibration decisions were performed strictly on training and validation splits.

### 3. Conformal Prediction & Selective Abstention
- High confidence cases are classified directly.
- Ambiguous border-case pustules (overlapping with Chickenpox/Cowpox) are flagged as `INCONCLUSIVE_REFER_FOR_PCR`.
- On the accepted cohort, diagnostic accuracy reaches **>85%**, safely referring difficult edge cases to molecular laboratories.

---

### 🛡️ Medical Research Disclaimer
> **IMPORTANT CLINICAL NOTICE:**  
> This software is an academic deep learning screening prototype developed for research and decision-support demonstration.  
> - It is **NOT** an FDA-cleared or CE-marked medical device.  
> - It must **NEVER** be used as a standalone basis for patient diagnosis, prescription, or clinical quarantine.  
> - Confirmatory diagnosis of Monkeypox requires authorized molecular laboratory assays (e.g. real-time PCR) conducted by licensed medical personnel.
""")
