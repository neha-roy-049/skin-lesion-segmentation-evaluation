# app.py
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modules.model import SkinLesionSegNet
from modules.uncertainty import estimate_uncertainty
from modules.abcd_features import extract_all_features
from modules.risk_eval import compute_risk_score
from modules.explainability import generate_gradcam

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache_resource
def load_model():
    model = SkinLesionSegNet().to(DEVICE)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=DEVICE))
    model.eval()
    return model

def preprocess_image(image_rgb):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transform(image=image_rgb)['image'].unsqueeze(0)

st.set_page_config(page_title="Skin Lesion Analyser", layout="wide")
st.title("Skin Lesion Segmentation with Interpretable Risk Evaluation")


st.caption("ABCDE-based interpretable AI for lesion segmentation")

    
uploaded = st.file_uploader("Upload a dermoscopy image (JPG or PNG)", type=['jpg','jpeg','png'])

if uploaded:
    image_pil = Image.open(uploaded).convert('RGB')
    image_rgb = np.array(image_pil)

    # Images section - original and segmentation mask side-by-side
    img_col1, img_col2 = st.columns(2)
    img_col1.subheader("Original image")
    img_col1.image(image_rgb, use_container_width=True)

    with st.spinner("Analysing lesion..."):
        # 1. Load Model & Preprocess
        model = load_model()
        image_tensor = preprocess_image(image_rgb)

        # 2. Get Segmentation Mask & Uncertainty
        mask, uncertainty_map, confidence = estimate_uncertainty(
            model, image_tensor, n_passes=15, device=DEVICE
        )

        orig_h, orig_w = image_rgb.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h))

        # 3. Extract Clinical Features & Compute Risk
        features = extract_all_features(image_rgb, mask_resized)
        result = compute_risk_score(features, confidence)
        
        # 4. Generate Grad-CAM for Technical Explainability
        cam_overlay, _ = generate_gradcam(model, image_tensor, image_rgb, DEVICE)

    img_col2.subheader("Segmentation mask")
    img_col2.image((mask_resized * 255).astype(np.uint8), use_container_width=True)

    st.divider()

    # AI Explainability Section (Both Technical and Clinical)
    st.subheader("AI Explainability Module")
    e_col1, e_col2 = st.columns(2)
    
    with e_col1:
        st.write("**Technical View: Feature Activation (Grad-CAM)**")
        st.image(cam_overlay, use_container_width=True, caption="Highlights the specific visual features driving the AI's prediction.")
        
    with e_col2:
        st.write(f"**Clinical View: Model Uncertainty (Overall Confidence: {result['confidence']}%)**")
        unc_display = cv2.resize(uncertainty_map, (orig_w, orig_h))
        unc_colored = cv2.applyColorMap((unc_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        unc_colored = cv2.cvtColor(unc_colored, cv2.COLOR_BGR2RGB)
        st.image(unc_colored, use_container_width=True, caption="Brighter areas indicate where the AI is less certain about the edge.")
        if result['confidence'] < 60:
            st.warning("Low confidence prediction — please consult a dermatologist.")

    st.divider()

    # Textual Explanation of ABCD Features
    st.subheader("Detailed Clinical Analysis (ABCD Features)")
    fc1, fc2, fc3, fc4 = st.columns(4)

    descriptions = {
        'asymmetry': {
            'title': "Asymmetry",
            'desc': "Is the lesion shape irregular when divided? (Score 0-1, higher = more asymmetric)"
        },
        'border': {
            'title': "Border irregularity",
            'desc': "Are the edges jagged or blurry? (Score 0-1, higher = more irregular)"
        },
        'color': {
            'title': "Color variation",
            'desc': "Are there multiple distinct colors/shades? (Score 0-1, higher = more variation)"
        },
        'diameter': {
            'title': "Diameter (norm.)",
            'desc': "Is the lesion size large relative to typical? (Score 0-1, normalized by image size)"
        }
    }

    def display_feature(col, feat_key, feat_val):
        with col:
            info = descriptions[feat_key]
            st.metric(info['title'], f"{feat_val:.2f}")
            st.caption(f"_{info['desc']}_")

    display_feature(fc1, 'asymmetry', features['asymmetry'])
    display_feature(fc2, 'border', features['border'])
    display_feature(fc3, 'color', features['color'])
    display_feature(fc4, 'diameter', features['diameter'])

    st.divider()

    # Final Risk Assessment
    st.subheader("Final Risk Assessment")
    icons = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
    
    st.markdown(f"### {icons[result['risk_class']]} {result['risk_class']}")
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.write(f"**Final Adjusted Risk Score:** {result['adjusted_score']} / 100")
        st.info("This score combines the clinical features with the model's certainty. If the model is uncertain, the score shifts toward the middle to prevent false confidence.")
    with r_col2:
        st.write(f"**Raw Feature Score:** {result['raw_score']} (Weighted sum of ABCD)")
        #st.write(f"**Normalized Score:** {result['normalized_score']}% (Scale of 0-100 before confidence adjustment)")