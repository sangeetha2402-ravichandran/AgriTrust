import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "final_multimodal_fusion_model.pth"
SCALER_PATH = "mm_scaler.pkl"
LABEL_ENCODER_PATH = "mm_label_encoder.pkl"

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class MultimodalFusionModel(nn.Module):
    def __init__(self, tabular_dim, num_classes):
        super(MultimodalFusionModel, self).__init__()

        self.image_branch = timm.create_model(
            "tf_efficientnetv2_b0",
            pretrained=False,
            num_classes=0
        )

        image_dim = self.image_branch.num_features

        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):
        image_features = self.image_branch(image)
        tabular_features = self.tabular_branch(tabular)
        combined = torch.cat([image_features, tabular_features], dim=1)
        output = self.classifier(combined)
        return output


@st.cache_resource
def load_assets():
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    model = MultimodalFusionModel(
        tabular_dim=len(FEATURE_COLS),
        num_classes=len(label_encoder.classes_)
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, label_encoder


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def prepare_image(image):
    return eval_transform(image).unsqueeze(0).to(DEVICE)


def prepare_metadata(N, P, K, temperature, humidity, ph, rainfall, scaler):
    metadata_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=FEATURE_COLS
    )
    metadata_scaled = scaler.transform(metadata_df)
    return torch.tensor(metadata_scaled, dtype=torch.float32).to(DEVICE), metadata_df


def get_trust_level(prediction, confidence_value):
    prediction_lower = str(prediction).lower()

    if "healthy" in prediction_lower and confidence_value >= 50:
        return "Healthy Leaf Detected", "trust-high"

    if confidence_value >= 90:
        return "High Confidence", "trust-high"
    elif confidence_value >= 70:
        return "Medium Confidence", "trust-medium"
    else:
        return "Needs Expert Review", "trust-low"


def confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={"suffix": "%", "font": {"size": 34}},
        title={"text": "Confidence Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#126b3a"},
            "steps": [
                {"range": [0, 70], "color": "#f7d6d6"},
                {"range": [70, 90], "color": "#fff0c2"},
                {"range": [90, 100], "color": "#cdeed8"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=45, b=10))
    return fig


def find_last_conv_layer(model):
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise ValueError("No convolution layer found for Grad-CAM++.")
    return target_layer


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, tabular_tensor, target_class):
        self.model.zero_grad()

        output = self.model(image_tensor, tabular_tensor)
        score = output[:, target_class].sum()
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        eps = 1e-8
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + torch.sum(
            activations * grad_3,
            dim=(2, 3),
            keepdim=True
        )

        alpha = alpha_num / (alpha_denom + eps)
        positive_gradients = F.relu(gradients)

        weights = torch.sum(
            alpha * positive_gradients,
            dim=(2, 3),
            keepdim=True
        )

        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def create_gradcam_overlay(model, original_image, image_tensor, tabular_tensor, predicted_class):
    target_layer = find_last_conv_layer(model)
    gradcam = GradCAMPlusPlus(model, target_layer)

    cam = gradcam.generate(
        image_tensor=image_tensor,
        tabular_tensor=tabular_tensor,
        target_class=predicted_class
    )

    gradcam.remove_hooks()

    image_resized = original_image.resize((224, 224)).convert("RGB")
    original_np = np.array(image_resized).astype(np.float32) / 255.0

    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_np = heatmap.astype(np.float32) / 255.0

    overlay = 0.55 * original_np + 0.45 * heatmap_np
    overlay = np.clip(overlay, 0, 1)

    return original_np, heatmap_np, overlay


def np_to_image_reader(np_img):
    img = Image.fromarray((np_img * 255).astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return ImageReader(buffer)


def make_pdf_report(prediction, confidence, trust, metadata_df, original_np=None, overlay_np=None):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left = 50
    right = width - 50

    pdf.setFillColor(colors.HexColor("#0f5132"))
    pdf.rect(0, height - 95, width, 95, fill=1, stroke=0)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawCentredString(width / 2, height - 38, "AGRI-TRUST")

    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width / 2, height - 62, "Plant Disease Prediction Using Multimodal Deep Learning")

    pdf.setFillColor(colors.black)

    y = height - 125
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(left, y, "Prediction Report")

    y -= 25
    pdf.setStrokeColor(colors.HexColor("#0f5132"))
    pdf.line(left, y, right, y)

    y -= 35
    box_height = 92
    pdf.setFillColor(colors.HexColor("#f4faf6"))
    pdf.roundRect(left, y - box_height + 18, right - left, box_height, 12, fill=1, stroke=0)
    pdf.setFillColor(colors.black)

    label_x = left + 22
    value_x = left + 170
    row_y = y

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(label_x, row_y, "Predicted Disease")
    pdf.drawString(label_x, row_y - 26, "Confidence")
    pdf.drawString(label_x, row_y - 52, "Trust Level")

    pdf.setFont("Helvetica", 11)
    clean_prediction = str(prediction).replace("___", " - ").replace("_", " ")
    pdf.drawString(value_x, row_y, clean_prediction[:55])
    pdf.drawString(value_x, row_y - 26, f"{confidence:.2f}%")
    pdf.drawString(value_x, row_y - 52, trust)

    y -= 125

    if original_np is not None and overlay_np is not None:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(left, y, "Visual Explanation")
        y -= 18

        img_w = 190
        img_h = 190
        gap = 55

        pdf.drawImage(np_to_image_reader(original_np), left + 35, y - img_h, width=img_w, height=img_h)
        pdf.drawImage(np_to_image_reader(overlay_np), left + 35 + img_w + gap, y - img_h, width=img_w, height=img_h)

        pdf.setFont("Helvetica", 10)
        pdf.drawCentredString(left + 35 + img_w / 2, y - img_h - 15, "Original Image")
        pdf.drawCentredString(left + 35 + img_w + gap + img_w / 2, y - img_h - 15, "Grad-CAM++ Overlay")

        y -= 235

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left, y, "Soil and Weather Metadata")
    y -= 22

    table_left = left
    table_top = y
    col1_w = 180
    col2_w = 160
    row_h = 24

    metadata_names = {
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH",
        "rainfall": "Rainfall"
    }

    pdf.setFont("Helvetica", 10)

    for i, col in enumerate(metadata_df.columns):
        row_y = table_top - (i * row_h)

        if i % 2 == 0:
            pdf.setFillColor(colors.HexColor("#f7f7f7"))
        else:
            pdf.setFillColor(colors.white)

        pdf.rect(table_left, row_y - row_h + 6, col1_w + col2_w, row_h, fill=1, stroke=0)

        pdf.setFillColor(colors.black)
        pdf.drawString(table_left + 10, row_y - 10, metadata_names.get(col, col))
        pdf.drawString(table_left + col1_w + 10, row_y - 10, str(metadata_df.iloc[0][col]))

    pdf.setStrokeColor(colors.HexColor("#cccccc"))
    pdf.rect(table_left, table_top - (len(metadata_df.columns) * row_h) + 6, col1_w + col2_w, len(metadata_df.columns) * row_h, fill=0, stroke=1)
    pdf.line(table_left + col1_w, table_top + 6, table_left + col1_w, table_top - (len(metadata_df.columns) * row_h) + 6)

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColor(colors.HexColor("#666666"))
   

    pdf.save()
    buffer.seek(0)
    return buffer


st.set_page_config(
    page_title="AGRI-TRUST",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f6faf7;
}

.main-title {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f5132, #1d7a46);
    color: white;
    margin-bottom: 24px;
    text-align: center;
}

.main-title h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.main-title h3 {
    font-size: 22px;
    font-weight: 400;
}

.section-card {
    background-color: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-card {
    background-color: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    min-height: 250px;
}

.trust-high {
    padding: 16px;
    border-radius: 12px;
    background-color: #d1e7dd;
    color: #0f5132;
    font-size: 22px;
    font-weight: 700;
    text-align: center;
}

.trust-medium {
    padding: 16px;
    border-radius: 12px;
    background-color: #fff3cd;
    color: #664d03;
    font-size: 22px;
    font-weight: 700;
    text-align: center;
}

.trust-low {
    padding: 16px;
    border-radius: 12px;
    background-color: #f8d7da;
    color: #842029;
    font-size: 22px;
    font-weight: 700;
    text-align: center;
}

.stButton > button {
    background: linear-gradient(135deg, #198754, #0f5132);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    font-size: 18px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #157347, #0b3d25);
    color: white;
    border: none;
}

div[data-testid="stDownloadButton"] button {
    width: 230px !important;
    font-size: 14px !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>AGRI-TRUST</h1>
    <h3>Plant Disease Prediction Using Multimodal Deep Learning</h3>
</div>
""", unsafe_allow_html=True)

tab_prediction, tab_about = st.tabs(["Prediction", "About Model"])

with tab_prediction:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Upload Leaf Image")
        uploaded_image = st.file_uploader(
            "Select a leaf image",
            type=["jpg", "jpeg", "png"]
        )

        image = None
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Soil and Weather Metadata")

        c1, c2 = st.columns(2)

        with c1:
            N = st.number_input("Nitrogen", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
            P = st.number_input("Phosphorus", min_value=0.0, max_value=200.0, value=42.0, step=1.0)
            K = st.number_input("Potassium", min_value=0.0, max_value=250.0, value=43.0, step=1.0)
            temperature = st.number_input("Temperature", min_value=-10.0, max_value=60.0, value=20.87, step=0.1)

        with c2:
            humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            rainfall = st.number_input("Rainfall", min_value=0.0, max_value=500.0, value=202.9, step=0.1)

        predict_button = st.button("Predict Disease", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_button:
        if image is None:
            st.error("Please upload a leaf image before prediction.")
        else:
            try:
                with st.spinner("Running prediction..."):
                    model, scaler, label_encoder = load_assets()

                    image_tensor = prepare_image(image)
                    tabular_tensor, metadata_df = prepare_metadata(
                        N, P, K, temperature, humidity, ph, rainfall, scaler
                    )

                    with torch.no_grad():
                        logits = model(image_tensor, tabular_tensor)
                        probabilities = F.softmax(logits, dim=1)
                        confidence, predicted_class = torch.max(probabilities, dim=1)

                    predicted_index = int(predicted_class.item())
                    prediction = label_encoder.inverse_transform([predicted_index])[0]
                    confidence_value = float(confidence.item() * 100)

                    trust_level, trust_class = get_trust_level(prediction, confidence_value)

                    original_np, heatmap_np, overlay_np = create_gradcam_overlay(
                        model=model,
                        original_image=image,
                        image_tensor=image_tensor,
                        tabular_tensor=tabular_tensor,
                        predicted_class=predicted_index
                    )

                st.markdown("## Prediction Result")

                r1, r2 = st.columns([1, 1])

                with r1:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.subheader("Disease")
                    st.markdown(f"### {prediction.replace('___', ' - ').replace('_', ' ')}")
                    st.subheader("Trust Level")
                    st.markdown(f'<div class="{trust_class}">{trust_level}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with r2:
                    st.plotly_chart(
                        confidence_gauge(confidence_value),
                        use_container_width=True
                    )

                st.markdown("## Disease Region Heatmap")

                h1, h2 = st.columns(2)

                with h1:
                    st.image(original_np, caption="Original Image", use_container_width=True)

                with h2:
                    st.image(overlay_np, caption="Grad-CAM++ Overlay", use_container_width=True)

                pdf_file = make_pdf_report(
                    prediction=prediction,
                    confidence=confidence_value,
                    trust=trust_level,
                    metadata_df=metadata_df,
                    original_np=original_np,
                    overlay_np=overlay_np
                )

                st.download_button(
                    label="Download Report",
                    data=pdf_file,
                    file_name="agritrust_prediction_report.pdf",
                    mime="application/pdf"
                )

            except FileNotFoundError:
                st.error("Model files are missing. Keep final_multimodal_fusion_model.pth, mm_scaler.pkl, and mm_label_encoder.pkl in the same folder as this app.")
            except Exception as e:
                st.error("Prediction failed.")
                st.write(str(e))

with tab_about:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("About AGRI-TRUST")
    st.write("AGRI-TRUST is a multimodal plant disease prediction system that combines plant leaf image analysis with soil and weather metadata.")
    st.markdown("""
    **Model:** Multimodal Fusion Model  
    **Image Backbone:** EfficientNetV2-B0  
    **Metadata Features:** Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall  
    **Dataset Size:** 2,200 matched records  
    **Number of Classes:** 22  
    **Reported Accuracy:** 98.18%  
    **Reported Macro F1:** 98.07%  
    **Trust Components:** Confidence, rejection logic, calibration, and Grad-CAM++ explainability
    """)
    st.markdown('</div>', unsafe_allow_html=True)
