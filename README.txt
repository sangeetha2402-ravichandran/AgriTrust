AGRI-TRUST Version 2 - Clean Aligned Report

Changes:
- Removed metadata table from the front-end result area
- Shows only Original Image and Grad-CAM++ Overlay
- PDF report is properly aligned
- PDF includes prediction result, confidence, trust level, original image, Grad-CAM++ overlay, and metadata table
- No icons or emojis

Required files:
final_multimodal_fusion_model.pth
mm_scaler.pkl
mm_label_encoder.pkl

Run:
pip install -r requirements.txt
streamlit run app_v2.py
