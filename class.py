import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import altair as alt
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# 1) PAGE CONFIG & CUSTOM STYLING
# -------------------------------
st.set_page_config(
    page_title="Part Classifier",
    layout="wide",
    page_icon="🔧"
)

st.markdown("""
<style>
body {
    background-color: #f9f9f9;
    color: #333;
    font-family: "Helvetica Neue", sans-serif;
}
h1, h2, h3 {
    color: #0072C6;
}
.stButton > button, .css-1cys7g1, .css-1n76uvr {
    background-color: #0072C6 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}
.stCameraInput > label > div {
    background-color: #0072C6 !important;
    color: white !important;
    border-radius: 8px;
    padding: 8px 16px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# 2) LOAD THE TRAINED MODEL
# --------------------------
MODEL_PATH = "final_model.keras"

@st.cache_resource
def load_model():
    mdl = tf.keras.models.load_model(MODEL_PATH)
    return mdl

model = load_model()
st.write("**Model output shape:**", model.output_shape)

# -------------------------
# 3) DEFINE CLASS LABELS
# -------------------------
# Use the exact list as produced during training. For 9 classes:
class_labels = [
    "1568-5070-L99",
    "1568-5072-L00",
    "1568-5072-r00",
    "1568-5080-L99",
    "1568-5080-r99",
    "1578-5070-R99",
    "1579-101255",
    "1579-101259",
    "1579-349104"
]

if len(class_labels) != model.output_shape[-1]:
    st.error(f"Mismatch: Model outputs {model.output_shape[-1]} classes but class_labels has {len(class_labels)} items.")

# ----------------------------
# 4) SIDEBAR SETTINGS & GUIDE
# ----------------------------
st.sidebar.title("Settings")
CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=1
)
st.sidebar.write(f"Images with confidence below {CONFIDENCE_THRESHOLD}% will be marked as 'Not recognized'.")
st.sidebar.markdown("---")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Choose **Camera** or **Upload** tab.
2. Capture or upload an image.
3. The model predicts the part code and shows confidence.
4. Check the bar chart for class probabilities.
""")

# --------------------------
# 5) PREDICTION FUNCTION
# --------------------------
def predict_part(image: Image.Image):
    """
    Preprocess the image and perform prediction.
    Returns (predicted_label, confidence, all_confidences).
    """
    # Convert PIL image to NumPy array
    img_array = np.array(image)
    
    # Ensure image has 3 channels
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize image to 300x300 (model expects 300x300)
    TARGET_SIZE = (300, 300)
    img_resized = cv2.resize(img_array, TARGET_SIZE)
    # Use the same preprocessing as during training:
    img_preprocessed = preprocess_input(img_resized.astype("float32"))
    
    # Expand dims to create batch dimension
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    
    with st.spinner("Predicting..."):
        preds = model.predict(img_expanded)[0]  # shape: (num_classes,)
    
    predicted_idx = np.argmax(preds)
    if predicted_idx >= len(class_labels):
        st.error("Predicted index out of range. Check your class_labels.")
        predicted_label = "Unknown"
    else:
        predicted_label = class_labels[predicted_idx]
    
    confidence = preds[predicted_idx] * 100
    all_confidences = preds * 100
    return predicted_label, confidence, all_confidences

# -------------------------------
# 6) MAIN LAYOUT: TABS FOR CAMERA / UPLOAD
# -------------------------------
st.title("🔧 Part Classifier")
st.subheader("Identify parts using live camera or file upload")

tab_camera, tab_upload = st.tabs(["📷 Camera", "📁 Upload"])

with tab_camera:
    st.header("Live Camera")
    camera_image = st.camera_input("Take a photo")
    
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_container_width=True)
        label, conf, all_confs = predict_part(image)
        if conf < CONFIDENCE_THRESHOLD:
            st.error(f"**Not recognized** (Confidence {conf:.2f}%).")
        else:
            st.success(f"**Predicted Part Code:** {label}")
            st.info(f"Confidence: {conf:.2f}%")
        df = pd.DataFrame({"Class": class_labels, "Confidence": all_confs})
        df = df.sort_values("Confidence", ascending=False)
        st.markdown("**Confidence for All Classes:**")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Confidence:Q", title="Confidence (%)"),
            y=alt.Y("Class:N", sort="-x"),
            color="Class:N"
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

with tab_upload:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        label, conf, all_confs = predict_part(image)
        if conf < CONFIDENCE_THRESHOLD:
            st.error(f"**Not recognized** (Confidence {conf:.2f}%).")
        else:
            st.success(f"**Predicted Part Code:** {label}")
            st.info(f"Confidence: {conf:.2f}%")
        df = pd.DataFrame({"Class": class_labels, "Confidence": all_confs})
        df = df.sort_values("Confidence", ascending=False)
        st.markdown("**Confidence for All Classes:**")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Confidence:Q", title="Confidence (%)"),
            y=alt.Y("Class:N", sort="-x"),
            color="Class:N"
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.markdown("<center>Made by DANIEL T - Powered by Streamlit & TensorFlow</center>", unsafe_allow_html=True)
