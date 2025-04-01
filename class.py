import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# 1) PAGE CONFIG & CUSTOM STYLING
# -------------------------------
st.set_page_config(
    page_title="Live Part Classifier",
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
</style>
""", unsafe_allow_html=True)

st.title("🔧 Live Part Classifier")
st.subheader("Live predictions using your camera")

# --------------------------
# 2) LOAD THE TRAINED MODEL
# --------------------------
# Use a relative path so it works on Streamlit Cloud.
MODEL_PATH = "final_model_effB3.keras"

@st.cache_resource
def load_model():
    mdl = tf.keras.models.load_model(MODEL_PATH)
    return mdl

model = load_model()
st.write("**Model output shape:**", model.output_shape)

# -------------------------
# 3) DEFINE CLASS LABELS
# -------------------------
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

# --------------------------
# 4) VIDEO TRANSFORMER FOR LIVE PREDICTION
# --------------------------
class PartClassifier(VideoTransformerBase):
    def __init__(self):
        self.model = model  # Already loaded via cache
        self.class_labels = class_labels

    def transform(self, frame):
        # Get the video frame in BGR format
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize image to model input size (300x300)
        img_resized = cv2.resize(img_rgb, (300, 300))
        # Preprocess using EfficientNet's preprocess_input (scales pixels to [-1, 1])
        img_preprocessed = preprocess_input(img_resized.astype("float32"))
        # Expand dimensions to create a batch (1, 300, 300, 3)
        img_expanded = np.expand_dims(img_preprocessed, axis=0)

        # Run prediction
        preds = self.model.predict(img_expanded)[0]
        predicted_idx = np.argmax(preds)
        predicted_label = self.class_labels[predicted_idx]
        confidence = preds[predicted_idx] * 100

        # Overlay prediction text on the original BGR image
        text = f"{predicted_label}: {confidence:.1f}%"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        return img

# --------------------------
# 5) LAUNCH LIVE STREAM
# --------------------------
webrtc_streamer(
    key="part-classifier",
    video_transformer_factory=PartClassifier,
    media_stream_constraints={"video": True, "audio": False}
)
