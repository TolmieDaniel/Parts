import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

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
# 2) LOAD THE TRAINED MODEL & DEFINE LABELS
# --------------------------
# Use a relative path if the model is in the same repo folder.
MODEL_PATH = "final_model_effB3.keras"

@st.cache_resource
def load_model():
    mdl = tf.keras.models.load_model(MODEL_PATH)
    return mdl

model = load_model()
st.write("**Model output shape:**", model.output_shape)

# Define the 9 class labels (make sure the order matches how your dataset was loaded)
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
# 3) DEFINE A VIDEO TRANSFORMER FOR LIVE PREDICTION
# --------------------------
class PartClassifier(VideoTransformerBase):
    def __init__(self):
        # Load the model and define labels once when the transformer is instantiated.
        self.model = model  # Already loaded by st.cache_resource
        self.class_labels = class_labels

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        # Convert frame (BGR) to NumPy array
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize image to 300x300 (model input size)
        img_resized = cv2.resize(img_rgb, (300, 300))
        # Preprocess using EfficientNet's preprocess_input (scales pixels to [-1,1])
        img_preprocessed = preprocess_input(img_resized.astype("float32"))
        # Expand dimensions to create a batch
        img_expanded = np.expand_dims(img_preprocessed, axis=0)
        
        # Predict the class probabilities
        preds = self.model.predict(img_expanded)[0]
        predicted_idx = np.argmax(preds)
        predicted_label = self.class_labels[predicted_idx]
        confidence = preds[predicted_idx] * 100
        
        # Overlay the prediction text on the frame
        text = f"{predicted_label}: {confidence:.1f}%"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

# --------------------------
# 4) LAUNCH THE LIVE VIDEO STREAM
# --------------------------
webrtc_streamer(
    key="part-classifier",
    video_transformer_factory=PartClassifier,
    media_stream_constraints={"video": True, "audio": False}
)
