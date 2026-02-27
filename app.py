import streamlit as st
import numpy as np
import keras
import gdown
import os
import requests
import base64
import io
import time
from PIL import Image

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = "disc_model_v2.keras"
GDRIVE_FILE_ID = "1xhlk9P9pIRLEdyX23HSMPBC-0Gck6Okm"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Good", "Defective"]
BRIDGE_URL = "https://calathiform-dorsoventral-gavyn.ngrok-free.dev"

# ─────────────────────────────────────────
# DOWNLOAD & LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Downloading model from Google Drive... (first time only)"):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH,
                quiet=False
            )
    return keras.models.load_model(MODEL_PATH, compile=False)

# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def predict(model, image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img, verbose=0)
    confidence = float(np.max(prediction))
    label = CLASS_NAMES[int(np.argmax(prediction))]
    return label, confidence

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.set_page_config(page_title="Disc Inspection", layout="wide", page_icon="🔍")
st.title("🔍 Industrial Disc Inspection — Live Feed")
st.caption(f"Model: `{MODEL_PATH}` | Input: `224x224` | Classes: Good / Defective")

model = load_model()
st.success("✅ Model loaded!")

# Counters
if "good_count" not in st.session_state:
    st.session_state.good_count = 0
if "defective_count" not in st.session_state:
    st.session_state.defective_count = 0
if "total" not in st.session_state:
    st.session_state.total = 0

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera Frame")
    frame_placeholder = st.empty()

with col2:
    st.subheader("🤖 Prediction Result")
    result_placeholder = st.empty()
    confidence_placeholder = st.empty()
    st.divider()
    st.subheader("📊 Stats")
    counter_placeholder = st.empty()

st.divider()
stop = st.button("⛔ Stop Inspection", type="primary")

# ─────────────────────────────────────────
# LIVE LOOP
# ─────────────────────────────────────────
while not stop:
    try:
        response = requests.get(
            f"{BRIDGE_URL}/latest",
            timeout=5,
            headers={"ngrok-skip-browser-warning": "true"}
        )
        data = response.json()

        if data.get("image"):
            image_bytes = base64.b64decode(data["image"])
            label, confidence = predict(model, image_bytes)

            st.session_state.total += 1
            if label == "Defective":
                st.session_state.defective_count += 1
                result_placeholder.error(f"🔴  **{label}**")
            else:
                st.session_state.good_count += 1
                result_placeholder.success(f"🟢  **{label}**")

            frame_placeholder.image(image_bytes, use_column_width=True)
            confidence_placeholder.metric("Confidence", f"{confidence * 100:.1f}%")

            defect_rate = (
                st.session_state.defective_count / st.session_state.total * 100
                if st.session_state.total > 0 else 0
            )
            counter_placeholder.markdown(f"""
| Status | Count |
|---|---|
| ✅ Good | {st.session_state.good_count} |
| ❌ Defective | {st.session_state.defective_count} |
| 🔢 Total | {st.session_state.total} |
| 📉 Defect Rate | {defect_rate:.1f}% |
            """)
        else:
            frame_placeholder.info("⏳ Waiting for camera frame...")

    except Exception as e:
        frame_placeholder.warning(f"⚠️ Bridge not reachable: {e}")

    time.sleep(2)
