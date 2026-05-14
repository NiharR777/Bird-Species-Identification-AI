import streamlit as st
import numpy as np
import tempfile
import os

from tensorflow.keras.models import load_model
from feature_extraction import extract_mel


# ===============================
# Load Model & Classes
# ===============================

MODEL_PATH = "models/bird_model.h5"
CLASSES_PATH = "features/classes.npy"

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)


# ===============================
# Page Config
# ===============================

st.set_page_config(
    page_title="Bird Species AI",
    page_icon="🐦",
    layout="centered"
)


# ===============================
# Custom CSS (UI Polish)
# ===============================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    text-align: center;
    color: #FFFFFF;
}

.upload-box {
    border: 2px dashed #ff4b4b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.result-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ===============================
# Title Section
# ===============================

st.markdown("<h1>🐦 Bird Species Identification System</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;'>Upload a bird sound and let AI identify the species</p>",
    unsafe_allow_html=True
)

st.divider()


# ===============================
# Upload Section
# ===============================

st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload Audio File",
    type=["wav", "mp3", "ogg"]
)

st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# Prediction
# ===============================

if uploaded_file is not None:

    # Show audio player
    st.audio(uploaded_file)

    with st.spinner("🔍 Analyzing audio... Please wait"):

        # Save temp file (Windows Safe)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(uploaded_file.read())
            temp_path = temp.name


        # Feature extraction
        mel = extract_mel(temp_path)
        mel = mel[np.newaxis, ..., np.newaxis]


        # Predict
        prediction = model.predict(mel)

        index = np.argmax(prediction)

        bird = classes[index]
        confidence = float(np.max(prediction))


        # Remove temp file
        os.remove(temp_path)


    # ===============================
    # Result Display
    # ===============================

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    st.success("✅ Prediction Complete!")

    st.markdown(f"### 🐤 Bird Species: **{bird}**")

    st.markdown(f"### 📊 Confidence: **{confidence*100:.2f}%**")

    st.progress(confidence)

    st.markdown("</div>", unsafe_allow_html=True)



# ===============================
# Footer
# ===============================

st.markdown("""
<div class="footer">
<hr>
Developed by Nihar | Bird AI System <br>
Powered by Deep Learning & Streamlit
</div>
""", unsafe_allow_html=True)
