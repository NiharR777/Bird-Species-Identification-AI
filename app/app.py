import streamlit as st
import numpy as np
import tempfile
import os

from tensorflow.keras.models import load_model
from feature_extraction import extract_mel

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Bird Species AI",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

/* ================================================= */
/* GLOBAL SETTINGS */
/* ================================================= */

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: #050816;
    color: white;
}

/* FULL WIDTH FIX */

.block-container {
    max-width: 100% !important;
    padding-top: 1rem;
    padding-left: 5rem;
    padding-right: 5rem;
    padding-bottom: 2rem;
}

section.main > div {
    max-width: 100%;
}

/* MAIN BACKGROUND */

.stApp {
    background:
    radial-gradient(circle at top left, #0f172a 0%, #050816 45%),
    radial-gradient(circle at bottom right, #111827 0%, #050816 40%);
}

/* Hide Streamlit */

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ================================================= */
/* HERO SECTION */
/* ================================================= */

.hero-container {
    padding-top: 40px;
    padding-bottom: 30px;
    text-align: center;
}

.hero-badge {

    display: inline-block;

    padding: 12px 24px;

    border-radius: 999px;

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    color: #38bdf8;

    font-size: 15px;

    margin-bottom: 35px;

    backdrop-filter: blur(10px);
}

.hero-title {

    font-size: 110px;

    font-weight: 800;

    line-height: 1.05;

    color: white;

    margin-bottom: 25px;
}

.hero-gradient {

    background: linear-gradient(
        90deg,
        #38bdf8,
        #8b5cf6,
        #ec4899
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero-subtitle {

    font-size: 24px;

    color: #94a3b8;

    max-width: 900px;

    margin: auto;

    line-height: 1.8;
}

/* ================================================= */
/* MAIN CARD */
/* ================================================= */

.main-card {

    width: 100%;

    margin-top: 55px;

    padding: 60px;

    border-radius: 35px;

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    box-shadow:
    0px 0px 80px rgba(56,189,248,0.08),
    0px 0px 30px rgba(139,92,246,0.05);
}

/* ================================================= */
/* SECTION TITLE */
/* ================================================= */

.section-title {

    font-size: 38px;

    font-weight: 700;

    margin-bottom: 25px;

    color: white;
}

/* ================================================= */
/* RESULT CARD */
/* ================================================= */

.result-card {

    margin-top: 50px;

    padding: 50px;

    border-radius: 35px;

    background:
    linear-gradient(
        145deg,
        rgba(17,24,39,0.95),
        rgba(30,41,59,0.95)
    );

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0px 0px 50px rgba(0,255,255,0.08);

    text-align: center;
}

.result-title {

    font-size: 34px;

    color: #38bdf8;

    font-weight: 700;
}

.result-bird {

    font-size: 75px;

    font-weight: 800;

    margin-top: 20px;

    margin-bottom: 20px;

    color: white;
}

.result-confidence {

    font-size: 32px;

    color: #22c55e;

    font-weight: 700;
}

/* ================================================= */
/* INFO GRID */
/* ================================================= */

.info-grid {

    display: flex;

    gap: 25px;

    margin-top: 50px;
}

.info-card {

    flex: 1;

    background: rgba(255,255,255,0.04);

    border-radius: 25px;

    padding: 35px;

    border: 1px solid rgba(255,255,255,0.06);

    text-align: center;

    transition: 0.3s;
}

.info-card:hover {

    transform: translateY(-8px);

    box-shadow:
    0px 0px 40px rgba(56,189,248,0.15);
}

.info-number {

    font-size: 55px;

    font-weight: 800;

    color: #38bdf8;
}

.info-label {

    font-size: 20px;

    color: #94a3b8;

    margin-top: 12px;
}

/* ================================================= */
/* FOOTER */
/* ================================================= */

.footer {

    margin-top: 80px;

    text-align: center;

    color: #64748b;

    font-size: 16px;

    padding-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================

model = load_model("models/bird_model.h5")

classes = np.load(
    "features/classes.npy",
    allow_pickle=True
)

# ======================================================
# HERO SECTION
# ======================================================

st.markdown("""

<div class="hero-container">

<div class="hero-badge">
🚀 AI Powered Bird Recognition System
</div>

<div class="hero-title">

Identify Bird Species <br>

<span class="hero-gradient">
Using Deep Learning
</span>

</div>

<div class="hero-subtitle">

Upload bird sounds and let Artificial Intelligence
analyze audio frequencies using Mel Spectrograms
and Convolutional Neural Networks.

</div>

</div>

""", unsafe_allow_html=True)

# ======================================================
# MAIN CARD
# ======================================================

st.markdown('<div class="main-card">', unsafe_allow_html=True)

left, right = st.columns([1.4, 1])

# ======================================================
# LEFT COLUMN
# ======================================================

with left:

    st.markdown(
        '<div class="section-title">🎵 Upload Bird Audio</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=["wav", "mp3", "ogg"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

# ======================================================
# RIGHT COLUMN
# ======================================================

with right:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3069/3069172.png",
        width=420
    )

# ======================================================
# PREDICTION
# ======================================================

if uploaded_file is not None:

    with st.spinner("🧠 AI is analyzing bird sound..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".ogg"
        ) as tmp:

            tmp.write(uploaded_file.read())

            temp_path = tmp.name

        mel = extract_mel(temp_path)

        if mel is not None:

            mel = mel[np.newaxis, ..., np.newaxis]

            prediction = model.predict(mel)

            index = np.argmax(prediction)

            bird = classes[index]

            confidence = np.max(prediction)

            st.markdown(f"""

            <div class="result-card">

            <div class="result-title">
            🎯 Prediction Result
            </div>

            <div class="result-bird">
            {bird}
            </div>

            <div class="result-confidence">
            Confidence: {confidence*100:.2f}%
            </div>

            </div>

            """, unsafe_allow_html=True)

            st.progress(float(confidence))

            st.balloons()

        else:

            st.error("Feature extraction failed!")

        try:
            os.remove(temp_path)
        except:
            pass

# ======================================================
# INFO SECTION
# ======================================================

st.markdown("""

<div class="info-grid">

<div class="info-card">
<div class="info-number">21</div>
<div class="info-label">Bird Species</div>
</div>

<div class="info-card">
<div class="info-number">3616</div>
<div class="info-label">Training Audios</div>
</div>

<div class="info-card">
<div class="info-number">54%</div>
<div class="info-label">Model Accuracy</div>
</div>

</div>

""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================

st.markdown("""

<div class="footer">

🚀 Developed by Nihar <br>
Bird Species Identification using Deep Learning,
Mel Spectrograms & CNN

</div>

""", unsafe_allow_html=True)