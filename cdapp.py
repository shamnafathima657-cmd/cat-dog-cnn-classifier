"""
Streamlit app — Cat vs Dog Classifier (Dark Theme)

Run with:
    streamlit run app.py

Expects the trained model file in the same folder (saved by the notebook):
    - cat_dog_cnn.keras

No technical knowledge needed to use this app — just upload a photo and
read the result.
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Basic setup
# ---------------------------------------------------------------------------
MODEL_PATH = "cat_dog_cnn.keras"
IMG_SIZE = (128, 128)
IDX_TO_LABEL = {0: "cat", 1: "dog"}  # matches class_indices printed in the notebook

st.set_page_config(
    page_title="Cat or Dog?",
    page_icon="🐾",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Theme — midnight studio: near-black background, warm amber accent for dog,
# cool violet accent for cat.
#   Background (base) : #14121A
#   Panel              : #1D1A24
#   Ink (text)         : #ECE8F2
#   Muted text         : #8B8699
#   Dog accent (amber) : #F2A33C
#   Cat accent (violet): #8B7CF6
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 0%, #1c1828 0%, #14121A 55%, #0F0D14 100%);
    color: #ECE8F2;
}

.paw-title {
    font-family: 'Fredoka', sans-serif;
    font-size: 2.7rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.1rem;
    background: linear-gradient(90deg, #F2A33C, #8B7CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.paw-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    color: #8B8699;
    text-align: center;
    margin-bottom: 2rem;
}

.result-card {
    border-radius: 20px;
    padding: 1.8rem 1.8rem;
    margin-top: 1.2rem;
    text-align: center;
    background: #1D1A24;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
}
.result-card.dog {
    border: 2px solid #F2A33C;
    box-shadow: 0 0 0 1px rgba(242, 163, 60, 0.15), 0 8px 28px rgba(242, 163, 60, 0.12);
}
.result-card.cat {
    border: 2px solid #8B7CF6;
    box-shadow: 0 0 0 1px rgba(139, 124, 246, 0.15), 0 8px 28px rgba(139, 124, 246, 0.12);
}
.result-emoji {
    font-size: 3.2rem;
    line-height: 1;
}
.result-label {
    font-family: 'Fredoka', sans-serif;
    font-size: 1.9rem;
    font-weight: 600;
    margin: 0.3rem 0 0.1rem 0;
}
.result-label.dog { color: #F2A33C; }
.result-label.cat { color: #8B7CF6; }

.result-confidence {
    font-size: 1rem;
    color: #8B8699;
    margin-bottom: 0.4rem;
}

[data-testid="stFileUploader"] {
    background: #1D1A24;
    border: 2px dashed #44405A;
    border-radius: 16px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #F2A33C;
}
section[data-testid="stFileUploaderDropzone"] {
    background: transparent;
}

div[data-testid="stProgress"] > div > div > div {
    background-image: linear-gradient(90deg, #8B7CF6, #F2A33C);
}

[data-testid="stMarkdownContainer"] p {
    color: #ECE8F2;
}

.paw-footer {
    text-align: center;
    color: #5E5A70;
    font-size: 0.85rem;
    margin-top: 2.5rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, pil_img: Image.Image):
    arr = preprocess(pil_img)
    prob = float(model.predict(arr, verbose=0)[0][0])
    pred_idx = int(prob > 0.5)
    label = IDX_TO_LABEL[pred_idx]
    confidence = prob if pred_idx == 1 else 1 - prob
    return label, confidence


st.markdown('<div class="paw-title">🐾 Cat or Dog?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="paw-subtitle">Upload a photo and I\'ll tell you what I see !</div>',
    unsafe_allow_html=True,
)

try:
    model = load_trained_model()
except Exception as e:
    st.error(
        "I couldn't find the trained model file (`cat_dog_cnn.keras`). "
        "Please make sure it's saved in the same folder as this app, then refresh the page.\n\n"
        f"Details: {e}"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a photo of a cat or a dog",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your photo", use_container_width=True)

    with st.spinner("Taking a look... 🔍"):
        label, confidence = predict(model, image)

    emoji = "🐶" if label == "dog" else "🐱"
    friendly = label.capitalize()

    st.markdown(
        f"""
        <div class="result-card {label}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label {label}">It's a {friendly}!</div>
            <div class="result-confidence">I'm {confidence:.0%} sure about this</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(confidence, 0.0), 1.0))
else:
    st.info("👆 Pick a photo above to get started — drag and drop works too!")

st.markdown(
    '<div class="paw-footer">Built with a simple CNN model trained on cat and dog photos 🐾</div>',
    unsafe_allow_html=True,
)