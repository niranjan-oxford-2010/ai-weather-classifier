from __future__ import annotations

from io import BytesIO

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "dima806/weather_types_image_detection"
WEATHER = {
    "dew": ("Dew", "💧", "Moisture has collected on a cool surface."),
    "fogsmog": ("Fog / haze", "🌫️", "Fog or haze reduces visibility in the scene."),
    "frost": ("Frost", "❄️", "A surface appears covered with frost."),
    "glaze": ("Freezing glaze", "🧊", "A smooth ice coating is visible."),
    "hail": ("Hail", "🧊", "Hail or ice pellets are visible."),
    "lightning": ("Lightning / storm", "⚡", "A lightning discharge is visible in the image."),
    "rain": ("Rain", "🌧️", "Rain is visible in the scene."),
    "rainbow": ("Rainbow", "🌈", "A rainbow is visible in the image."),
    "rime": ("Rime ice", "🧊", "Rime ice is visible on exposed surfaces."),
    "sandstorm": ("Dust / sandstorm", "🌪️", "Dust or sand appears to be carried by strong wind."),
    "snow": ("Snow", "🌨️", "Snow is visible in the scene."),
}

st.set_page_config(
    page_title="SkyLens · Weather AI",
    page_icon="🌦️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse at 50% 5%, rgba(33, 92, 128, .42), transparent 47%),
          linear-gradient(155deg, #071522 0%, #0a1b2d 48%, #081522 100%);
        color: #f4f8fc;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
        max-width: 850px;
        padding-top: 3.2rem;
        padding-bottom: 4rem;
    }
    .weather-scatter {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
    }
    .weather-photo {
        position: absolute;
        width: clamp(150px, 17vw, 250px);
        aspect-ratio: 1.42;
        border: 1px solid rgba(226, 241, 255, .2);
        border-radius: 20px;
        background-size: cover;
        background-position: center;
        box-shadow: 0 20px 54px rgba(0, 0, 0, .28);
        opacity: .27;
        filter: saturate(.82);
    }
    .weather-photo::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(145deg, rgba(9, 24, 39, .05), rgba(9, 24, 39, .45));
    }
    .weather-photo--storm { left: 3.2vw; top: 13vh; transform: rotate(-7deg); background-image: url('https://upload.wikimedia.org/wikipedia/commons/1/19/Orage-nuit.jpg'); }
    .weather-photo--snow { right: 3.5vw; top: 18vh; transform: rotate(6deg); background-image: url('https://images.unsplash.com/photo-1705448082121-dd6888a685d4?auto=format&fit=crop&w=720&q=68'); }
    .weather-photo--fog { left: 8vw; bottom: 11vh; transform: rotate(5deg); background-image: url('https://images.unsplash.com/photo-1695238070092-3ed5d342af84?auto=format&fit=crop&w=720&q=68'); }
    .weather-photo--sky { right: 7vw; bottom: 10vh; transform: rotate(-6deg); background-image: url('https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=720&q=68'); }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        margin-bottom: .85rem;
        padding: .42rem .75rem;
        border: 1px solid rgba(127, 210, 255, .24);
        border-radius: 999px;
        background: rgba(30, 74, 100, .34);
        color: #a7dcf6;
        font: 600 .78rem 'DM Sans', sans-serif;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 0;
        color: #f5f9ff;
        font: 800 clamp(2.2rem, 5vw, 3.8rem)/1.08 'Manrope', sans-serif;
        letter-spacing: -.055em;
    }
    .hero-copy {
        max-width: 620px;
        margin: .8rem 0 1.5rem;
        color: #b6c8d8;
        font: 400 1.08rem/1.6 'DM Sans', sans-serif;
    }
    [data-testid="stAlert"] { border-radius: 16px; }
    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(133, 198, 235, .45);
        border-radius: 18px;
        background: rgba(16, 38, 58, .82);
    }
    [data-testid="stTabs"] [role="tablist"] {
        gap: .5rem;
        border-bottom-color: rgba(151, 187, 214, .16);
    }
    [data-testid="stTabs"] button[role="tab"] { border-radius: 12px 12px 0 0; }
    [data-testid="stButton"] button {
        min-height: 3.15rem;
        border: 1px solid rgba(119, 210, 255, .24);
        border-radius: 13px;
        background: linear-gradient(110deg, #17a8b5, #2584cb);
        color: white;
        font: 700 1rem 'DM Sans', sans-serif;
        box-shadow: 0 10px 30px rgba(14, 134, 177, .2);
    }
    [data-testid="stMetric"] {
        padding: 1rem 1.15rem;
        border: 1px solid rgba(172, 208, 233, .16);
        border-radius: 16px;
        background: rgba(13, 35, 54, .78);
    }
    [data-testid="stImage"] img { border-radius: 18px; }
    @media (max-width: 900px) {
        .weather-photo { width: 150px; opacity: .14; }
        .weather-photo--storm { left: -40px; top: 10vh; }
        .weather-photo--snow { right: -38px; top: 14vh; }
        .weather-photo--fog { left: -48px; bottom: 8vh; }
        .weather-photo--sky { right: -42px; bottom: 8vh; }
    }
    @media (max-width: 600px) {
        .weather-photo { width: 110px; opacity: .12; border-radius: 14px; }
        [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
        .hero-copy { font-size: .98rem; }
    }
    </style>
    <div class="weather-scatter" aria-hidden="true">
      <div class="weather-photo weather-photo--storm"></div>
      <div class="weather-photo weather-photo--snow"></div>
      <div class="weather-photo weather-photo--fog"></div>
      <div class="weather-photo weather-photo--sky"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">✦ AI sky analysis · 11 visual labels</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Read the sky.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Upload a photo or take one to see which weather patterns the image model recognizes, from lightning and rain to fog, hail, and snow.</p>',
    unsafe_allow_html=True,
)
st.info(
    "A visual demonstration only: one image cannot measure local conditions, confirm what is outside the frame, or forecast future weather. Clear or overcast skies may not match this model's labels."
)


@st.cache_resource(show_spinner="Loading the weather vision model for its first use…")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify(image: Image.Image):
    processor, model = load_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probabilities, k=min(3, probabilities.numel()))
    labels = model.config.id2label
    return [
        (labels[int(index)].lower(), float(value))
        for value, index in zip(values.tolist(), indices.tolist())
    ]


camera_tab, upload_tab = st.tabs(["📷 Take a photo", "🖼️ Upload an image"])
with camera_tab:
    camera_photo = st.camera_input("Point your camera at the sky")
with upload_tab:
    uploaded_photo = st.file_uploader(
        "Choose a weather photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best results, use a clear outdoor image where the weather is visible.",
    )

photo = camera_photo if camera_photo is not None else uploaded_photo

if photo is not None:
    try:
        image = Image.open(BytesIO(photo.getvalue())).convert("RGB")
        st.image(image, caption="Your image", use_container_width=True)

        if st.button("Analyze photo", type="primary", use_container_width=True):
            with st.spinner("Looking for visible weather patterns…"):
                predictions = classify(image)

            label, confidence = predictions[0]
            display_name, icon, explanation = WEATHER.get(
                label, (label.replace("_", " ").replace("/", " / ").title(), "🌤️", "")
            )
            if confidence < 0.35:
                st.subheader("🔎 No clear match")
                st.write("This photo may show a condition outside the model's labels, or the scene may be difficult to identify.")
                st.metric("Top suggestion score", f"{confidence:.1%}")
                st.warning(f"The model's low-confidence top suggestion is **{icon} {display_name}**. Treat it as uncertain.")
            else:
                st.subheader(f"{icon} {display_name}")
                st.write(explanation)
                st.metric("Model score", f"{confidence:.1%}")
            if 0.35 <= confidence < 0.45:
                st.warning("The model is uncertain. A clearer image with the weather pattern in view may help.")

            with st.expander("Other likely labels"):
                for candidate, score in predictions[1:]:
                    name, candidate_icon, _ = WEATHER.get(
                        candidate, (candidate.replace("_", " ").replace("/", " / ").title(), "🌤️", "")
                    )
                    st.write(f"{candidate_icon} **{name}** — {score:.1%}")

            st.caption(
                "Scores rank the model's 11 labels; they are not calibrated probabilities or a guarantee of correct identification."
            )
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("That image could not be opened. Please choose a JPG, PNG, or WebP image.")
    except Exception as error:
        st.error("The model could not be loaded. Please try again in a moment.")
        st.caption(f"Details: {error}")
else:
    st.caption("Your photo is only analyzed after you select **Analyze photo**.")

