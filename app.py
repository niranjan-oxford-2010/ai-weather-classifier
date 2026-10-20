from __future__ import annotations

from io import BytesIO

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_ID = "prithivMLmods/Weather-Image-Classification"
WEATHER = {
    "cloudy/overcast": ("Cloudy", "☁️", "Cloud cover or overcast conditions."),
    "foggy/hazy": ("Foggy / hazy", "🌫️", "Reduced visibility from fog or haze."),
    "rain/strom": ("Rainy / stormy", "🌧️", "Rain or storm conditions."),
    "snow/frosty": ("Snowy / frosty", "❄️", "Snow, frost, or icy conditions."),
    "sun/clear": ("Sunny / clear", "☀️", "Mostly clear or sunny conditions."),
}

st.set_page_config(
    page_title="AI Weather Classifier",
    page_icon="🌦️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🌦️ AI Weather Classifier")
st.caption("Take a picture of the sky or choose an image to see what weather pattern the AI recognizes.")

st.info(
    "This is an image classifier for a science demonstration. It describes visible conditions in one photo; "
    "it does not measure temperature, confirm rain outside the frame, or forecast future weather."
)

@st.cache_resource(show_spinner="Loading the weather model (the first load may take a minute)…")
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID)
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
        (labels[int(index)], float(value))
        for value, index in zip(values.tolist(), indices.tolist())
    ]

camera_tab, upload_tab = st.tabs(["📷 Take a photo", "🖼️ Upload an image"])
with camera_tab:
    camera_photo = st.camera_input("Point your camera at the sky")
with upload_tab:
    uploaded_photo = st.file_uploader(
        "Choose a sky photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="For best results, use a clear outdoor image with the sky visible.",
    )

photo = camera_photo if camera_photo is not None else uploaded_photo

if photo is not None:
    try:
        image = Image.open(BytesIO(photo.getvalue())).convert("RGB")
        st.image(image, caption="Image to classify", use_container_width=True)

        if st.button("Analyze weather", type="primary", use_container_width=True):
            with st.spinner("Analyzing the image…"):
                predictions = classify(image)

            label, confidence = predictions[0]
            display_name, icon, explanation = WEATHER.get(
                label.lower(), (label.replace("/", " / ").title(), "🌤️", "")
            )
            st.subheader(f"{icon} {display_name}")
            st.write(explanation)
            st.metric("Model confidence", f"{confidence:.1%}")
            if confidence < 0.45:
                st.warning("The model is uncertain. Try a clearer photo that shows more of the sky.")

            with st.expander("See the other likely labels"):
                for candidate, score in predictions[1:]:
                    name, candidate_icon, _ = WEATHER.get(
                        candidate.lower(), (candidate.replace("/", " / ").title(), "🌤️", "")
                    )
                    st.write(f"{candidate_icon} **{name}** — {score:.1%}")

            st.caption(
                "Confidence is the model's score among its five labels, not a guarantee that the prediction is correct."
            )
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("That image could not be opened. Please choose a JPG, PNG, or WebP image.")
    except Exception as error:  # Surface a useful message if the hosted model cannot load.
        st.error("The model could not be loaded. Please try again in a moment.")
        st.caption(f"Details: {error}")
else:
    st.caption("Your photo is analyzed only when you select **Analyze weather**.")


