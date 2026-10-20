# AI Weather Classifier

A mobile-friendly Streamlit app that identifies visible weather patterns in a photo. It supports a phone camera and image uploads, and uses a fine-tuned 11-label vision model with a dedicated lightning class.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads the model from Hugging Face and may take a minute. An internet connection is required.

## Model and limitations

The app uses [dima806/weather_types_image_detection](https://huggingface.co/dima806/weather_types_image_detection), a ViT image classifier with 11 labels: dew, fog/smog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, and snow. Its model card reports 95.8% accuracy and 95.9% macro F1 on a balanced evaluation set of 3,190 images (290 per class), including 1.00 F1 for lightning. These are model-card results from its evaluation set, not a guarantee for every real-world photo. The model card lists Apache-2.0 licensing.

### Local scenario check

The candidate model was run locally on five public example photos before publishing. Top predictions were lightning for the storm photo (77.7%), rain for heavy rain (48.7%), fog/smog for fog (29.6%), snow for a snowy scene (60.0%), and lightning for a sunny sky (23.5%). The app treats scores below 35% as “No clear match” and shows the weak guess as uncertain; clear and overcast skies are not classes in this model. This small spot-check is not a substitute for a representative test set.

The interface uses subtle, scattered weather photographs from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Orage-nuit.jpg) and [Unsplash](https://unsplash.com/) for decoration. They are loaded remotely and are not used as model inputs.

This is a demonstration of visible image classification, not a weather station or forecast. A single image cannot confirm conditions outside the frame, measure temperature, or predict future weather. The model's scores are ranking scores, not calibrated probabilities. Uploaded photos are processed in the running app session and are not intentionally saved by this app.

