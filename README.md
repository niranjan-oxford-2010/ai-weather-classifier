# AI Weather Classifier

A mobile-friendly Streamlit science-exhibition app that classifies a photo of the sky as cloudy/overcast, foggy/hazy, rainy/stormy, snowy/frosty, or sunny/clear. It supports a phone camera and image uploads.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The first analysis downloads the model from Hugging Face and may take a minute. An internet connection is required.

## Model and limitations

The app uses [prithivMLmods/Weather-Image-Classification](https://huggingface.co/prithivMLmods/Weather-Image-Classification), a five-label image classifier. Its model card reports 85.89% accuracy on its evaluation set; that result does not guarantee accuracy on every camera photo. The model card lists the weights under Apache-2.0. The model's five labels are cloudy/overcast, foggy/hazy, rain/storm, snow/frosty, and sun/clear.

This is a demonstration of visible image classification, not a weather station or forecast. A single image cannot confirm conditions outside the frame, measure temperature, or predict future weather. The app does not store uploaded photos; images are processed in the running app session.
