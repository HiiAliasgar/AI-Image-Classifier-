# AI Image Classifier

A Streamlit application that classifies uploaded images using the MobileNetV2 model pretrained on ImageNet and displays the highest-scoring predicted categories.

## Features

- Upload an image through the Streamlit interface.
- Preprocess images for the model input.
- Run inference with MobileNetV2.
- Display top predictions and their confidence scores.

## How it works

1. The user uploads an image.
2. The application resizes and normalizes the image for model input.
3. MobileNetV2 produces class scores.
4. The interface displays the top predicted ImageNet labels.

## Technology

- Python 3.10+
- Streamlit
- TensorFlow
- MobileNetV2
- OpenCV

## Run locally

```bash
git clone https://github.com/HiiAliasgar/AI-Image-Classifier-.git
cd AI-Image-Classifier-
pip install streamlit tensorflow opencv-python
streamlit run main.py
```

Open `http://localhost:8501` in your browser.

## Project files

```text
main.py
pyproject.toml
uv.lock
.python-version
.gitignore
```

## Model scope

MobileNetV2 is pretrained on ImageNet. Predictions are limited to the classes represented by that model and can be unreliable for images outside its training distribution. Confidence scores should not be interpreted as calibrated probabilities.

[Source code](https://github.com/HiiAliasgar/AI-Image-Classifier-)
