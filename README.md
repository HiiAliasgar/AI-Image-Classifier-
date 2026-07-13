<h1 align="center">AI Image Classifier</h1>

<p align="center">
  <b>Classify images into real-world categories using deep learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EEF?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
</p>

---

## Overview

An interactive web application that classifies images into real-world categories using a pre-trained deep learning model (MobileNetV2). Simply upload an image and get instant predictions with confidence scores.

### Features

| Feature | Description |
|---------|-------------|
| 🖼️ Image Upload | Drag & drop or browse to upload images |
| 🧠 Deep Learning | MobileNetV2 pre-trained on ImageNet |
| 📊 Confidence Scores | See prediction probabilities for top categories |
| 🎯 1000+ Categories | Recognizes objects from ImageNet classes |
| ⚡ Fast Inference | Real-time classification in the browser |

---

## Demo

```bash
# Install dependencies
pip install streamlit tensorflow opencv-python

# Run the app
streamlit run main.py
```

Then open http://localhost:8501 in your browser.

---

## How It Works

1. **Upload** an image through the Streamlit interface
2. **Preprocess** the image (resize to 224x224, normalize)
3. **Classify** using MobileNetV2 pre-trained on ImageNet
4. **Display** top predictions with confidence scores

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| Streamlit | Web interface |
| TensorFlow | Deep learning framework |
| MobileNetV2 | Pre-trained classification model |
| OpenCV | Image processing |

---

## Project Structure

```
AI-Image-Classifier-/
├── main.py           # Streamlit application
├── pyproject.toml    # Project configuration
├── uv.lock          # Dependency lock file
├── .python-version   # Python version
└── .gitignore        # Git ignore rules
```

---

## Requirements

- Python 3.10+
- Streamlit
- TensorFlow
- OpenCV

---

## Author

**Aliasgar Lohawala** - [@HiiAliasgar](https://github.com/HiiAliasgar)

---

<p align="center">
  Made with ❤️ and TensorFlow
</p>
