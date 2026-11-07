from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


def load_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Validate a bounded image upload and return an EXIF-corrected RGB image."""
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 10 MB upload limit.")
    try:
        with Image.open(BytesIO(file_bytes)) as probe:
            if probe.format not in SUPPORTED_FORMATS:
                raise ValueError("Only JPEG, PNG and WebP images are supported.")
            probe.verify()
        with Image.open(BytesIO(file_bytes)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid supported image.") from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prepare an image for MobileNetV2: RGB, 224x224, [-1, 1], batch dimension."""
    rgb_image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb_image, dtype=np.float32)
    pixels = pixels / 127.5 - 1.0
    return np.expand_dims(pixels, axis=0)
