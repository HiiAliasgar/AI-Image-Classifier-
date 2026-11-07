import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from image_utils import MAX_IMAGE_BYTES, load_uploaded_image, preprocess_image


def encoded_image(mode="RGB", fmt="PNG"):
    image = Image.new(mode, (16, 12), color=0)
    output = BytesIO()
    image.save(output, format=fmt)
    return output.getvalue()


class ImageInputTests(unittest.TestCase):
    def test_loads_rgb_png(self):
        image = load_uploaded_image(encoded_image())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (16, 12))

    def test_converts_rgba_and_grayscale_to_rgb(self):
        self.assertEqual(load_uploaded_image(encoded_image("RGBA")).mode, "RGB")
        self.assertEqual(load_uploaded_image(encoded_image("L")).mode, "RGB")

    def test_rejects_invalid_image(self):
        with self.assertRaisesRegex(ValueError, "not a valid supported image"):
            load_uploaded_image(b"not an image")

    def test_rejects_oversized_image(self):
        with self.assertRaisesRegex(ValueError, "10 MB"):
            load_uploaded_image(b"x" * (MAX_IMAGE_BYTES + 1))

    def test_preprocessing_shape_and_range(self):
        batch = preprocess_image(Image.new("L", (8, 8), color=255))
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertTrue(np.isfinite(batch).all())
        self.assertGreaterEqual(float(batch.min()), -1.0)
        self.assertLessEqual(float(batch.max()), 1.0)
        self.assertAlmostEqual(float(batch.mean()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
