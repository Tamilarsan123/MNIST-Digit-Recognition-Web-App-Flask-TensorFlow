from __future__ import annotations

import os

import numpy as np
from PIL import Image


def preprocess_image(image):
    if image is None:
        raise ValueError("Please upload a valid image.")

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened_image:
            return _prepare_array(opened_image)

    if isinstance(image, Image.Image):
        return _prepare_array(image)

    if hasattr(image, "read"):
        image.seek(0)
        with Image.open(image) as opened_image:
            return _prepare_array(opened_image)

    raise ValueError("Please upload a valid image.")


def _prepare_array(image):
    gray_image = image.convert("L")
    resized_image = gray_image.resize((28, 28))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)
