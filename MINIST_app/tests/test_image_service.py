import io

import numpy as np
from PIL import Image

from app.services.image_service import preprocess_image


def test_preprocess_image_returns_model_compatible_shape():
    image = Image.new("L", (64, 64), color=255)
    image.putpixel((10, 10), 0)
    image = image.resize((28, 28))

    arr = preprocess_image(image)

    assert arr.shape == (1, 28, 28)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0
