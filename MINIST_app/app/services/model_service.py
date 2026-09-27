from __future__ import annotations

import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH


class ModelService:
    _model = None

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return cls._model

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Please ensure models/digits_intel.h5 exists before starting the app."
            )

        cls._model = load_model(str(MODEL_PATH))
        return cls._model

    def predict(self, image_input):
        model = self.load_model()
        prediction = model.predict(image_input, verbose=0)[0]
        digit = int(np.argmax(prediction))
        confidence = float(prediction[digit])
        probabilities = {str(index): float(value) for index, value in enumerate(prediction)}
        return digit, confidence, probabilities
