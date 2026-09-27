# DigitAI — MNIST Handwritten Digit Recognition

A production-style Flask application for recognizing handwritten digits from uploaded images using a trained MNIST Keras model.

## Features

- Upload a digit image
- Validate file type and size securely
- Preprocess images using the same pipeline as the notebook
- Predict digit with confidence score
- Clean and modern AI dashboard UI

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Then open http://localhost:5000

## Model

The app expects a trained model at `models/digits_intel.h5`.
