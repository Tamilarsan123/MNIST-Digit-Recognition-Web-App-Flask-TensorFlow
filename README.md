# DigitAI — MNIST Handwritten Digit Recognition

A production-style Flask web application that classifies handwritten digit images (0–9) using a Keras neural network trained on the MNIST dataset. Users upload an image, the backend runs it through the same preprocessing pipeline used during training, and the app returns the predicted digit with a confidence score and full per-class probability breakdown.

**Live demo:** _add your deployed URL here_
**Repository:** https://github.com/Tamilarsan123/MNIST-Digit-Recognition-Web-App-Flask-TensorFlow

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

DigitAI wraps a trained MNIST classifier in a clean Flask application built with a modular, service-oriented architecture rather than a single monolithic `app.py`. The project separates concerns into routes, services, and utilities so the codebase is easy to test, extend, and deploy.

**How it works:**

1. User uploads a digit image (PNG / JPG / JPEG / WEBP) through the browser.
2. The frontend sends the file to the backend via `fetch` (no page reload).
3. Flask validates the file (type, size, integrity).
4. The image service converts it to grayscale, resizes it to 28×28, and normalizes pixel values.
5. The model service loads the trained Keras model **once** at startup and runs inference.
6. The API returns the predicted digit, a confidence score, and the full 10-class probability distribution as JSON.
7. The frontend renders the result with an animated confidence bar.

## Features

- 🖼️ Image upload with instant client-side preview
- 🧠 Inference via a pre-trained Keras/TensorFlow model (loaded once, reused across requests)
- 📊 Confidence score returned for every prediction, plus raw softmax probabilities for all 10 classes
- 🔒 Server-side upload validation: extension allow-list, 5 MB size cap, `secure_filename`, and image-integrity verification via Pillow
- 🧹 One-click Clear/Reset without a full page reload
- 📱 Responsive, modern "AI dashboard" UI
- 🧪 Pytest test suite covering routes and the preprocessing pipeline
- 🚀 Gunicorn-ready for production deployment (via `Procfile`)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, Flask 3.1 |
| ML / Inference | TensorFlow 2.16, Keras 3.0 |
| Image processing | Pillow, NumPy |
| WSGI server | Gunicorn |
| Frontend | HTML5, CSS3 (CSS custom properties), vanilla JavaScript (`fetch` API) |
| Testing | Pytest |
| Model training | Jupyter Notebook (`MINIST Digits Classification.ipynb`) |

## Model Architecture

The model is a simple fully-connected (dense) neural network trained on the classic 60,000-image MNIST training set, defined and trained in `train_model.py` / the accompanying notebook:

```
Input:            28 × 28 grayscale image (flattened to 784 values)
Hidden Layer 1:    Dense(100, activation="relu")
Hidden Layer 2:    Dense(100, activation="relu")
Output Layer:      Dense(10, activation="softmax")

Optimizer:  RMSprop
Loss:       sparse_categorical_crossentropy
Metric:     sparse_categorical_accuracy
```

The trained weights are shipped in the repo at `MINIST_app/models/digits_intel.h5` and are loaded lazily on first request, then cached for the lifetime of the process (`ModelService`).

> ⚠️ The model expects input shaped `(1, 28, 28)` — **not** `(1, 28, 28, 1)` — because it uses `Flatten()` directly on a 2D array. Don't change the preprocessing output shape without retraining the model.

## Preprocessing Pipeline

Every uploaded image is processed identically to how the training data was prepared, in `app/services/image_service.py`:

```python
image = image.convert("L")                     # grayscale
image = image.resize((28, 28))                  # match MNIST dimensions
array = np.array(image, dtype=np.float32)
array = array / 255.0                            # normalize to [0, 1]
array = np.expand_dims(array, axis=0)            # add batch dimension -> (1, 28, 28)
```

## Project Structure

```
MINIST_app/
├── app/
│   ├── __init__.py              # Flask application factory
│   ├── routes/
│   │   ├── main_routes.py       # "/" — renders the UI
│   │   └── prediction_routes.py # "/api/predict" — inference endpoint
│   ├── services/
│   │   ├── image_service.py     # preprocessing pipeline
│   │   └── model_service.py     # model loading + inference (singleton)
│   ├── utils/
│   │   └── validators.py        # upload validation (type, size, integrity)
│   ├── templates/
│   │   ├── base.html
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── models/
│   └── digits_intel.h5          # trained Keras model
├── tests/
│   ├── test_routes.py
│   ├── test_prediction.py
│   └── test_image_service.py
├── MINIST Digits Classification.ipynb   # training notebook
├── train_model.py               # standalone training script
├── config.py                    # app configuration
├── run.py                       # local dev entrypoint
├── requirements.txt
├── Procfile                     # gunicorn start command
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+ (developed/tested on 3.12)
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tamilarsan123/MNIST-Digit-Recognition-Web-App-Flask-TensorFlow.git
cd MNIST-Digit-Recognition-Web-App-Flask-TensorFlow/MINIST_app

# 2. Create and activate a virtual environment
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python run.py
```

Then open **http://127.0.0.1:5000** in your browser.

> The app requires a trained model at `models/digits_intel.h5`. This file is already included in the repository — no separate download step is needed unless you retrain the model.

## API Reference

### `POST /api/predict`

Predicts the digit in an uploaded image.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | Yes | PNG, JPG, JPEG, or WEBP — max 5 MB |

**Success response** — `200 OK`

```json
{
  "success": true,
  "digit": 7,
  "confidence": 0.9821,
  "probabilities": {
    "0": 0.0001, "1": 0.0002, "2": 0.0003, "3": 0.0010, "4": 0.0001,
    "5": 0.0020, "6": 0.0001, "7": 0.9821, "8": 0.0030, "9": 0.0111
  }
}
```

**Error responses**

| Status | Cause | Example body |
|---|---|---|
| `400` | No file / empty file / wrong type / oversized / corrupted image | `{"success": false, "error": "Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP."}` |
| `500` | Model file missing, or an unexpected inference error | `{"success": false, "error": "Prediction failed. Please try a different image."}` |

Internal exception details are never exposed to the client; only user-safe messages are returned.

### `GET /`

Renders the web UI (upload form, preview, and results panel).

## Running Tests

```bash
pip install pytest   # already listed in requirements.txt
pytest
```

The suite covers:
- Home page returns `200`
- `/api/predict` rejects a request with no image (`400`)
- Preprocessed image array has shape `(1, 28, 28)`, dtype `float32`, and values in `[0, 1]`

## Deployment

The app ships with a `Procfile` for Gunicorn-based deployment (Render, Railway, Heroku-style platforms, etc.):

```
web: gunicorn run:app
```

**General deployment steps:**

1. Push the repository to GitHub.
2. Create a new web service on your hosting platform and point it at this repo (root: `MINIST_app/` if the platform requires a subdirectory).
3. Set the Python runtime version (3.10+).
4. Install dependencies: `pip install -r requirements.txt`.
5. Confirm `models/digits_intel.h5` is present in the deployed build — it's required at runtime and is already tracked in the repo.
6. Set environment variables (see below).
7. Start command: `gunicorn run:app`.
8. Verify `/` loads and `POST /api/predict` returns a valid prediction.

> The Flask development server (`python run.py`) is for local development only — never use it in production. Gunicorn (or an equivalent WSGI server) handles that role here.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `digitai-dev-secret` | Flask secret key. **Override this in production.** |
| `FLASK_DEBUG` | `0` | Set to `1` to enable debug mode locally. Must be `0`/unset in production. |

Set these in your hosting platform's environment/config panel rather than committing them to source control.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError: Model file not found` | Ensure `models/digits_intel.h5` exists relative to `config.py`'s `BASE_DIR`. |
| `413`/upload rejected unexpectedly | File exceeds the 5 MB limit defined in `config.py` (`MAX_IMAGE_SIZE`). |
| `Unsupported file type` error on a valid-looking image | Only `.png`, `.jpg`, `.jpeg`, `.webp` extensions are allowed — check the file extension, not just the content. |
| Predictions look inconsistent for inverted (white-on-black) images | The model was trained on MNIST's black-background/white-digit convention; images with the opposite polarity aren't auto-inverted in the current pipeline (see Roadmap). |
| TensorFlow import errors on install | Confirm your Python version and OS are compatible with `tensorflow==2.16.2`; on Apple Silicon or ARM you may need `tensorflow-macos`/`tensorflow-aarch64` equivalents. |

## Roadmap

Planned/possible improvements beyond the current implementation:

- [ ] Visual per-digit probability bars in the UI (the API already returns all 10 class probabilities)
- [ ] Drag-and-drop upload support
- [ ] `/health` endpoint for uptime/deployment health checks
- [ ] Automatic polarity detection/inversion for white-digit-on-dark-background uploads
- [ ] Dockerfile for containerized deployment
- [ ] CI workflow (GitHub Actions) to run `pytest` on every push
- [ ] `.env.example` and structured config classes (Development/Production/Testing)

## License

Released under the [MIT License](./LICENSE).

---

Built by [Tamilarasan K](https://github.com/Tamilarasan123) · [LinkedIn](https://www.linkedin.com/in/tamilarasan-k-03a27834b/)
