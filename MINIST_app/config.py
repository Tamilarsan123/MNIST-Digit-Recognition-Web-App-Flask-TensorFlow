import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "digits_intel.h5"
UPLOAD_FOLDER = BASE_DIR / "uploads"
MAX_IMAGE_SIZE = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

SECRET_KEY = os.environ.get("SECRET_KEY", "digitai-dev-secret")
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

JSON_SORT_KEYS = False
