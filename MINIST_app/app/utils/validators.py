import os

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE


def validate_upload(file_storage):
    if file_storage is None or not file_storage.filename:
        raise ValueError("Please upload a valid image.")

    filename = secure_filename(file_storage.filename)
    if not filename or filename in {".", ".."}:
        raise ValueError("Please upload a valid image.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP.")

    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)

    if file_size == 0:
        raise ValueError("Uploaded file is empty.")
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError("Image exceeds the 5 MB limit.")

    try:
        file_storage.seek(0)
        with Image.open(file_storage) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise ValueError("The uploaded file is not a valid image.")

    file_storage.seek(0)
    return Image.open(file_storage)
