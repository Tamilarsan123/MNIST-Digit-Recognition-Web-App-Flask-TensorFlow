from flask import Blueprint, jsonify, request

from app.services.image_service import preprocess_image
from app.services.model_service import ModelService
from app.utils.validators import validate_upload

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")
model_service = ModelService()


@prediction_bp.route("/predict", methods=["POST"])
def predict_digit():
    image_file = request.files.get("image")

    if image_file is None or image_file.filename in (None, ""):
        return jsonify({"success": False, "error": "Please upload a valid image."}), 400

    try:
        validated_image = validate_upload(image_file)
        prepared_image = preprocess_image(validated_image)
        digit, confidence, probabilities = model_service.predict(prepared_image)

        return jsonify(
            {
                "success": True,
                "digit": int(digit),
                "confidence": float(confidence),
                "probabilities": probabilities,
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": "Prediction failed. Please try a different image."}), 500
