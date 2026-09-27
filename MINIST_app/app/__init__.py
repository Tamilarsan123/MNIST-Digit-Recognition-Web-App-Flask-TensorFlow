from flask import Flask

from app.routes.main_routes import main_bp
from app.routes.prediction_routes import prediction_bp


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object("config")

    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)

    return app
