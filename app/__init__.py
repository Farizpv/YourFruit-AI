# app/__init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)

    # Import and register blueprints or routes here
    from . import routes
    app.register_blueprint(routes.bp)

    return app