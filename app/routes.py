import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from flask import Blueprint, render_template, request, jsonify
from .predict_combined import predict_combined
from .predict_combined import freshness_tips

bp = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

FRUIT_CONFIDENCE_THRESHOLD = 0.70 #70%

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/')
def index():
    return render_template('index.html', fruit_data=freshness_tips)


@bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        try:
            results = predict_combined(file.read())

            if results['fruit_confidence'] < FRUIT_CONFIDENCE_THRESHOLD:
                results['fruit'] = "Unidentified Item"
                results['freshness'] = "N/A"
                results['freshness_confidence'] = 0.0
                results['shelf_life'] = "N/A"
                results['storage_tip'] = "Please upload an image of a recognized fruit."
                #ambiguity message if it's an unidentified item
                results['ambiguity_message'] = None

            results['fruit_confidence'] = float(results['fruit_confidence'])
            results['freshness_confidence'] = float(results['freshness_confidence'])

            return jsonify(results)

        except Exception as e:
            #Catch any unexpected errors during processing
            print(f"Error during prediction: {e}")  #Log the error for debugging
            return jsonify({"error": f"An error occurred while processing the image: {str(e)}."}), 500

    return jsonify({"error": "Unknown error."}), 400