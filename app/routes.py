# app/routes.py
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from flask import Blueprint, render_template, request, jsonify

# Import your combined prediction function
from .predict_combined import predict_combined
# --- NEW: Import freshness_tips from predict_combined.py ---
from .predict_combined import freshness_tips

bp = Blueprint('main', __name__)

# --- Configuration for uploads ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Confidence Threshold for fruit identification ---
FRUIT_CONFIDENCE_THRESHOLD = 0.70 # Adjust this value (e.g., 0.60 to 0.80) based on testing


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Homepage Route ---
@bp.route('/')
def index():
    # --- MODIFIED: Pass the freshness_tips dictionary to the index.html template ---
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
            # Call the combined prediction function
            # It now handles image loading, both predictions, and data lookup
            results = predict_combined(file.read())

            # --- NEW LOGIC FOR UNIDENTIFIED ITEMS (Confidence Thresholding) ---
            # Apply thresholding based on the fruit_confidence returned by predict_combined
            if results['fruit_confidence'] < FRUIT_CONFIDENCE_THRESHOLD:
                results['fruit'] = "Unidentified Item"
                results['freshness'] = "N/A"
                results['freshness_confidence'] = 0.0
                results['shelf_life'] = "N/A"
                results['storage_tip'] = "Please upload an image of a recognized fruit."
                # Also clear ambiguity message if it's an unidentified item
                results['ambiguity_message'] = None
            # --- END NEW LOGIC ---

            # Return the results as JSON
            # Ensure confidence values are standard floats for JSON serialization
            results['fruit_confidence'] = float(results['fruit_confidence'])
            results['freshness_confidence'] = float(results['freshness_confidence'])

            return jsonify(results)

        except Exception as e:
            # Catch any unexpected errors during processing
            print(f"Error during prediction: {e}")  # Log the error for debugging
            return jsonify({"error": f"An error occurred while processing the image: {str(e)}."}), 500

    return jsonify({"error": "Unknown error."}), 400