# app/predict_combined.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
# Important: Use the specific preprocess_input for MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import json


# Get the absolute path to the directory where THIS SCRIPT (predict_combined.py) is located
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Update Paths ---
# FRUIT_MODEL_PATH: It's in FruitAI/models, so relative to current_script_dir (app/) it's ../models/
FRUIT_MODEL_PATH = os.path.join(current_script_dir, '../models/fruit_classifier_mobilenetv2.h5')

# FRESHNESS_MODELS_DIR: Same as above
FRESHNESS_MODELS_DIR = os.path.join(current_script_dir, '../models/freshness_models')

# fruit_details.json: It's in FruitAI/, so relative to current_script_dir (app/) it's ../
FRUIT_DETAILS_PATH = os.path.join(current_script_dir, '../fruit_details.json')

# Load freshness advice data once
try:
    with open(FRUIT_DETAILS_PATH, "r", encoding='utf-8') as f:
        freshness_tips = json.load(f)
except FileNotFoundError:
    print(f"Error: fruit_details.json not found at {FRUIT_DETAILS_PATH}. Please check the path.")
    freshness_tips = {} # Initialize as empty to prevent crashes

# --- IMPORTANT: Ensure this matches the training class_names list EXACTLY ---
# Based on your predict_fruit.py, assuming this is the correct list
FRUIT_CLASSES = ['apple', 'banana', 'grapes', 'guava', 'mango',
                 'orange', 'papaya', 'pineapple', 'strawberry', 'watermelon']
# If your fruit_classifier_mobilenetv2.h5 was trained on more (like dragonfruit, kiwi, pomegranate),
# you MUST update this list to reflect that EXACTLY.

# Image size for models
IMG_SIZE = (150, 150)

# Helper: Load and preprocess image
def load_and_prepare_image(image_bytes, target_size=IMG_SIZE):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(target_size)

        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)

        # --- CRITICAL FIX: Use MobileNetV2's specific preprocessing ---
        img_array = preprocess_input(img_array) # This handles scaling to [-1, 1]

        return img_array, img
    except Exception as e:
        raise ValueError(f"Could not load or prepare image from bytes: {e}")

# Step 1: Predict fruit type
def predict_fruit_type(img_array):
    try:
        model = load_model(FRUIT_MODEL_PATH)
        preds = model.predict(img_array, verbose=0) # Added verbose=0 to suppress output
        predicted_index = np.argmax(preds)
        return FRUIT_CLASSES[predicted_index], preds[0][predicted_index], preds[0]
    except Exception as e:
        print(f"Error predicting fruit type: {e}")
        # Return sensible defaults if model loading or prediction fails
        return "unknown", 0.0, np.zeros(len(FRUIT_CLASSES))


def predict_freshness(fruit, img_array):
    model_path = os.path.join(FRESHNESS_MODELS_DIR, f'freshness_classifier_{fruit}.h5')
    labels_path = os.path.join(FRESHNESS_MODELS_DIR, f'freshness_classifier_{fruit}_labels.json')

    # Default number of freshness classes (e.g., 'ripe', 'unripe', 'rotten')
    # This is a safe fallback for np.zeros, but if a model is found,
    # the actual num_classes will come from its labels.
    DEFAULT_NUM_FRESHNESS_CLASSES = 3

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print(f"Warning: No freshness model or labels available for {fruit} at {model_path} or {labels_path}")
        return "unknown", 0.0, np.zeros(DEFAULT_NUM_FRESHNESS_CLASSES)

    try:
        model = load_model(model_path)

        with open(labels_path, 'r') as f:
            class_indices = json.load(f)

        index_to_label = {int(v): k for k, v in class_indices.items()} # Ensure key is int for lookup

        preds_array = model.predict(img_array, verbose=0)[0] # Added verbose=0
        pred_index = np.argmax(preds_array)

        # Ensure pred_index is a valid key in index_to_label
        if pred_index in index_to_label:
            label = index_to_label[pred_index]
            confidence = preds_array[pred_index]
            return label, confidence, preds_array
        else:
            print(f"Error: Predicted index {pred_index} not found in freshness labels for {fruit}.")
            return "unknown", 0.0, np.zeros(len(class_indices)) # Fallback to actual num classes if labels loaded

    except Exception as e:
        print(f"Error predicting freshness for {fruit}: {e}")
        return "unknown", 0.0, np.zeros(DEFAULT_NUM_FRESHNESS_CLASSES)


def predict_combined(img_bytes):
    try:
        img_array, img = load_and_prepare_image(img_bytes)
    except ValueError as e:
        return {"error": str(e)}

    fruit_name_from_model, fruit_conf, all_fruit_probs = predict_fruit_type(img_array)
    fruit_name_from_model = fruit_name_from_model.lower()
    print(f"\n🟢 Predicted Fruit: {fruit_name_from_model} (Confidence: {fruit_conf*100:.1f}%)")

    freshness_label_from_model, freshness_conf, all_freshness_probs = predict_freshness(fruit_name_from_model, img_array)
    freshness_label_lower = freshness_label_from_model.lower()
    print(f"🔵 Freshness: {freshness_label_from_model} (Confidence: {freshness_conf*100:.1f}%)")

    freshness_key_map = {
        "fresh": "ripe",
        "good": "ripe",
        "bad": "rotten",
        "overripe": "overripe",
        "ripe": "ripe",     # Redundant but harmless
        "unripe": "unripe", # Redundant but harmless
        "rotten": "overripe"  # Redundant but harmless
    }

    effective_freshness_key = freshness_key_map.get(freshness_label_lower, freshness_label_lower)

    tips_data = freshness_tips.get(fruit_name_from_model, {}).get(effective_freshness_key)

    shelf_life_str = "N/A"
    storage_tip_str = "No specific tip available."

    if tips_data:
        print(f"🕒 Shelf Life: {tips_data.get('shelf_life', 'N/A')}")
        print(f"📦 Storage Tip: {tips_data.get('storage_tips', 'No specific tip available.')}")
        shelf_life_str = tips_data.get('shelf_life', "N/A")
        storage_tip_str = tips_data.get('storage_tips', "No specific tip available.")
    else:
        print(f"ℹ️ No specific storage info found for {fruit_name_from_model} with freshness {freshness_label_from_model} (mapped to '{effective_freshness_key}'). Please check fruit_details.json.")


    # --- POST-PROCESSING LOGIC STARTS HERE ---
    ambiguity_message = None
    confused_threshold_percentage = 10.0 # If confidence difference is less than 10%
    min_confidence_for_ambiguity_percentage = 50.0 # Both must be somewhat confident to be ambiguous

    # Scenario: Unripe Guava vs. Unripe Mango Confusion
    # Ensure 'mango' and 'guava' are actually in FRUIT_CLASSES before attempting indexing
    if fruit_name_from_model == "guava" and freshness_label_lower == "unripe" and \
       'mango' in FRUIT_CLASSES and 'guava' in FRUIT_CLASSES:
        try:
            mango_idx = FRUIT_CLASSES.index('mango')
            mango_confidence_raw = all_fruit_probs[mango_idx] * 100

            guava_confidence_raw = fruit_conf * 100

            print(f"DEBUG: Guava confidence: {guava_confidence_raw:.1f}%, Mango confidence: {mango_confidence_raw:.1f}%")

            if abs(guava_confidence_raw - mango_confidence_raw) < confused_threshold_percentage and \
               guava_confidence_raw > min_confidence_for_ambiguity_percentage and \
               mango_confidence_raw > min_confidence_for_ambiguity_percentage:
                ambiguity_message = "Heads up! This looks like an unripe guava, but it's pretty close to an unripe mango. Sometimes they can be tricky to tell apart!"
                print(f"DEBUG: Ambiguity detected between unripe guava ({guava_confidence_raw:.1f}%) and unripe mango ({mango_confidence_raw:.1f}%).")
        except (ValueError, IndexError) as e: # Catch if index is out of bounds or fruit not found
            print(f"WARNING: Error checking mango-guava ambiguity: {e}. Check FRUIT_CLASSES consistency.")

    # --- POST-PROCESSING LOGIC ENDS HERE ---

    return {
        "fruit": fruit_name_from_model,
        "fruit_confidence": fruit_conf*100,
        "freshness": freshness_label_from_model,
        "freshness_confidence": freshness_conf*100,
        "shelf_life": shelf_life_str,
        "storage_tip": storage_tip_str,
        "ambiguity_message": ambiguity_message
    }

# The __main__ block for CLI testing
if __name__ == "__main__":
    try:
        img_path = input("🔍 Enter image path for CLI test: ").strip()
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found at {img_path}")

        with open(img_path, 'rb') as f:
            img_bytes_for_cli = f.read()

        results = predict_combined(img_bytes_for_cli)

        if "error" in results:
            print(f"\n--- CLI Error ---")
            print(results["error"])
        else:
            print("\n--- CLI Output ---")
            print(f"Fruit: {results['fruit'].title()} ({results['fruit_confidence']:.1f}%)")
            print(f"Freshness: {results['freshness'].title()} ({results['freshness_confidence']:.1f}%)")
            print(f"Shelf Life: {results['shelf_life']}")
            print(f"Storage Tip: {results['storage_tip']}")
            if results['ambiguity_message']:
                print(f"🚨 Ambiguity: {results['ambiguity_message']}")

            # For CLI execution, we still show the plot for visual confirmation
            # Note: predict_combined doesn't return the PIL Image object, so re-load it for plt.imshow
            try:
                img_for_plot = Image.open(io.BytesIO(img_bytes_for_cli)).convert("RGB")
                plt.imshow(img_for_plot)
                plt.title(f"{results['fruit'].title()} - {results['freshness'].title()}")
                plt.axis('off')
                plt.show()
            except Exception as plot_e:
                print(f"Could not display image for plot: {plot_e}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CLI test: {e}")