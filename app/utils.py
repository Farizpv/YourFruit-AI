import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

FRUIT_MODEL_PATH = os.path.join("models", "fruit_classifier_mobilenetv2.h5")
FRESHNESS_MODEL_DIR = os.path.join("models", "freshness_models")

def load_models():
    fruit_model = load_model(FRUIT_MODEL_PATH)

    with open("fruit_details.json") as f:
        fruit_labels = json.load(f)["fruit_labels"]

    freshness_models = {}
    freshness_labels = {}
    for filename in os.listdir(FRESHNESS_MODEL_DIR):
        if filename.endswith(".h5"):
            fruit_name = filename.replace("freshness_classifier_", "").replace(".h5", "")
            freshness_models[fruit_name] = load_model(os.path.join(FRESHNESS_MODEL_DIR, filename))
            with open(os.path.join(FRESHNESS_MODEL_DIR, f"freshness_classifier_{fruit_name}_labels.json")) as f:
                freshness_labels[fruit_name] = json.load(f)

    return fruit_model, freshness_models, fruit_labels, freshness_labels

def preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_fruit_type(image, model, labels):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    return labels[np.argmax(prediction)]

def predict_freshness(image, fruit_type, models, labels_dict):
    processed = preprocess_image(image)
    model = models.get(fruit_type)
    labels = labels_dict.get(fruit_type)
    if not model or not labels:
        return "Unknown"

    prediction = model.predict(processed)[0]
    return labels[np.argmax(prediction)]
