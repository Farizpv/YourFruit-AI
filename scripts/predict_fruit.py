import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

#Loading the trained model
model = load_model('fruit_classifier_mobilenetv2.h5')

class_names = ['apple', 'banana', 'grapes', 'guava', 'mango',
               'orange', 'papaya', 'pineapple', 'strawberry', 'watermelon']

IMG_SIZE = (150, 150)

def predict_image(img_path):
    #Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  #Batch details

    #Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    print(f"\nPrediction: {predicted_label.upper()} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    while True: 
        test_img_path = input("Enter the path to the image (or 'q' to quit): ").strip()

        if test_img_path.lower() == 'q':
            print("Exiting prediction tool. Stay cool!")
            break

        if os.path.exists(test_img_path):
            predict_image(test_img_path)
        else:
            print(f"Error: File '{test_img_path}' not found. Please check the path and try again.")
        print("-" * 30)