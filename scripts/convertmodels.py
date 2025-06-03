import tensorflow as tf
import os

# Path to your fruit classification model (.h5 file)
fruit_model_path = '../models/fruit_classifier_mobilenetv2.h5'
fruit_tflite_path = 'fruit_classifier_mobilenetv2.tflite'

# Convert the fruit classification model
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(fruit_model_path))
    tflite_model = converter.convert()

    # Save the converted model
    with open(fruit_tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"✅ Fruit classification model converted and saved to: {fruit_tflite_path}")

except FileNotFoundError:
    print(f"⚠️ Fruit model not found at: {fruit_model_path}. Make sure the path is correct.")
except Exception as e:
    print(f"⚠️ An error occurred during fruit model conversion: {e}")

# Path to the directory containing freshness models (relative to the current script)
freshness_models_dir = '../models/freshness_models'

# Loop through all files in the freshness models directory
for filename in os.listdir(freshness_models_dir):
    if filename.endswith('.h5') and filename.startswith('freshness_classifier_'):
        freshness_model_path = os.path.join(freshness_models_dir, filename)
        # Ensure the output .tflite file also goes into the freshness_models directory
        freshness_tflite_path = os.path.join(freshness_models_dir, filename.replace('.h5', '.tflite'))

        try:
            print(f"\n🔄 Attempting to convert: {freshness_model_path}")
            converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(freshness_model_path))
            tflite_model = converter.convert()

            # Save the converted model
            with open(freshness_tflite_path, 'wb') as f:
                f.write(tflite_model)

            print(f"✅ Converted and saved to: {freshness_tflite_path}")

        except FileNotFoundError:
            print(f"⚠️ Model not found at: {freshness_model_path}. Skipping.")
        except Exception as e:
            print(f"⚠️ An error occurred during conversion of {freshness_model_path}: {e}")

print("\n✅ Conversion process for all freshness models (if found) completed.")