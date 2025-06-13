import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import time

def train_freshness_model(fruit_name, batch_size=32, img_size=(150,150), initial_epochs=10, fine_tune_epochs=20):
    base_dir = '../data/split/dataset_freshness_split'
    train_dir = os.path.join(base_dir, 'train', fruit_name)
    test_dir = os.path.join(base_dir, 'test', fruit_name)

    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        horizontal_flip=True,
        rotation_range=30,
        zoom_range=0.2,
        shear_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    #Check if there are enough classes for the fruit
    if train_generator.num_classes < 2:
        print(f"Skipping {fruit_name}: Not enough freshness categories (found {train_generator.num_classes}). Need at least 2.")
        return 

    num_classes = train_generator.num_classes

    #Build model with MobileNetV2 base
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),  
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

    print(f"Starting initial training for {initial_epochs} epochs...")
    
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=initial_epochs,
        callbacks=[early_stop, reduce_lr]
    )

    #Fine-tuning unfreeze last 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    #lower LR for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Starting fine-tuning for {fine_tune_epochs} epochs...")
    #Adjust fine-tuning epochs
    total_fine_tune_epochs = initial_epochs + fine_tune_epochs
    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=total_fine_tune_epochs,
        initial_epoch=history.epoch[-1], #Starting from where initial training left
        callbacks=[early_stop, reduce_lr]
    )

    #Save model
    model.save(f'freshness_classifier_{fruit_name}.h5')
    print(f'Model for {fruit_name} saved!')

    #saving labels
    with open(f'freshness_classifier_{fruit_name}_labels.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Label mapping for {fruit_name} saved!")

if __name__ == "__main__":
    base_train_path = '../data/split/dataset_freshness_split/train'
    base_test_path = '../data/split/dataset_freshness_split/test'

    train_fruits = set(f for f in os.listdir(base_train_path) if os.path.isdir(os.path.join(base_train_path, f)))
    test_fruits = set(f for f in os.listdir(base_test_path) if os.path.isdir(os.path.join(base_test_path, f)))

    common_fruits = sorted(list(train_fruits.intersection(test_fruits)))
    total_fruits = len(common_fruits)

    print(f"\nFruits to be trained (exist in both train & test): {common_fruits}\n")

    total_start_time = time.time()

    for i, fruit in enumerate(common_fruits, 1):
        print(f"\n[{i}/{total_fruits}] Training model for: {fruit.upper()}")

        start_time = time.time()

        try:
            train_freshness_model(fruit)
        except Exception as e:
            print(f"Failed to train model for {fruit}: {e}")

        elapsed = time.time() - start_time
        print(f"Time taken for {fruit}: {elapsed:.2f} seconds")

        # ETA calculation
        avg_time_per_fruit = (time.time() - total_start_time) / i
        remaining_fruits = total_fruits - i
        eta = avg_time_per_fruit * remaining_fruits
        print(f"Estimated time remaining: {eta / 60:.2f} minutes")

    total_elapsed = time.time() - total_start_time
    print(f"\nAll done in {total_elapsed / 60:.2f} minutes!")