import os
import shutil
import random

source_dir = '../data/raw/dataset_fruit_type'
target_dir = '../data/split/dataset_fruit_type_split'
split_ratio = 0.8  # 80% training, 20% testing

#Create the train and test directories
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    train_count = int(len(images) * split_ratio)

    train_dir = os.path.join(target_dir, 'train', category)
    test_dir = os.path.join(target_dir, 'test', category)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for i, img in enumerate(images):
        src_path = os.path.join(category_path, img)
        if i < train_count:
            dst_path = os.path.join(train_dir, img)
        else:
            dst_path = os.path.join(test_dir, img)
        shutil.copy(src_path, dst_path)

print("Dataset split into train and test folders successfully.")
