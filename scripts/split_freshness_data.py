import os
import shutil
import random


def split_dataset(source_dir, target_dir, split_ratio=0.8):
    fruits = os.listdir(source_dir)

    for fruit in fruits:
        fruit_path = os.path.join(source_dir, fruit)
        if not os.path.isdir(fruit_path):
            continue

        freshness_levels = os.listdir(fruit_path)

        for level in freshness_levels:
            class_path = os.path.join(fruit_path, level)
            if not os.path.isdir(class_path):
                continue

            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(images)

            split_point = int(len(images) * split_ratio)
            train_imgs = images[:split_point]
            test_imgs = images[split_point:]

            for img_type, img_list in zip(['train', 'test'], [train_imgs, test_imgs]):
                dest_path = os.path.join(target_dir, img_type, fruit, level)
                os.makedirs(dest_path, exist_ok=True)
                for img in img_list:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(dest_path, img)
                    shutil.copy2(src, dst)

    print("✅ Done splitting the dataset into train/test sets!")


# Paths
source_dir = "../data/raw/dataset_freshness"
target_dir = "../data/split/dataset_freshness_split"

split_dataset(source_dir, target_dir)
