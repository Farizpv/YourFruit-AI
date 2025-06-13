from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os
import shutil

fruit_freshness = {
    'apple': ['fresh', 'rotten'],
    'banana': ['unripe', 'ripe', 'overripe'],
    'dragonfruit': ['fresh', 'overripe'],
    'grapes': ['fresh', 'rotten'],
    'guava': ['unripe', 'ripe', 'rotten'],
    'kiwi': ['unripe', 'ripe', 'overripe'],
    'mango': ['unripe', 'ripe', 'overripe'],
    'orange': ['fresh', 'rotten'],
    'papaya': ['unripe', 'ripe', 'rotten'],
    'pineapple': ['unripe', 'ripe', 'overripe'],
    'pomegranate': ['fresh', 'old'],
    'strawberry': ['fresh', 'rotten'],
    'watermelon': ['fresh', 'rotten']
}

base_dir = "../data/raw/dataset_freshness"
os.makedirs(base_dir, exist_ok=True)

def is_valid_image(path):
    try:
        if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return False
        if os.path.getsize(path) < 10 * 1024:
            return False
        Image.open(path).verify()
        return True
    except:
        return False

for fruit, stages in fruit_freshness.items():
    for stage in stages:
        query = f"{stage} {fruit}"
        download_dir = os.path.join(base_dir, fruit, stage)
        os.makedirs(download_dir, exist_ok=True)

        print(f"\nDownloading: {query}")
        crawler = GoogleImageCrawler(storage={'root_dir': download_dir})
        crawler.crawl(keyword=query, max_num=50)

        # Clean bad files
        for file in os.listdir(download_dir):
            file_path = os.path.join(download_dir, file)
            if not is_valid_image(file_path):
                os.remove(file_path)
                print(f"Removed invalid: {file_path}")

print("\nAll images downloaded and cleaned successfully!")
