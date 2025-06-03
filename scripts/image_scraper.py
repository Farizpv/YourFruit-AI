import os
import hashlib
import requests
from PIL import Image
from io import BytesIO
import time
import random

# For DuckDuckGo Search
from duckduckgo_search import DDGS

# For Google Image Search with Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import urllib.parse  # For URL encoding

# --- Configuration ---
BASE_DOWNLOAD_DIR = "../data/raw_new_downloads"

# List of fruits to download. Add specific queries for problem areas.
FRUIT_QUERIES = [
    "apple fruit", "ripe apple", "unripe apple",
    "orange fruit", "ripe orange", "unripe orange",
    "banana fruit", "ripe banana", "unripe banana",
    "mango fruit", "unripe mango", "green mango", "raw mango",  # Focused on mango confusion
    "dragonfruit fruit", "ripe dragonfruit",
    "grapes fruit", "green grapes", "red grapes",
    "guava fruit", "unripe guava", "green guava",  # Focused on guava confusion
    "kiwi fruit", "ripe kiwi", "unripe kiwi",
    "papaya fruit", "unripe papaya",
    "pineapple fruit", "ripe pineapple",
    "strawberry fruit", "unripe strawberry",
    "watermelon fruit", "unripe watermelon",
    "pomegranate fruit",
]

MAX_IMAGES_PER_QUERY = 100  # Target number of images per query

# Your existing fruit classes (ensure this matches your model's classes)
FRUIT_CLASSES = ['apple', 'banana', 'dragonfruit', 'grapes', 'guava', 'kiwi',
                 'mango', 'orange', 'papaya', 'pineapple', 'pomegranate', 'strawberry', 'watermelon']


# --- Helper Functions ---

def download_image_from_url(url, save_path, query_name):
    """Helper to download and save a single image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.save(save_path, format='JPEG')
        return True
    except requests.exceptions.RequestException as e:
        print(f"[{query_name}] Network error downloading {url}: {e}")
    except Image.UnidentifiedImageError:
        print(f"[{query_name}] Failed to identify image from {url}")
    except Exception as e:
        print(f"[{query_name}] General error downloading {url}: {e}")
    return False


def download_images_ddg(query, save_dir, max_images):
    os.makedirs(save_dir, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        downloaded = 0
        for i, result in enumerate(results):
            url = result.get("image")
            if not url:
                continue
            filename = f"{query.replace(' ', '_').replace('/', '_')}_{i}.jpg"
            save_path = os.path.join(save_dir, filename)
            if download_image_from_url(url, save_path, query):
                downloaded += 1
            time.sleep(0.5 + random.random() * 0.5)
    print(f"✅ DuckDuckGo - {query}: Downloaded {downloaded} images.")
    return downloaded


def download_images_google_selenium(query, save_dir, max_images):
    os.makedirs(save_dir, exist_ok=True)
    downloaded = 0

    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode (no GUI)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Google Images search URL
        search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(query)}"
        driver.get(search_url)

        # Scroll down to load more images
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2 + random.random() * 2)  # Give time for content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # Try clicking the "Show more results" button if it exists
                try:
                    show_more_button = driver.find_element(By.CSS_SELECTOR, 'input[value="Show more results"]')
                    show_more_button.click()
                    time.sleep(2 + random.random() * 2)  # Wait for new results
                    new_height = driver.execute_script("return document.body.scrollHeight")  # Check height again
                    if new_height == last_height:  # If still no new content, break
                        break
                except Exception:
                    break  # No more results button or already clicked

            last_height = new_height
            if downloaded >= max_images:  # Break early if we have enough images
                break

        # Extract image URLs
        img_elements = driver.find_elements(By.CSS_SELECTOR, 'img.Q4LuWd')  # This CSS selector might change!
        urls = []
        for img in img_elements:
            src = img.get_attribute('src')
            if src and src.startswith('http') and not src.startswith('data:image'):
                urls.append(src)
                if len(urls) >= max_images:
                    break

        # Download images
        for i, url in enumerate(urls):
            if downloaded >= max_images:
                break
            filename = f"{query.replace(' ', '_').replace('/', '_')}_{i}.jpg"
            save_path = os.path.join(save_dir, filename)
            if download_image_from_url(url, save_path, query):
                downloaded += 1
            time.sleep(0.1 + random.random() * 0.1)  # Faster download once URLs are collected

    except Exception as e:
        print(f"🔥 Google Images - Error during Selenium search for '{query}': {e}")
    finally:
        if 'driver' in locals() and driver:
            driver.quit()  # Always close the browser

    print(f"✅ Google Images - {query}: Downloaded {downloaded} images.")
    return downloaded


def remove_duplicates(folder_path):
    hashes = set()
    removed = 0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                if not os.path.isfile(filepath):
                    continue
                with open(filepath, "rb") as f:
                    filehash = hashlib.md5(f.read()).hexdigest()
                if filehash in hashes:
                    os.remove(filepath)
                    removed += 1
                else:
                    hashes.add(filehash)
            except Exception as e:
                print(f"Error reading or deleting {filepath}: {e}")
    print(f"🗑️ Removed {removed} duplicates in {folder_path}")


# --- MAIN EXECUTION ---
# Ensure the base download directory exists
os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

# Choose your image source here: 'ddg' or 'google'
# WARNING: 'google' is experimental and may fail frequently!
IMAGE_SOURCE = 'ddg'  # Set this to 'google' if you want to try Google Images

for query in FRUIT_QUERIES:
    # Determine the fruit name from the query for folder naming
    fruit_name_for_folder = None
    for known_fruit in FRUIT_CLASSES:
        if known_fruit in query.lower():  # Check if known fruit is in query (case-insensitive)
            fruit_name_for_folder = known_fruit
            break

    if fruit_name_for_folder is None:
        print(f"Skipping query '{query}': Could not determine fruit folder name from known classes.")
        continue

    # Create a subfolder for each fruit within the BASE_DOWNLOAD_DIR
    save_folder = os.path.join(BASE_DOWNLOAD_DIR, fruit_name_for_folder)

    if IMAGE_SOURCE == 'ddg':
        download_images_ddg(query, save_folder, max_images=MAX_IMAGES_PER_QUERY)
    elif IMAGE_SOURCE == 'google':
        download_images_google_selenium(query, save_folder, max_images=MAX_IMAGES_PER_QUERY)
    else:
        print("Invalid IMAGE_SOURCE specified. Please choose 'ddg' or 'google'.")
        break  # Exit if source is invalid

    # Remove duplicates within the newly downloaded batch (per query)
    remove_duplicates(save_folder)

print("🎉 Done downloading and cleaning new dataset!")
print(f"New images are in: {BASE_DOWNLOAD_DIR}")
print("NEXT STEP: Manually inspect and integrate these into your main training/testing dataset.")