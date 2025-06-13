import os
import hashlib
import requests
from PIL import Image
from io import BytesIO
import time
import random
from duckduckgo_search import DDGS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import urllib.parse 

BASE_DOWNLOAD_DIR = "../data/raw_new_downloads"

FRUIT_QUERIES = [
    "apple fruit", "ripe apple", "unripe apple",
    "orange fruit", "ripe orange", "unripe orange",
    "banana fruit", "ripe banana", "unripe banana",
    "mango fruit", "unripe mango", "green mango", "raw mango",  
    "dragonfruit fruit", "ripe dragonfruit",
    "grapes fruit", "green grapes", "red grapes",
    "guava fruit", "unripe guava", "green guava",  
    "kiwi fruit", "ripe kiwi", "unripe kiwi",
    "papaya fruit", "unripe papaya",
    "pineapple fruit", "ripe pineapple",
    "strawberry fruit", "unripe strawberry",
    "watermelon fruit", "unripe watermelon",
    "pomegranate fruit",
]

MAX_IMAGES_PER_QUERY = 100 


FRUIT_CLASSES = ['apple', 'banana', 'dragonfruit', 'grapes', 'guava', 'kiwi',
                 'mango', 'orange', 'papaya', 'pineapple', 'pomegranate', 'strawberry', 'watermelon']


def download_image_from_url(url, save_path, query_name):
    """Helper to download and save a single image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
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
    print(f"DuckDuckGo - {query}: Downloaded {downloaded} images.")
    return downloaded


def download_images_google_selenium(query, save_dir, max_images):
    os.makedirs(save_dir, exist_ok=True)
    downloaded = 0

    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        f"user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        #Google Images search
        search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(query)}"
        driver.get(search_url)

        
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2 + random.random() * 2) 
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    show_more_button = driver.find_element(By.CSS_SELECTOR, 'input[value="Show more results"]')
                    show_more_button.click()
                    time.sleep(2 + random.random() * 2)  
                    new_height = driver.execute_script("return document.body.scrollHeight") 
                    if new_height == last_height:  
                        break
                except Exception:
                    break  

            last_height = new_height
            if downloaded >= max_images: 
                break

        img_elements = driver.find_elements(By.CSS_SELECTOR, 'img.Q4LuWd')  
        urls = []
        for img in img_elements:
            src = img.get_attribute('src')
            if src and src.startswith('http') and not src.startswith('data:image'):
                urls.append(src)
                if len(urls) >= max_images:
                    break

        for i, url in enumerate(urls):
            if downloaded >= max_images:
                break
            filename = f"{query.replace(' ', '_').replace('/', '_')}_{i}.jpg"
            save_path = os.path.join(save_dir, filename)
            if download_image_from_url(url, save_path, query):
                downloaded += 1
            time.sleep(0.1 + random.random() * 0.1) 

    except Exception as e:
        print(f"Google Images - Error during Selenium search for '{query}': {e}")
    finally:
        if 'driver' in locals() and driver:
            driver.quit() 

    print(f"Google Images - {query}: Downloaded {downloaded} images.")
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
    print(f"Removed {removed} duplicates in {folder_path}")


os.makedirs(BASE_DOWNLOAD_DIR, exist_ok=True)

IMAGE_SOURCE = 'ddg'  #set to duckduckgo

for query in FRUIT_QUERIES:
    fruit_name_for_folder = None
    for known_fruit in FRUIT_CLASSES:
        if known_fruit in query.lower():  #Checking if known fruit is in query 
            break

    if fruit_name_for_folder is None:
        print(f"Skipping query '{query}': Could not determine fruit folder name from known classes.")
        continue

    #Create a subfolder for each fruit within the BASE_DOWNLOAD_DIR
    save_folder = os.path.join(BASE_DOWNLOAD_DIR, fruit_name_for_folder)

    if IMAGE_SOURCE == 'ddg':
        download_images_ddg(query, save_folder, max_images=MAX_IMAGES_PER_QUERY)
    elif IMAGE_SOURCE == 'google':
        download_images_google_selenium(query, save_folder, max_images=MAX_IMAGES_PER_QUERY)
    else:
        print("Invalid IMAGE_SOURCE specified. Please choose 'ddg' or 'google'.")
        break  

    #Remove duplicates within the newly downloaded batch
    remove_duplicates(save_folder)

print("Done downloading and cleaning new dataset!")
print(f"New images are in: {BASE_DOWNLOAD_DIR}")