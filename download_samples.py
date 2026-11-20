"""
Downloads high-quality standard ImageNet benchmark sample images for testing the Adversarial Playground.
Uses reliable GitHub raw sources from ImageNet sample repository.
"""

import os
import urllib.request

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(SAMPLE_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"

SAMPLES = {
    "stop_sign.jpg": f"{BASE_URL}/n06794110_street_sign.JPEG",
    "giant_panda.jpg": f"{BASE_URL}/n02510455_giant_panda.JPEG",
    "tabby_cat.jpg": f"{BASE_URL}/n02123045_tabby.JPEG",
    "lemon.jpg": f"{BASE_URL}/n07749582_lemon.JPEG",
    "dog.jpg": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
}

def download_all():
    print(f"Downloading sample images to {SAMPLE_DIR}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    for filename, url in SAMPLES.items():
        out_path = os.path.join(SAMPLE_DIR, filename)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response, open(out_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"  ✓ Successfully downloaded {filename}")
        except Exception as e:
            print(f"  ⚠ Failed downloading {filename} ({e})")

if __name__ == "__main__":
    download_all()
