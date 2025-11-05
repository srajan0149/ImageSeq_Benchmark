import os
from PIL import Image

input_dir = "../img"   # folder with original images
output_dir = "images" # folder for resized images
max_width = 1280             # new max width (you can change)
max_height = 720             # new max height (you can change)

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            continue

        with Image.open(input_path) as img:
            # keep aspect ratio
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            img.save(output_path)
            print(f"Resized: {filename} → {img.size}")
