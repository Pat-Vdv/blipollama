import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration
import glob
import os

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Specify the directory where your images are
image_dir = "/Photos/"
image_exts = ["jpg", "jpeg", "png"]  # specify the image file extensions to search for

# Open a file to write the captions
with open("captions.txt", "w", encoding="utf-8" ) as caption_file:
# Iterate over each image file in the directory
    for image_ext in image_exts:
        print(os.path.join(image_dir, f"*.{image_ext}"))

        #for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
        for img_path in glob.glob(os.path.join(image_dir, "*")):
            if img_path.lower().endswith(f".{image_ext}"):
                print(img_path)
                try:
                    # Load your image
                    raw_image = Image.open(img_path)

                    # Skip very small images
                    if raw_image.size[0] * raw_image.size[1] < 200:
                        continue

                    raw_image = raw_image.convert("RGB")

                    # Process the image with a text prompt
                    text = "the image of"
                    inputs = processor(images=raw_image, text=text, return_tensors="pt")
                    out = model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                    caption_file.write(f"{img_path}: {caption}\n")
                    print(f"[{img_path}] Caption saved")
                except Exception as e:
                    print(f"[{img_path}] Error: {e}")
                continue
    
