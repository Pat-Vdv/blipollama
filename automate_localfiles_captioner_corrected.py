import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"
IMAGE_DIR = Path("/Photos/")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # set for faster lookup
OUT_FILE = Path("captions.txt")

def iter_images(folder: Path, exts: set[str]):
    # Parcours récursif: remplace rglob par glob si tu ne veux pas de sous-dossiers
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()

    prompt = "a photo of"  # souvent plus naturel que "the image of"

    with OUT_FILE.open("w", encoding="utf-8") as f, torch.no_grad():
        for img_path in iter_images(IMAGE_DIR, IMAGE_EXTS):
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")

                    # skip small images (ex: < 32x32)
                    if im.size[0] < 32 or im.size[1] < 32:
                        continue

                    inputs = processor(images=im, text=prompt, return_tensors="pt").to(device)

                    out = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3,       # qualité ↑ (un peu plus lent)
                        do_sample=False
                    )
                    caption = processor.decode(out[0], skip_special_tokens=True)

                f.write(f"{img_path}: {caption}\n")
                print(f"[OK] {img_path.name} -> {caption}")

            except Exception as e:
                print(f"[ERR] {img_path} -> {e}")

if __name__ == "__main__":
    main()

    
