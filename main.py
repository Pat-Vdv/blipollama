"""
CLI to caption images in a directory using either:
  - BLIP (Transformers/PyTorch)
  - Ollama multimodal model (e.g., llava)

Examples:
  python main.py --backend blip --dir /Photos/ --ext jpg jpeg png
  python main.py --backend ollama --dir /Photos/ --ext jpg png --ollama-model llava
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image

import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

from blipollama import VisionCaptionService
from blipollama.models import CaptionBackend


class BlipBackend:
    name = "blip"

    def __init__(
        self,
        *,
        model_id: str,
        prompt: str = "a photo of",
        max_new_tokens: int = 50,
        device: str | None = None,
        min_side: int = 32,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.min_side = min_side

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    def caption(self, image_path: Path) -> str:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            if im.size[0] < self.min_side or im.size[1] < self.min_side:
                return ""

            with torch.no_grad():
                inputs = self.processor(images=im, text=self.prompt, return_tensors="pt").to(self.device)
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                return self.processor.decode(out[0], skip_special_tokens=True).strip()


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        *,
        model: str,
        prompt: str = "Describe the image",
        max_new_tokens: int = 80,
        url: str = "http://localhost:11434",
        timeout_s: int = 120,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def caption(self, image_path: Path) -> str:
        with image_path.open("rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": self.max_new_tokens},
        }

        r = requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption images in a directory (BLIP or Ollama).")

    p.add_argument("--backend", choices=["blip", "ollama"], required=True, help="Captioning backend to use.")
    p.add_argument("--dir", required=True, help="Directory containing images.")
    p.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png"], help="Extensions to include (e.g. jpg png).")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    p.add_argument("--out", default="captions.txt", help="Output text file.")

    # BLIP options
    p.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base", help="BLIP model id.")
    p.add_argument("--blip-prompt", default="a photo of", help="BLIP prompt.")
    p.add_argument("--blip-max-new-tokens", type=int, default=50, help="BLIP max_new_tokens.")
    p.add_argument("--blip-device", default=None, help="BLIP device: cuda/cpu (default auto).")
    p.add_argument("--min-side", type=int, default=32, help="Skip images smaller than this (width/height).")

    # Ollama options
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
    p.add_argument("--ollama-model", default="llava", help="Ollama vision model (e.g., llava, moondream).")
    p.add_argument("--ollama-prompt", default="Describe the image", help="Ollama prompt.")
    p.add_argument("--ollama-max-new-tokens", type=int, default=80, help="Ollama output token budget.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds for Ollama.")

    return p.parse_args()


def build_backend(args: argparse.Namespace) -> CaptionBackend:
    if args.backend == "blip":
        return BlipBackend(
            model_id=args.blip_model,
            prompt=args.blip_prompt,
            max_new_tokens=args.blip_max_new_tokens,
            device=args.blip_device,
            min_side=args.min_side,
        )
    return OllamaBackend(
        model=args.ollama_model,
        prompt=args.ollama_prompt,
        max_new_tokens=args.ollama_max_new_tokens,
        url=args.ollama_url,
        timeout_s=args.timeout,
    )


def main() -> None:
    args = parse_args()

    image_dir = Path(args.dir)
    if not image_dir.exists():
        raise SystemExit(f"Directory not found: {image_dir}")

    backend = build_backend(args)

    svc = VisionCaptionService(backend=backend, recursive=args.recursive)
    results = svc.caption_directory(image_dir, extensions=args.ext)

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r.image_path}: {r.caption}\n")

    print(f"OK: {len(results)} captions written to {out_path} (backend={backend.name})")


if __name__ == "__main__":
    main()
