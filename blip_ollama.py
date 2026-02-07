from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import requests
from PIL import Image

# BLIP deps (chargées seulement si besoin)
try:
    import torch
    from transformers import AutoProcessor, BlipForConditionalGeneration
except Exception:
    torch = None
    AutoProcessor = None
    BlipForConditionalGeneration = None


Backend = Literal["blip", "ollama"]

blip_model  : str = "Salesforce/blip-image-captioning-base"
ollama_model: str = "llava"


@dataclass
class CaptionResult:
    image_path: Path
    backend: Backend
    caption: str


class VisionCaptionService:

    def __init__(
        self,
        backend: Backend = "blip",
        *,
        # Common
        prompt: str = "a photo of",
        max_new_tokens: int = 50,
        recursive: bool = False,

        # BLIP
        blip_model_id,
        device: Optional[str] = None,

        # Ollama
        ollama_url: str = "http://localhost:11434",
        ollama_model,
        timeout_s: int = 120,
    ):
        self.backend = backend
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.recursive = recursive

        self.blip_model_id = blip_model_id
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.timeout_s = timeout_s

        self._device = None
        self._processor = None
        self._blip_model = None

        if backend == "blip":
            self._init_blip(device)

    # -------------------------
    # Public API
    # -------------------------
    def caption_directory(
        self,
        directory: str | Path,
        extensions: Iterable[str],
    ) -> list[CaptionResult]:

        directory = Path(directory)
        exts = {self._norm_ext(e) for e in extensions}

        if not directory.exists():
            raise FileNotFoundError(directory)

        results: list[CaptionResult] = []

        iterator = directory.rglob("*") if self.recursive else directory.iterdir()

        for path in iterator:
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue

            try:
                caption = self._caption_one(path)
                if caption:
                    results.append(
                        CaptionResult(
                            image_path=path,
                            backend=self.backend,
                            caption=caption,
                        )
                    )
            except Exception as e:
                print(f"[ERR] {path} → {e}")

        return results

    # -------------------------
    # Internals
    # -------------------------
    def _caption_one(self, image_path: Path) -> str:
        if self.backend == "blip":
            return self._caption_blip(image_path)
        return self._caption_ollama(image_path)

    @staticmethod
    def _norm_ext(ext: str) -> str:
        ext = ext.lower()
        return ext if ext.startswith(".") else f".{ext}"

    # -------------------------
    # BLIP
    # -------------------------
    def _init_blip(self, device: Optional[str]):
        if torch is None:
            raise RuntimeError("BLIP backend requested but torch/transformers not installed")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)
        self._processor = AutoProcessor.from_pretrained(self.blip_model_id)
        self._blip_model = (
            BlipForConditionalGeneration
            .from_pretrained(self.blip_model_id)
            .to(self._device)
            .eval()
        )

    def _caption_blip(self, image_path: Path) -> str:
        with Image.open(image_path) as im:
            im = im.convert("RGB")

            if im.size[0] < 32 or im.size[1] < 32:
                return ""

            with torch.no_grad():
                inputs = self._processor(
                    images=im,
                    text=self.prompt,
                    return_tensors="pt",
                ).to(self._device)

                out = self._blip_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

                return self._processor.decode(out[0], skip_special_tokens=True)

    # -------------------------
    # Ollama
    # -------------------------
    def _caption_ollama(self, image_path: Path) -> str:
        with image_path.open("rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.ollama_model,
            "prompt": self.prompt or "Describe the image",
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": self.max_new_tokens},
        }

        r = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
