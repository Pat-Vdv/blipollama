from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .models import CaptionResult, Backend
from .backends.blip_backend import BlipCaptioner
from .backends.ollama_backend import OllamaCaptioner


def _norm_ext(ext: str) -> str:
    ext = ext.lower().strip()
    return ext if ext.startswith(".") else f".{ext}"


class VisionCaptionService:
    """
    Service qui parcourt un répertoire et génère des captions via:
      - BLIP (Transformers)
      - Ollama (llava/moondream/...)
    """

    def __init__(
        self,
        *,
        backend: Backend,
        recursive: bool = False,
        # BLIP
        blip_model_id: str = "Salesforce/blip-image-captioning-base",
        blip_device: Optional[str] = None,
        blip_prompt: str = "a photo of",
        blip_max_new_tokens: int = 50,
        # Ollama
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llava",
        ollama_prompt: str = "Describe the image",
        ollama_max_new_tokens: int = 80,
        ollama_timeout_s: int = 120,
    ) -> None:
        self.backend = backend
        self.recursive = recursive

        self._blip: Optional[BlipCaptioner] = None
        self._ollama: Optional[OllamaCaptioner] = None

        if backend == "blip":
            self._blip = BlipCaptioner(
                model_id=blip_model_id,
                device=blip_device,
                prompt=blip_prompt,
                max_new_tokens=blip_max_new_tokens,
            )
        elif backend == "ollama":
            self._ollama = OllamaCaptioner(
                url=ollama_url,
                model=ollama_model,
                prompt=ollama_prompt,
                max_new_tokens=ollama_max_new_tokens,
                timeout_s=ollama_timeout_s,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def caption_directory(
        self,
        directory: str | Path,
        extensions: Iterable[str],
    ) -> list[CaptionResult]:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(directory)

        exts = {_norm_ext(e) for e in extensions}
        iterator = directory.rglob("*") if self.recursive else directory.iterdir()

        results: list[CaptionResult] = []

        for p in iterator:
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue

            try:
                caption = self._caption_one(p)
                if caption:
                    results.append(CaptionResult(image_path=p, backend=self.backend, caption=caption))
            except Exception as e:
                print(f"[ERR] {p} → {e}")

        return results

    def _caption_one(self, image_path: Path) -> str:
        if self.backend == "blip":
            assert self._blip is not None
            return self._blip.caption(image_path)
        assert self._ollama is not None
        return self._ollama.caption(image_path)

