from __future__ import annotations

from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from .models import CaptionBackend, CaptionResult


def _norm_ext(ext: str) -> str:
    ext = ext.lower().strip()
    return ext if ext.startswith(".") else f".{ext}"


class VisionCaptionService:
    """
    Service ORCHESTRATEUR.
    Il ne connaît ni BLIP, ni Ollama, ni les modèles.
    """

    def __init__(self, backend: CaptionBackend, *, recursive: bool = False) -> None:
        self.backend = backend
        self.recursive = recursive

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

        files = [
            p for p in iterator
            if p.is_file() and p.suffix.lower() in exts
        ]

        results: list[CaptionResult] = []

        for p in tqdm(
            files,
            desc=f"Captioning ({self.backend.name})",
            unit="img",
        ):
            try:
                caption = self.backend.caption(p)
                if caption:
                    results.append(
                        CaptionResult(
                            image_path=p,
                            backend=self.backend.name,
                            caption=caption,
                        )
                    )
            except Exception as e:
                tqdm.write(f"[ERR] {p} → {e}")

        return results


