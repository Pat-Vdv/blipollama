from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Literal


BackendName = Literal["blip", "ollama"]


class CaptionBackend(Protocol):
    name: BackendName

    def caption(self, image_path: Path) -> str:
        ...
        

@dataclass
class CaptionResult:
    image_path: Path
    backend: BackendName
    caption: str

