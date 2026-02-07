from pathlib import Path

from blipollama import VisionCaptionService
from blipollama.models import CaptionBackend


class DummyBackend:
    name = "dummy"

    def caption(self, image_path: Path) -> str:
        # On retourne une caption déterministe basée sur le nom du fichier
        return f"caption:{image_path.name}"


def test_caption_directory_filters_extensions_and_returns_results(tmp_path: Path):
    # Arrange: crée des fichiers "images" + des fichiers à ignorer
    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.png"
    ignore = tmp_path / "c.txt"

    img1.write_bytes(b"fakejpg")
    img2.write_bytes(b"fakepng")
    ignore.write_text("not an image", encoding="utf-8")

    backend: CaptionBackend = DummyBackend()
    svc = VisionCaptionService(backend=backend, recursive=False)

    # Act
    results = svc.caption_directory(tmp_path, extensions=["jpg", "png"])

    # Assert
    assert len(results) == 2
    paths = {r.image_path.name for r in results}
    assert paths == {"a.jpg", "b.png"}

    captions = {r.caption for r in results}
    assert captions == {"caption:a.jpg", "caption:b.png"}

    assert all(r.backend == "dummy" for r in results)

