from pathlib import Path
from blipollama import VisionCaptionService

class DummyBackend:
    name = "dummy"
    def caption(self, image_path: Path) -> str:
        return image_path.name

def test_caption_directory_recursive(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()

    (tmp_path / "a.jpg").write_bytes(b"x")
    (sub / "b.jpg").write_bytes(b"y")
    (sub / "c.txt").write_text("nope", encoding="utf-8")

    svc = VisionCaptionService(backend=DummyBackend(), recursive=True)
    results = svc.caption_directory(tmp_path, extensions=["jpg"])

    names = sorted(r.image_path.name for r in results)
    assert names == ["a.jpg", "b.jpg"]

