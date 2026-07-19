"""VLM (Gemini) P&ID reader — the primary half of the hybrid.

Frontier VLMs locate diagram components well but miss small tags on dense full sheets,
so we tile large drawings into overlapping patches, read each, and stitch. For a clean
demo sheet a single whole-image call is enough. Results are cached by the LLM client,
so once a drawing is read it replays offline (demo-safe).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.config import ROOT

_PROMPT: str | None = None


def _prompt() -> str:
    global _PROMPT
    if _PROMPT is None:
        _PROMPT = (ROOT / "prompts" / "pnid_read.md").read_text(encoding="utf-8")
    return _PROMPT


def _normalize(raw: dict, source: str = "vlm") -> dict:
    symbols = []
    for s in raw.get("symbols", []) or []:
        symbols.append({"tag": s.get("tag"), "cls": s.get("cls"),
                        "bbox": s.get("bbox"),
                        "center": _center(s.get("bbox"))})
    conns = []
    for c in raw.get("connections", []) or []:
        if isinstance(c, (list, tuple)) and len(c) == 2 and all(c):
            conns.append([c[0], c[1]])
    return {"symbols": symbols, "tags": raw.get("tags", []) or [],
            "connections": conns, "source": source}


def _center(bbox):
    if not bbox or len(bbox) != 4:
        return None
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def tile_image(image_path: Path, size: int = 1024, overlap: int = 128
               ) -> list[tuple[bytes, tuple[int, int]]]:
    """Overlapping tiles as (png_bytes, (x_off, y_off)). One tile if the sheet is small."""
    import io

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    if W <= size and H <= size:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return [(buf.getvalue(), (0, 0))]
    step = size - overlap
    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            crop = img.crop((x, y, min(x + size, W), min(y + size, H)))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            tiles.append((buf.getvalue(), (x, y)))
    return tiles


def read_pnid_vlm(llm, image_path: Path) -> dict | None:
    """Return normalized structure, or None if the VLM is unavailable/empty."""
    if llm is None or getattr(llm, "quota_blocked", False):
        return None
    raw = llm.vision_json(_prompt(), Path(image_path).read_bytes(),
                          mime="image/png", max_tokens=4096)
    if not raw or not raw.get("symbols"):
        return None
    return _normalize(raw, source="vlm")
