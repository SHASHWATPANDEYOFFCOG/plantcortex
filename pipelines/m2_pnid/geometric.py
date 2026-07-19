"""Deterministic (offline) P&ID structural reader.

Reads the drawing's PIXELS to infer connectivity — the geometric half of the hybrid,
analogous to the Relationformer edge task. Symbol/tag identities come from the drawing's
text/vector layer sidecar (real vector P&IDs expose extractable text; scanned ones need
the VLM). Requires no network and no OCR binary.

Connection inference: for each pair of symbols, walk the straight segment between their
centres, skip the parts inside any symbol, and measure the fraction of the *gap* that
sits on a drawn (dark) line. Solid process lines score high; the dashed instrument
signal (PT-108) scores low and is correctly rejected — matching the P&ID's true topology.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _binary(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    # drawing is dark-on-white -> 1 where ink
    return (img < 128).astype(np.uint8)


def _in_any_symbol(x: float, y: float, symbols: list[dict], margin: float = 6.0) -> bool:
    for s in symbols:
        x0, y0, x1, y1 = s["bbox"]
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
            return True
    return False


def _passes_through_other(a: dict, b: dict, symbols: list[dict]) -> bool:
    """True if a third symbol's centre lies on the a-b segment (collinear-through)."""
    (ax, ay), (bx, by) = a["center"], b["center"]
    for s in symbols:
        if s is a or s is b:
            continue
        cx, cy = s["center"]
        # distance from point to segment
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            continue
        t = ((cx - ax) * dx + (cy - ay) * dy) / seg2
        if 0.05 < t < 0.95:
            px, py = ax + t * dx, ay + t * dy
            if (px - cx) ** 2 + (py - cy) ** 2 <= 30 ** 2:
                return True
    return False


def _line_fraction(ink: np.ndarray, a: dict, b: dict, symbols: list[dict],
                   samples: int = 160, nbhd: int = 3) -> float:
    (ax, ay), (bx, by) = a["center"], b["center"]
    h, w = ink.shape
    hits = gap = 0
    for i in range(samples + 1):
        t = i / samples
        x, y = ax + t * (bx - ax), ay + t * (by - ay)
        if _in_any_symbol(x, y, [a, b]):
            continue
        gap += 1
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - nbhd), min(w, xi + nbhd + 1)
        y0, y1 = max(0, yi - nbhd), min(h, yi + nbhd + 1)
        if ink[y0:y1, x0:x1].any():
            hits += 1
    return hits / gap if gap else 0.0


def detect_connections(image_path: Path, symbols: list[dict],
                       threshold: float = 0.9) -> list[dict]:
    # Solid process lines score ~1.0; dashed instrument signals score lower and are
    # rejected -> we recover process CONNECTED_TO topology, not signal lines.
    ink = _binary(image_path)
    conns = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = symbols[i], symbols[j]
            if _passes_through_other(a, b, symbols):
                continue
            frac = _line_fraction(ink, a, b, symbols)
            if frac >= threshold:
                conns.append({"from": a["tag"], "to": b["tag"],
                              "score": round(frac, 3)})
    return conns


def detect_symbols_cv(image_path: Path) -> dict:
    """Best-effort CV symbol detection, for a recall metric vs. the vector layer."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=120, param2=30, minRadius=15, maxRadius=90)
    n_circles = 0 if circles is None else len(circles[0])
    inv = (img < 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if 2500 < area < 30000 and 0.4 < w / max(h, 1) < 2.5:
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                rects += 1
    return {"circles": int(n_circles), "rectangles": int(rects)}


def load_layer(layer_path: Path) -> dict:
    return json.loads(Path(layer_path).read_text(encoding="utf-8"))


def geometric_read(image_path: Path, layer_path: Path) -> dict:
    """Structural read: symbols/tags from the vector layer, connections from pixels."""
    layer = load_layer(layer_path)
    symbols = layer["symbols"]
    connections = detect_connections(Path(image_path), symbols)
    cv_stats = detect_symbols_cv(Path(image_path))
    return {"symbols": symbols, "tags": layer.get("tags", []),
            "connections": [[c["from"], c["to"]] for c in connections],
            "connection_detail": connections,
            "cv_detection": {"detected": cv_stats,
                             "vector_symbols": len(symbols)},
            "source": "geometric"}
