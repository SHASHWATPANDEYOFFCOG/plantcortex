"""Route documents to parsers. Text -> pages; work orders -> structured rows.

Scanned images (inspections, P&ID) have no text layer here — their OCR / vision path
is M2 (D3). This module returns what a layout-aware parser would; Docling/unstructured
can be swapped in behind the same return shapes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pypdf import PdfReader


def read_pdf_pages(path: Path) -> list[str]:
    """Return text per page. Empty strings for pages with no text layer (scanned)."""
    try:
        reader = PdfReader(str(path))
    except Exception:
        return []
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def read_workorder_rows(path: Path) -> list[dict]:
    df = pd.read_excel(path, engine="openpyxl")
    rows = []
    for i, r in df.iterrows():
        rows.append({"row": int(i), **{k: (None if pd.isna(v) else v)
                                       for k, v in r.items()}})
    return rows


def has_text(pages: list[str]) -> bool:
    return any(p.strip() for p in pages)
