"""The answer contract — a hard requirement: every answer carries citations."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["lookup", "multihop", "global", "refusal"]


class Citation(BaseModel):
    doc_id: str
    page: Optional[int] = None
    row: Optional[int] = None
    bbox: Optional[list[float]] = None
    quote: str = ""                      # <= 15 words, verbatim from source
    chunk_id: Optional[str] = None


class PathEdge(BaseModel):
    source: str
    edge: str
    target: str


class Answer(BaseModel):
    answer_markdown: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    mode: Mode = "lookup"
    path: list[PathEdge] = Field(default_factory=list)
    seeds: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    llm_used: bool = False


REFUSAL_TEXT = ("I couldn't find this in the plant corpus. "
                "Here is the closest related material I have — please refine the "
                "question or check whether the relevant document has been ingested.")
