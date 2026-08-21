"""Shared public result contract for independent paragraph retrievers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class ParagraphSearchResult:
    """One ranked paragraph hit returned by dense or lexical retrieval."""

    paragraph_uid: str
    text: str
    case_id: int
    title: str
    case_number: str | None
    court: str | None
    judgment_date: date | None
    source_url: str | None
    paragraph_number: int
    page_number: int | None
    score: float
    rank: int
