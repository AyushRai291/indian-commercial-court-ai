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


@dataclass(frozen=True, slots=True)
class HybridSearchResult(ParagraphSearchResult):
    """A fused paragraph hit with the native ranks and scores that produced it."""

    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None

    @property
    def rrf_score(self) -> float:
        """Expose the inherited score explicitly as the final RRF score."""

        return self.score
