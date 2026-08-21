"""Deterministic BM25 paragraph retrieval over canonical PostgreSQL data."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from legal_rag.models import Case, Paragraph
from legal_rag.retrieval.results import ParagraphSearchResult


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize_legal_text(text: str) -> list[str]:
    """Apply NFKC, case-folding, and Unicode alphanumeric tokenization."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


@dataclass(frozen=True, slots=True)
class ParagraphDocument:
    """Canonical paragraph and case metadata used to construct a BM25 index."""

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


class BM25ParagraphRetriever:
    """In-memory Okapi BM25 index derived deterministically from PostgreSQL."""

    def __init__(
        self,
        documents: Iterable[ParagraphDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        ordered = sorted(
            documents,
            key=lambda item: (
                item.paragraph_uid,
                item.case_id,
                item.paragraph_number,
                item.text,
            ),
        )
        unique: dict[str, ParagraphDocument] = {}
        for document in ordered:
            if not document.paragraph_uid.strip():
                raise ValueError("paragraph_uid must not be empty")
            unique.setdefault(document.paragraph_uid, document)

        self.documents = tuple(unique.values())
        self.k1 = float(k1)
        self.b = float(b)
        self._document_lengths: list[int] = []
        postings: dict[str, dict[int, int]] = defaultdict(dict)

        for index, document in enumerate(self.documents):
            frequencies = Counter(tokenize_legal_text(document.text))
            self._document_lengths.append(sum(frequencies.values()))
            for token, frequency in frequencies.items():
                postings[token][index] = frequency

        self._postings = dict(postings)
        self.average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_count = len(self.documents)
        self._idf = {
            token: math.log(
                1.0
                + (document_count - len(token_postings) + 0.5)
                / (len(token_postings) + 0.5)
            )
            for token, token_postings in self._postings.items()
        }

    @classmethod
    def from_session(
        cls,
        session: Session,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25ParagraphRetriever":
        """Load the authoritative paragraph corpus and associated case metadata."""

        statement = (
            select(
                Paragraph.paragraph_uid,
                Paragraph.text,
                Paragraph.paragraph_number,
                Paragraph.page_number,
                Case.id.label("case_id"),
                Case.title,
                Case.case_number,
                Case.court,
                Case.judgment_date,
                Case.source_url,
            )
            .join(Case, Paragraph.case_id == Case.id)
            .order_by(Paragraph.paragraph_uid)
        )
        documents = (
            ParagraphDocument(
                paragraph_uid=row.paragraph_uid,
                text=row.text,
                case_id=row.case_id,
                title=row.title,
                case_number=row.case_number,
                court=row.court,
                judgment_date=row.judgment_date,
                source_url=row.source_url,
                paragraph_number=row.paragraph_number,
                page_number=row.page_number,
            )
            for row in session.execute(statement)
        )
        return cls(documents, k1=k1, b=b)

    @property
    def indexed_paragraphs(self) -> int:
        return len(self.documents)

    def search(self, query: str, *, top_k: int) -> list[ParagraphSearchResult]:
        """Return positive-scoring paragraphs ordered by BM25 then durable UID."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_frequencies = Counter(tokenize_legal_text(query))
        if not query_frequencies:
            raise ValueError("query must contain at least one lexical token")
        if not self.documents or self.average_document_length <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for token, query_frequency in query_frequencies.items():
            idf = self._idf.get(token)
            if idf is None:
                continue
            for document_index, term_frequency in self._postings[token].items():
                document_length = self._document_lengths[document_index]
                length_normalization = 1.0 - self.b + self.b * (
                    document_length / self.average_document_length
                )
                numerator = term_frequency * (self.k1 + 1.0)
                denominator = term_frequency + self.k1 * length_normalization
                scores[document_index] += (
                    query_frequency * idf * numerator / denominator
                )

        ranked = sorted(
            (
                (score, self.documents[document_index])
                for document_index, score in scores.items()
                if score > 0
            ),
            key=lambda item: (-item[0], item[1].paragraph_uid),
        )[:top_k]

        return [
            ParagraphSearchResult(
                paragraph_uid=document.paragraph_uid,
                text=document.text,
                case_id=document.case_id,
                title=document.title,
                case_number=document.case_number,
                court=document.court,
                judgment_date=document.judgment_date,
                source_url=document.source_url,
                paragraph_number=document.paragraph_number,
                page_number=document.page_number,
                score=float(score),
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked, start=1)
        ]
