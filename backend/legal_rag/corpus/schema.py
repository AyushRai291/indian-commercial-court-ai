"""Source-independent corpus records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from legal_rag.corpus.text import document_hash, normalize_text

_INLINE_WHITESPACE_RE = re.compile(r"\s+")


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _INLINE_WHITESPACE_RE.sub(" ", str(value)).strip()
    return cleaned or None


@dataclass(frozen=True, slots=True)
class CanonicalCase:
    """A normalized case record independent of its source dataset."""

    title: str
    raw_text: str
    case_number: str | None = None
    court: str | None = None
    judgment_date: date | None = None
    source: str | None = None
    source_url: str | None = None
    document_hash: str = ""

    def __post_init__(self) -> None:
        title = _clean_optional(self.title)
        if not title:
            raise ValueError("title must not be empty")

        normalized_raw_text = normalize_text(self.raw_text)
        if not normalized_raw_text:
            raise ValueError("raw_text must not be empty")

        if self.judgment_date is not None and not isinstance(
            self.judgment_date, date
        ):
            raise TypeError("judgment_date must be a date or None")

        computed_hash = document_hash(normalized_raw_text)
        supplied_hash = self.document_hash.strip().lower()
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError("document_hash does not match raw_text")

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "raw_text", normalized_raw_text)
        object.__setattr__(self, "case_number", _clean_optional(self.case_number))
        object.__setattr__(self, "court", _clean_optional(self.court))
        object.__setattr__(self, "source", _clean_optional(self.source))
        object.__setattr__(self, "source_url", _clean_optional(self.source_url))
        object.__setattr__(self, "document_hash", computed_hash)

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""

        return {
            "title": self.title,
            "case_number": self.case_number,
            "court": self.court,
            "judgment_date": (
                self.judgment_date.isoformat() if self.judgment_date else None
            ),
            "source": self.source,
            "source_url": self.source_url,
            "raw_text": self.raw_text,
            "document_hash": self.document_hash,
        }
