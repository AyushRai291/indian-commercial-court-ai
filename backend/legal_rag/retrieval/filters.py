"""Shared exact-match metadata constraints for paragraph retrieval."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import date


COURT_FILTER_FIELD = "court_filter"
CASE_NUMBER_FILTER_FIELD = "case_number_filter"
JUDGMENT_YEAR_FILTER_FIELD = "year"


def normalize_metadata_value(value: str) -> str:
    """Normalize one exact-match metadata value without fuzzy interpretation."""

    if not isinstance(value, str):
        raise TypeError("metadata filter values must be strings")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("metadata filter values must not be empty")
    return normalized


def normalized_payload_value(value: str | None) -> str | None:
    """Return a normalized payload value while preserving missing metadata."""

    if value is None or not value.strip():
        return None
    return normalize_metadata_value(value)


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """Optional AND-combined exact metadata filters shared by all retrievers."""

    court: str | None = None
    year: int | None = None
    case_number: str | None = None

    def __post_init__(self) -> None:
        if self.court is not None:
            object.__setattr__(self, "court", normalize_metadata_value(self.court))
        if self.case_number is not None:
            object.__setattr__(
                self,
                "case_number",
                normalize_metadata_value(self.case_number),
            )
        if self.year is not None:
            if isinstance(self.year, bool) or not isinstance(self.year, int):
                raise TypeError("year must be an integer")
            if not 1 <= self.year <= 9999:
                raise ValueError("year must be between 1 and 9999")

    @property
    def is_active(self) -> bool:
        """Return whether at least one metadata constraint is present."""

        return (
            self.court is not None
            or self.year is not None
            or self.case_number is not None
        )

    def matches(
        self,
        *,
        court: str | None,
        judgment_date: date | None,
        case_number: str | None,
    ) -> bool:
        """Apply all active constraints to canonical display metadata."""

        if self.court is not None:
            if court is None or normalized_payload_value(court) != self.court:
                return False
        if self.year is not None:
            if judgment_date is None or judgment_date.year != self.year:
                return False
        if self.case_number is not None:
            if (
                case_number is None
                or normalized_payload_value(case_number) != self.case_number
            ):
                return False
        return True


def build_retrieval_filters(
    *,
    court: str | None,
    year: int | None,
    case_number: str | None,
) -> RetrievalFilters | None:
    """Build the shared contract only when a caller supplied a constraint."""

    if court is None and year is None and case_number is None:
        return None
    return RetrievalFilters(court=court, year=year, case_number=case_number)
