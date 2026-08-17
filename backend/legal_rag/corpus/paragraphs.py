"""Paragraph extraction and within-document deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from legal_rag.corpus.text import document_hash, normalize_text

_NUMBERED_PARAGRAPH_RE = re.compile(
    r"^\s*(?:"
    r"\[(?P<bracket>\d+)\]"
    r"|\((?P<parenthesized>\d+)\)"
    # A digit immediately after the punctuation means a decimal/version/date,
    # not a paragraph marker (for example ``2.1`` or ``2024.05.10``).
    r"|(?P<punctuated>\d+)[.)](?!\d)"
    r"|(?:para(?:graph)?\.?\s+)(?P<labelled>\d+)[:.)-]?"
    r")\s*(?P<text>.*)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CanonicalParagraph:
    """A paragraph ready for relational storage and vector indexing."""

    paragraph_number: int
    page_number: int | None
    text: str
    text_hash: str = ""

    def __post_init__(self) -> None:
        normalized = normalize_text(self.text)
        if not normalized:
            raise ValueError("paragraph text must not be empty")
        if self.paragraph_number <= 0:
            raise ValueError("paragraph_number must be greater than zero")
        if self.page_number is not None and self.page_number <= 0:
            raise ValueError("page_number must be greater than zero")

        computed_hash = document_hash(normalized)
        supplied_hash = self.text_hash.strip().lower()
        if supplied_hash and supplied_hash != computed_hash:
            raise ValueError("text_hash does not match paragraph text")

        object.__setattr__(self, "text", normalized)
        object.__setattr__(self, "text_hash", computed_hash)


def _explicit_number(match: re.Match[str]) -> int:
    for group in ("bracket", "parenthesized", "punctuated", "labelled"):
        value = match.group(group)
        if value is not None:
            return int(value)
    raise AssertionError("numbered-paragraph regex matched without a number")


def extract_paragraphs(text: str) -> list[CanonicalParagraph]:
    """Split normalized legal text into numbered canonical paragraphs.

    Blank lines and explicit paragraph markers start paragraphs. Consecutive
    physical lines are treated as wrapped text. Form-feeds start a new page;
    without a page-boundary marker the page number is unknown and remains null.
    Explicit markers are removed from paragraph text and retained as the
    paragraph number.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = normalize_text(text)
    if not normalized:
        return []

    pages = normalized.split("\f")
    has_page_boundaries = len(pages) > 1
    extracted: list[CanonicalParagraph] = []
    explicit_numbers = {
        _explicit_number(match)
        for page in pages
        for line in page.split("\n")
        if (match := _NUMBERED_PARAGRAPH_RE.match(line.strip()))
    }
    synthetic_numbers: set[int] = set()
    next_number = 1

    def next_synthetic_number() -> int:
        nonlocal next_number
        while next_number in explicit_numbers or next_number in synthetic_numbers:
            next_number += 1
        number = next_number
        synthetic_numbers.add(number)
        next_number += 1
        return number

    for page_index, page in enumerate(pages, start=1):
        current_lines: list[str] = []
        current_number: int | None = None

        def flush() -> None:
            nonlocal current_lines, current_number, next_number
            paragraph_text = normalize_text(" ".join(current_lines))
            if not paragraph_text:
                current_lines = []
                current_number = None
                return

            number = (
                current_number
                if current_number is not None
                else next_synthetic_number()
            )
            extracted.append(
                CanonicalParagraph(
                    paragraph_number=number,
                    page_number=page_index if has_page_boundaries else None,
                    text=paragraph_text,
                )
            )
            current_lines = []
            current_number = None

        for line in page.split("\n"):
            stripped_line = line.strip()
            if not stripped_line:
                flush()
                continue

            match = _NUMBERED_PARAGRAPH_RE.match(stripped_line)
            if match:
                flush()
                current_number = _explicit_number(match)
                marker_text = match.group("text").strip()
                if marker_text:
                    current_lines.append(marker_text)
                continue

            current_lines.append(stripped_line)

        flush()

    return extracted


def deduplicate_paragraphs(
    paragraphs: Iterable[CanonicalParagraph],
) -> list[CanonicalParagraph]:
    """Retain the first occurrence of each normalized paragraph text."""

    unique: list[CanonicalParagraph] = []
    seen_hashes: set[str] = set()
    for paragraph in paragraphs:
        if not isinstance(paragraph, CanonicalParagraph):
            raise TypeError("paragraphs must contain CanonicalParagraph values")
        if paragraph.text_hash in seen_hashes:
            continue
        seen_hashes.add(paragraph.text_hash)
        unique.append(paragraph)
    return unique
