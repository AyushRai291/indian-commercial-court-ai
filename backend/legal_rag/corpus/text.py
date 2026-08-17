"""Deterministic text cleanup and hashing helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_HORIZONTAL_WHITESPACE_RE = re.compile(r"[^\S\n\f]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_HASH_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return stable corpus text while retaining paragraphs and page breaks.

    Unicode compatibility characters and newline styles are normalized,
    non-printing controls are removed, horizontal whitespace is collapsed, and
    blank-line runs are limited to one blank line. A form-feed is deliberately
    retained because it is the only page-boundary signal in many text exports.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_characters: list[str] = []
    for character in value:
        if character in {"\n", "\f", "\t"}:
            cleaned_characters.append(character)
            continue
        if unicodedata.category(character).startswith("C"):
            continue
        cleaned_characters.append(character)

    value = "".join(cleaned_characters)
    pages: list[str] = []
    for page in value.split("\f"):
        lines = [
            _HORIZONTAL_WHITESPACE_RE.sub(" ", line).strip()
            for line in page.split("\n")
        ]
        normalized_page = "\n".join(lines).strip()
        normalized_page = _EXCESS_BLANK_LINES_RE.sub("\n\n", normalized_page)
        pages.append(normalized_page)

    # Empty pages are meaningful between two non-empty pages, so retain them.
    return "\f".join(pages).strip(" \n\f")


def sha256_text(text: str) -> str:
    """Return the lowercase SHA256 digest of the supplied string."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_hash(text: str) -> str:
    """Hash semantic text while ignoring source-specific layout whitespace.

    Stored text retains paragraphs and form-feed page boundaries, but those
    layout details differ across HTML, PDF, and API exports of the same judgment.
    The document identity therefore collapses every whitespace run before SHA256.
    """

    normalized = normalize_text(text)
    hash_input = _HASH_WHITESPACE_RE.sub(" ", normalized).strip()
    return sha256_text(hash_input)
