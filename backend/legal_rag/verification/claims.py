"""Deterministic material-claim segmentation for grounded answer text."""

from __future__ import annotations

import re

from legal_rag.generation.evidence import cited_evidence_ids
from legal_rag.generation.service import NO_EVIDENCE_ANSWER
from legal_rag.verification.models import ExtractedClaim


_CITATION_PATTERN = re.compile(r"\[(?:E[1-9]\d*)\]")
_LIST_PREFIX_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s+|[-*\u2022]\s+|\d+[.)]\s+)",
)
_CONNECTIVE_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"based on (?:the )?(?:retrieved|supplied|cited) evidence|"
    r"according to (?:the )?(?:retrieved|supplied|cited) evidence|"
    r"in summary|in conclusion|overall|therefore|however|"
    r"additionally|moreover"
    r")\s*(?:[:,;-]\s*|\s+)",
    re.IGNORECASE,
)
_NON_MATERIAL_PATTERNS = (
    re.compile(
        r"^(?:answer|analysis|conclusion|holding|"
        r"legal position|result|summary)[.:]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:based on|according to) (?:the )?"
        r"(?:retrieved|supplied|cited) evidence[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:this (?:answer|response) is|the following is) "
        r"not legal advice[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:consult|please consult) (?:a |an )?"
        r"(?:lawyer|advocate|legal professional).*$",
        re.IGNORECASE,
    ),
)
_ABBREVIATIONS = {
    "art.",
    "co.",
    "dr.",
    "inc.",
    "ltd.",
    "mr.",
    "mrs.",
    "no.",
    "nos.",
    "pvt.",
    "s.",
    "sec.",
    "v.",
    "vs.",
}


def extract_material_claims(answer: str) -> list[ExtractedClaim]:
    """Return ordered C1.. claims without carrying citations between claims."""

    claim_values: list[tuple[str, list[str]]] = []
    normalized = answer.replace("\r\n", "\n").replace("\r", "\n")

    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line or _is_heading(line):
            continue
        line = _LIST_PREFIX_PATTERN.sub("", line, count=1).strip()
        for segment in _split_sentences(line):
            citation_ids = cited_evidence_ids(segment)
            claim_text = _CITATION_PATTERN.sub("", segment)
            claim_text = re.sub(r"\s+", " ", claim_text).strip()
            claim_text = re.sub(r"\s+([.,;:!?])", r"\1", claim_text)
            claim_text = _strip_non_material_prefixes(claim_text)
            if _is_material_claim(claim_text):
                claim_values.append((claim_text, citation_ids))

    return [
        ExtractedClaim(
            claim_id=f"C{index}",
            text=text,
            citation_ids=citation_ids,
        )
        for index, (text, citation_ids) in enumerate(claim_values, start=1)
    ]


def _split_sentences(line: str) -> list[str]:
    """Split prose while keeping citations after punctuation with that claim."""

    segments: list[str] = []
    start = 0
    index = 0
    while index < len(line):
        if line[index] not in ".!?" or _is_nonterminal_abbreviation(line, index):
            index += 1
            continue

        cursor = index + 1
        while cursor < len(line) and line[cursor] in " \t":
            cursor += 1
        while True:
            citation = _CITATION_PATTERN.match(line, cursor)
            if citation is None:
                break
            cursor = citation.end()
            while cursor < len(line) and line[cursor] in " \t":
                cursor += 1

        if cursor >= len(line) or _starts_new_sentence(line[cursor]):
            segment = line[start:cursor].strip()
            if segment:
                segments.append(segment)
            start = cursor
            index = cursor
            continue
        index += 1

    tail = line[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def _is_nonterminal_abbreviation(line: str, punctuation_index: int) -> bool:
    if line[punctuation_index] != ".":
        return False
    prefix = line[: punctuation_index + 1]
    match = re.search(r"([A-Za-z]+\.)$", prefix)
    if match is None:
        return False
    token = match.group(1).lower()
    if token not in _ABBREVIATIONS and len(token) != 2:
        return False

    cursor = punctuation_index + 1
    while cursor < len(line) and line[cursor] in " \t":
        cursor += 1
    citation = _CITATION_PATTERN.match(line, cursor)
    if citation is not None:
        return False
    return True


def _starts_new_sentence(character: str) -> bool:
    return (
        character.isupper()
        or character.isdigit()
        or character in "\"'([{#-*\u2022"
    )


def _strip_non_material_prefixes(text: str) -> str:
    stripped = text
    while True:
        updated = _CONNECTIVE_PREFIX_PATTERN.sub("", stripped, count=1).strip()
        if updated == stripped:
            return stripped
        stripped = updated


def _is_heading(line: str) -> bool:
    without_citations = _CITATION_PATTERN.sub("", line).strip()
    if line.lstrip().startswith("#"):
        return True
    words = re.findall(r"[A-Za-z0-9]+", without_citations)
    if not words:
        return True
    if without_citations.endswith(":") and len(words) <= 12:
        return True
    return (
        len(words) <= 12
        and not without_citations.endswith((".", "!", "?"))
        and without_citations.upper() == without_citations
        and any(character.isalpha() for character in without_citations)
    )


def _is_material_claim(text: str) -> bool:
    stripped = text.strip(" \t-:;")
    if not stripped:
        return False
    if stripped == NO_EVIDENCE_ANSWER:
        return False
    if any(pattern.fullmatch(stripped) for pattern in _NON_MATERIAL_PATTERNS):
        return False
    words = re.findall(r"[A-Za-z0-9]+", stripped)
    return len(words) >= 2
