"""Stable identifiers for corpus entities."""

from __future__ import annotations

import re
from uuid import UUID, uuid5


# This namespace is part of the persisted identity contract. Changing it would
# change every paragraph UID and invalidate durable citations and Qdrant points.
PARAGRAPH_UID_NAMESPACE = UUID("e61a47e8-5f0b-5df5-a8d6-84cb84760e42")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _validated_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA256 hex digest")
    return normalized


def generate_paragraph_uid(
    case_document_hash: str,
    paragraph_number: int,
    paragraph_text_hash: str,
) -> str:
    """Return the deterministic UUIDv5 identity for a legal paragraph.

    The UUID name is an unambiguous serialization of the case document hash,
    paragraph number, and normalized paragraph text hash. The canonical UUID
    string is accepted directly by Qdrant as a point ID.
    """

    document_hash = _validated_sha256(
        case_document_hash, field_name="case_document_hash"
    )
    text_hash = _validated_sha256(
        paragraph_text_hash, field_name="paragraph_text_hash"
    )
    if isinstance(paragraph_number, bool) or not isinstance(paragraph_number, int):
        raise TypeError("paragraph_number must be an integer")
    if paragraph_number <= 0:
        raise ValueError("paragraph_number must be greater than zero")

    uuid_name = f"{document_hash}:{paragraph_number}:{text_hash}"
    return str(uuid5(PARAGRAPH_UID_NAMESPACE, uuid_name))
