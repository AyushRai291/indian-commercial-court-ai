from __future__ import annotations

import hashlib

from legal_rag.corpus import (
    deduplicate_paragraphs,
    document_hash,
    extract_paragraphs,
    sha256_text,
)


def test_sha256_text_matches_standard_sha256() -> None:
    expected = hashlib.sha256(b"abc").hexdigest()

    assert sha256_text("abc") == expected


def test_document_hash_uses_normalized_document_text() -> None:
    first = "  A   commercial dispute.\r\n\r\nThe decree follows.  "
    equivalent = "A commercial dispute.\n\nThe decree follows."

    assert document_hash(first) == document_hash(equivalent)
    assert document_hash(first) != document_hash("A different judgment.")


def test_document_hash_ignores_source_line_wrapping_and_page_breaks() -> None:
    html_export = "The commercial claim is allowed. Costs follow."
    paginated_export = "The commercial\nclaim is allowed.\fCosts follow."

    assert document_hash(html_export) == document_hash(paginated_export)


def test_paragraph_deduplication_uses_normalized_text_hash_and_keeps_first() -> None:
    first = extract_paragraphs("1. The contract was valid.")[0]
    duplicate = extract_paragraphs("[8]  The   contract was valid.  ")[0]
    distinct = extract_paragraphs("9) The claim was within limitation.")[0]

    assert first.text_hash == duplicate.text_hash

    unique = deduplicate_paragraphs([first, duplicate, distinct])

    assert unique == [first, distinct]
    assert unique[0].paragraph_number == 1
