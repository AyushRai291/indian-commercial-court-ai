from __future__ import annotations

from legal_rag.corpus import CanonicalCase, document_hash, normalize_record, normalize_text


def test_normalize_text_canonicalizes_unicode_whitespace_and_line_endings() -> None:
    raw = "  \uff21\u00a0court\torder  \r\n\r\n  Second   line  "

    assert normalize_text(raw) == "A court order\n\nSecond line"


def test_normalize_record_maps_canonical_fields_and_computes_hash() -> None:
    record = {
        "title": "  Alpha Pvt. Ltd.   v. Beta Ltd. ",
        "case_number": " CS(COMM) 42/2024 ",
        "court": " Delhi High Court ",
        "judgment_date": "2024-05-16",
        "source": " Indian Kanoon ",
        "source_url": " https://indiankanoon.org/doc/42/ ",
        "raw_text": "  1.  The suit is decreed.\r\n",
    }

    case = normalize_record(record)

    assert isinstance(case, CanonicalCase)
    assert case.title == "Alpha Pvt. Ltd. v. Beta Ltd."
    assert case.case_number == "CS(COMM) 42/2024"
    assert case.court == "Delhi High Court"
    assert str(case.judgment_date) == "2024-05-16"
    assert case.source == "Indian Kanoon"
    assert case.source_url == "https://indiankanoon.org/doc/42/"
    assert case.raw_text == "1. The suit is decreed."
    assert case.document_hash == document_hash(case.raw_text)


def test_normalize_record_maps_common_source_aliases() -> None:
    record = {
        "case_title": "Gamma Industries v. Union of India",
        "case_no": "O.M.P.(COMM) 7/2023",
        "court_name": "High Court of Delhi",
        "date": "2023-11-02",
        "url": "https://example.test/judgments/7",
        "text": "1. The petition is allowed.",
    }

    case = normalize_record(record, default_source="example-feed")

    assert case.title == "Gamma Industries v. Union of India"
    assert case.case_number == "O.M.P.(COMM) 7/2023"
    assert case.court == "High Court of Delhi"
    assert str(case.judgment_date) == "2023-11-02"
    assert case.source == "example-feed"
    assert case.source_url == "https://example.test/judgments/7"
    assert case.raw_text == "1. The petition is allowed."


def test_blank_top_level_alias_does_not_mask_nested_source_value() -> None:
    case = normalize_record(
        {
            "title": "   ",
            "raw_text": "\t",
            "metadata": {
                "case_title": "Nested Commercial Case",
                "judgment_text": "1. The nested judgment text is retained.",
            },
        }
    )

    assert case.title == "Nested Commercial Case"
    assert case.raw_text == "1. The nested judgment text is retained."
