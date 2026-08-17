from __future__ import annotations

from legal_rag.corpus import extract_paragraphs, sha256_text


def test_extract_paragraphs_preserves_explicit_numbers_and_tracks_page_breaks() -> None:
    text = (
        "1. The plaintiff instituted the commercial suit.\n"
        "This sentence continues paragraph one.\n\n"
        "[7] The defendant contested jurisdiction.\f\n"
        "(12) The court framed the issues.\n\n"
        "13) The suit was decreed."
    )

    paragraphs = extract_paragraphs(text)

    assert [paragraph.paragraph_number for paragraph in paragraphs] == [1, 7, 12, 13]
    assert [paragraph.page_number for paragraph in paragraphs] == [1, 1, 2, 2]
    assert paragraphs[0].text == (
        "The plaintiff instituted the commercial suit. "
        "This sentence continues paragraph one."
    )
    assert paragraphs[2].text == "The court framed the issues."
    assert all(paragraph.text_hash == sha256_text(paragraph.text) for paragraph in paragraphs)


def test_extract_paragraphs_assigns_sequential_numbers_to_unnumbered_blocks() -> None:
    paragraphs = extract_paragraphs(
        "The first unnumbered paragraph.\n\nThe second unnumbered paragraph."
    )

    assert [paragraph.paragraph_number for paragraph in paragraphs] == [1, 2]
    assert [paragraph.page_number for paragraph in paragraphs] == [None, None]
    assert [paragraph.text for paragraph in paragraphs] == [
        "The first unnumbered paragraph.",
        "The second unnumbered paragraph.",
    ]


def test_decimal_and_date_prefixes_are_not_treated_as_paragraph_markers() -> None:
    paragraphs = extract_paragraphs(
        "Preamble dated 2024.05.10.\n\n"
        "2.1 Interest is payable under the contract.\n\n"
        "1. The court records its first numbered finding."
    )

    assert [paragraph.text for paragraph in paragraphs] == [
        "Preamble dated 2024.05.10.",
        "2.1 Interest is payable under the contract.",
        "The court records its first numbered finding.",
    ]
    assert len({paragraph.paragraph_number for paragraph in paragraphs}) == 3
    assert paragraphs[-1].paragraph_number == 1
