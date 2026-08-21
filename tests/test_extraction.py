from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from legal_rag.acquisition import InvalidPdfError, ScannedPdfError, extract_pdf_text
from legal_rag.extraction import CANONICAL_FIELDS, PilotCorpusExtractor


def _two_page_text_pdf() -> bytes:
    streams = [
        b"BT /F1 12 Tf 36 720 Td (JUDGMENT arbitration contract "
        + b"evidence " * 30
        + b") Tj ET",
        b"BT /F1 12 Tf 36 720 Td (Second page insolvency analysis "
        + b"reasons " * 30
        + b") Tj ET",
    ]
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(streams[0])).encode("ascii")
        + b" >>\nstream\n"
        + streams[0]
        + b"\nendstream",
        b"<< /Length "
        + str(len(streams[1])).encode("ascii")
        + b" >>\nstream\n"
        + streams[1]
        + b"\nendstream",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(document)


def _manifest_record(pdf: bytes) -> dict[str, object]:
    return {
        "title": "Example Commercial Ltd. v. Contracting Ltd.",
        "case_number": "Civil Appeal No. 1 of 2024",
        "court": "Supreme Court of India",
        "judgment_date": "2024-01-02",
        "source_name": "Supreme Court of India",
        "source_page_url": "https://www.sci.gov.in/judgements-case-no/",
        "direct_pdf_url": "https://api.sci.gov.in/judgments/example.pdf",
        "local_filename": "example.pdf",
        "document_type": "judgment",
        "retrieval_timestamp": "2024-01-03T00:00:00+00:00",
        "sha256": hashlib.sha256(pdf).hexdigest(),
        "file_size": len(pdf),
        "mime_type": "application/pdf",
    }


def test_extraction_preserves_pages_and_resumes_atomically(tmp_path: Path) -> None:
    pdf = _two_page_text_pdf()
    record = _manifest_record(pdf)
    manifest = tmp_path / "manifest.jsonl"
    raw_dir = tmp_path / "raw"
    output = tmp_path / "processed.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    failure_log = tmp_path / "failures.jsonl"
    audit_path = tmp_path / "audit.json"
    raw_dir.mkdir()
    (raw_dir / "example.pdf").write_bytes(pdf)
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    extractor = PilotCorpusExtractor()
    first = extractor.run(
        manifest_path=manifest,
        raw_dir=raw_dir,
        output_path=output,
        checkpoint_path=checkpoint,
        failure_log=failure_log,
        audit_path=audit_path,
        restart=True,
    )
    (raw_dir / "example.pdf").unlink()
    resumed = extractor.run(
        manifest_path=manifest,
        raw_dir=raw_dir,
        output_path=output,
        checkpoint_path=checkpoint,
        failure_log=failure_log,
        audit_path=audit_path,
    )

    canonical = json.loads(output.read_text(encoding="utf-8"))
    assert set(canonical) == set(CANONICAL_FIELDS)
    assert canonical["source_url"] == record["direct_pdf_url"]
    assert canonical["raw_text"].count("\f") == 1
    assert first.extracted[0]["page_count"] == 2
    assert first.extracted[0]["form_feed_boundaries"] == 1
    assert resumed.extracted[0]["resumed"] is True
    assert not failure_log.exists()


def test_pdf_extraction_rejects_corrupt_scanned_and_encrypted_files() -> None:
    with pytest.raises(InvalidPdfError):
        extract_pdf_text(b"not a PDF")

    blank_output = BytesIO()
    blank_writer = PdfWriter()
    blank_writer.add_blank_page(width=612, height=792)
    blank_writer.write(blank_output)
    with pytest.raises(ScannedPdfError):
        extract_pdf_text(blank_output.getvalue())

    encrypted_output = BytesIO()
    encrypted_writer = PdfWriter()
    encrypted_writer.clone_document_from_reader(PdfReader(BytesIO(_two_page_text_pdf())))
    encrypted_writer.encrypt("secret")
    encrypted_writer.write(encrypted_output)
    with pytest.raises(InvalidPdfError):
        extract_pdf_text(encrypted_output.getvalue())


def test_extractor_audits_pdf_failure_and_missing_metadata(tmp_path: Path) -> None:
    corrupt = b"<html>not a PDF</html>"
    corrupt_record = _manifest_record(corrupt)
    corrupt_record["direct_pdf_url"] = (
        "https://api.sci.gov.in/judgments/corrupt.pdf"
    )
    corrupt_record["local_filename"] = "corrupt.pdf"
    missing_record = dict(corrupt_record)
    missing_record["title"] = ""
    missing_record["direct_pdf_url"] = (
        "https://api.sci.gov.in/judgments/missing.pdf"
    )
    missing_record["local_filename"] = "missing.pdf"

    manifest = tmp_path / "manifest.jsonl"
    raw_dir = tmp_path / "raw"
    output = tmp_path / "processed.jsonl"
    checkpoint = tmp_path / "checkpoint.json"
    failure_log = tmp_path / "failures.jsonl"
    audit_path = tmp_path / "audit.json"
    raw_dir.mkdir()
    (raw_dir / "corrupt.pdf").write_bytes(corrupt)
    manifest.write_text(
        json.dumps(corrupt_record) + "\n" + json.dumps(missing_record) + "\n",
        encoding="utf-8",
    )

    audit = PilotCorpusExtractor().run(
        manifest_path=manifest,
        raw_dir=raw_dir,
        output_path=output,
        checkpoint_path=checkpoint,
        failure_log=failure_log,
        audit_path=audit_path,
        restart=True,
    )

    assert not audit.extracted
    assert len(audit.failures) == 1
    assert len(audit.missing_metadata) == 1
    assert output.read_text(encoding="utf-8") == ""
    assert len(failure_log.read_text(encoding="utf-8").splitlines()) == 2
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted["counts"] == {
        "extracted": 0,
        "failures": 1,
        "missing_metadata": 1,
        "pages": 0,
        "characters": 0,
    }
