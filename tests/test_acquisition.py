from __future__ import annotations

import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter

from legal_rag.acquisition import (
    JudgmentDownloader,
    MissingMetadataError,
    load_manifest,
    validate_source_record,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _text_pdf(text: str) -> bytes:
    """Build a tiny text-native PDF without an external fixture dependency."""

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 36 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
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


def _blank_pdf() -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    return output.getvalue()


def _record(name: str, *, title: str | None = None) -> dict[str, object]:
    return {
        "title": title if title is not None else f"{name} Ltd. v. Example Ltd.",
        "case_number": f"Civil Appeal No. {name}",
        "court": "Supreme Court of India",
        "judgment_date": "2024-01-02",
        "source_name": "Supreme Court of India",
        "source_page_url": "https://www.sci.gov.in/judgements-case-no/",
        "direct_pdf_url": f"https://api.sci.gov.in/judgments/{name}.pdf",
        "local_filename": f"{name}.pdf",
        "document_type": "judgment",
        "retrieval_timestamp": None,
        "sha256": None,
        "file_size": None,
        "mime_type": None,
    }


def test_downloader_retries_then_resumes_without_another_request(
    tmp_path: Path,
) -> None:
    pdf = _text_pdf("Commercial arbitration judgment text " * 20)
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            content=pdf,
            request=request,
        )

    manifest = tmp_path / "manifest.jsonl"
    output_dir = tmp_path / "raw"
    failure_log = tmp_path / "failed.jsonl"
    audit_path = tmp_path / "audit.json"
    write_manifest(manifest, [_record("101")])
    sleeps: list[float] = []

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        downloader = JudgmentDownloader(
            client,
            retries=3,
            sleep=sleeps.append,
        )
        first_audit = downloader.run(
            manifest_path=manifest,
            output_dir=output_dir,
            failure_log=failure_log,
            audit_path=audit_path,
        )
        second_audit = downloader.run(
            manifest_path=manifest,
            output_dir=output_dir,
            failure_log=failure_log,
            audit_path=audit_path,
        )

    stored = load_manifest(manifest)[0]
    assert requests == 2
    assert sleeps == [1.0]
    assert len(first_audit.downloaded) == 1
    assert len(second_audit.downloaded) == 1
    assert stored["sha256"]
    assert stored["file_size"] == len(pdf)
    assert stored["mime_type"] == "application/pdf"
    assert stored["retrieval_timestamp"]
    assert (output_dir / "101.pdf").read_bytes() == pdf
    assert not failure_log.exists()


def test_audit_categorizes_duplicate_invalid_scanned_failed_and_missing(
    tmp_path: Path,
) -> None:
    valid_pdf = _text_pdf("Insolvency and commercial contract judgment " * 20)
    responses = {
        "/judgments/valid.pdf": (200, valid_pdf),
        "/judgments/duplicate.pdf": (200, valid_pdf),
        "/judgments/invalid.pdf": (200, b"<html>not a PDF</html>"),
        "/judgments/scanned.pdf": (200, _blank_pdf()),
        "/judgments/missing-url.pdf": (404, b"not found"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        status, content = responses[request.url.path]
        return httpx.Response(status, content=content, request=request)

    records = [
        _record("valid"),
        _record("duplicate"),
        _record("invalid"),
        _record("scanned"),
        _record("missing-url"),
        _record("metadata", title=""),
    ]
    manifest = tmp_path / "manifest.jsonl"
    output_dir = tmp_path / "raw"
    failure_log = tmp_path / "failed.jsonl"
    audit_path = tmp_path / "audit.json"
    write_manifest(manifest, records)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        audit = JudgmentDownloader(
            client,
            retries=2,
            sleep=lambda _seconds: None,
        ).run(
            manifest_path=manifest,
            output_dir=output_dir,
            failure_log=failure_log,
            audit_path=audit_path,
        )

    assert len(audit.downloaded) == 1
    assert len(audit.duplicates) == 1
    assert len(audit.invalid_pdfs) == 1
    assert len(audit.scanned_pdfs) == 1
    assert len(audit.failed_urls) == 1
    assert len(audit.missing_metadata) == 1
    assert sorted(path.name for path in output_dir.iterdir()) == ["valid.pdf"]
    assert len(failure_log.read_text(encoding="utf-8").splitlines()) == 5

    persisted_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted_audit["counts"] == {
        "downloaded": 1,
        "duplicates": 1,
        "invalid_pdfs": 1,
        "scanned_pdfs": 1,
        "failed_urls": 1,
        "missing_metadata": 1,
    }


def test_tracked_pilot_manifest_has_100_complete_unique_records() -> None:
    records = load_manifest(
        PROJECT_ROOT / "data" / "manifests" / "judgments_pilot.jsonl"
    )

    assert len(records) == 100
    for record in records:
        validate_source_record(record)
        assert datetime.fromisoformat(str(record["retrieval_timestamp"]))
        assert re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
        assert isinstance(record["file_size"], int)
        assert record["file_size"] > 0
        assert record["mime_type"] == "application/pdf"

    assert {record["document_type"] for record in records} == {"judgment"}

    for field in ("direct_pdf_url", "local_filename", "sha256"):
        values = [record[field] for record in records]
        assert len(set(values)) == len(values)


def test_order_document_type_is_rejected() -> None:
    record = _record("order")
    record["document_type"] = "order"
    with pytest.raises(MissingMetadataError, match="document_type"):
        validate_source_record(record)


def test_allowlisted_supreme_court_archive_download_is_accepted() -> None:
    record = _record("archive")
    record["source_page_url"] = "https://scr.sci.gov.in/scrsearch/"
    record["direct_pdf_url"] = (
        "https://indian-supreme-court-judgments.s3.amazonaws.com/"
        "data/pdf/year=2021/english/example_EN.pdf"
    )
    validate_source_record(record)


def test_untrusted_download_archive_is_rejected() -> None:
    record = _record("archive")
    record["direct_pdf_url"] = "https://example.com/judgment.pdf"
    with pytest.raises(MissingMetadataError, match="direct_pdf_url"):
        validate_source_record(record)
