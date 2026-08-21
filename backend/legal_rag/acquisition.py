"""Resumable acquisition and validation of official judgment PDFs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader


SOURCE_FIELDS = (
    "title",
    "case_number",
    "court",
    "judgment_date",
    "source_name",
    "source_page_url",
    "direct_pdf_url",
    "local_filename",
    "document_type",
)
RESULT_FIELDS = (
    "retrieval_timestamp",
    "sha256",
    "file_size",
    "mime_type",
)
MANIFEST_FIELDS = SOURCE_FIELDS + RESULT_FIELDS

_OFFICIAL_HOST_SUFFIXES = (".gov.in", ".nic.in")
_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class AcquisitionError(RuntimeError):
    """Base class for one-record acquisition failures."""


class MissingMetadataError(AcquisitionError):
    """Raised when provenance or required metadata is incomplete."""


class InvalidPdfError(AcquisitionError):
    """Raised when downloaded bytes are not a readable PDF."""


class ScannedPdfError(AcquisitionError):
    """Raised when a PDF has no meaningful extractable text layer."""


class DuplicatePdfError(AcquisitionError):
    """Raised when two manifest records resolve to identical PDF bytes."""


@dataclass(frozen=True, slots=True)
class PdfValidation:
    """Validated properties of a downloaded judgment."""

    sha256: str
    file_size: int
    page_count: int
    text_characters: int
    mime_type: str = "application/pdf"


@dataclass(frozen=True, slots=True)
class ExtractedPdf:
    """Validated PDF text with form-feed page boundaries."""

    raw_text: str
    sha256: str
    file_size: int
    page_count: int
    text_characters: int
    mime_type: str = "application/pdf"


@dataclass(slots=True)
class AcquisitionAudit:
    """Categorized results for a complete manifest pass."""

    downloaded: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    invalid_pdfs: list[dict[str, Any]] = field(default_factory=list)
    scanned_pdfs: list[dict[str, Any]] = field(default_factory=list)
    failed_urls: list[dict[str, Any]] = field(default_factory=list)
    missing_metadata: list[dict[str, Any]] = field(default_factory=list)

    def add_failure(
        self,
        category: str,
        *,
        record_number: int,
        record: Mapping[str, Any],
        error: Exception,
    ) -> dict[str, Any]:
        detail = {
            "record_number": record_number,
            "title": record.get("title"),
            "direct_pdf_url": record.get("direct_pdf_url"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        getattr(self, category).append(detail)
        return detail

    def to_dict(self, *, manifest_path: Path) -> dict[str, Any]:
        categories = (
            "downloaded",
            "duplicates",
            "invalid_pdfs",
            "scanned_pdfs",
            "failed_urls",
            "missing_metadata",
        )
        return {
            "generated_at": _utc_now(),
            "manifest": manifest_path.as_posix(),
            "document_types": {
                "judgment": sum(
                    item.get("document_type") == "judgment"
                    for item in self.downloaded
                )
            },
            "counts": {
                category: len(getattr(self, category)) for category in categories
            },
            "details": {
                category: getattr(self, category) for category in categories
            },
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_official_url(value: str) -> bool:
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and bool(hostname)
        and any(
            hostname == suffix[1:] or hostname.endswith(suffix)
            for suffix in _OFFICIAL_HOST_SUFFIXES
        )
    )


def validate_source_record(record: Mapping[str, Any]) -> None:
    """Validate required metadata, filename safety, and official provenance."""

    missing = [field for field in SOURCE_FIELDS if not str(record.get(field) or "").strip()]
    if missing:
        raise MissingMetadataError(f"missing required metadata: {', '.join(missing)}")

    try:
        datetime.strptime(str(record["judgment_date"]), "%Y-%m-%d")
    except ValueError as error:
        raise MissingMetadataError("judgment_date must use YYYY-MM-DD") from error

    filename = str(record["local_filename"])
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise MissingMetadataError("local_filename must be a plain .pdf filename")

    if str(record["document_type"]).strip().lower() != "judgment":
        raise MissingMetadataError("document_type must be 'judgment'")

    for field_name in ("source_page_url", "direct_pdf_url"):
        value = str(record[field_name])
        if not _is_official_url(value):
            raise MissingMetadataError(
                f"{field_name} must be an HTTPS URL on an official .gov.in or .nic.in host"
            )


def extract_pdf_text(
    content: bytes, *, min_text_characters: int = 200
) -> ExtractedPdf:
    """Extract text while requiring a readable, text-native PDF."""

    if min_text_characters <= 0:
        raise ValueError("min_text_characters must be positive")
    if not content.startswith(b"%PDF-"):
        raise InvalidPdfError("response does not begin with a PDF file signature")

    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise InvalidPdfError("PDF is encrypted and cannot be opened")
        if not reader.pages:
            raise InvalidPdfError("PDF contains no pages")
        page_texts = [page.extract_text() or "" for page in reader.pages]
    except InvalidPdfError:
        raise
    except Exception as error:
        raise InvalidPdfError(f"PDF parser could not open the response: {error}") from error

    raw_text = "\f".join(page_texts)
    text_characters = len("".join(raw_text.split()))
    if text_characters < min_text_characters:
        raise ScannedPdfError(
            "PDF text layer is too small: "
            f"{text_characters} characters; required {min_text_characters}"
        )

    return ExtractedPdf(
        raw_text=raw_text,
        sha256=hashlib.sha256(content).hexdigest(),
        file_size=len(content),
        page_count=len(reader.pages),
        text_characters=text_characters,
    )


def validate_document_type(record: Mapping[str, Any], raw_text: str) -> None:
    """Confirm that a declared judgment contains an explicit judgment marker."""

    document_type = str(record.get("document_type") or "").strip().lower()
    if document_type != "judgment":
        raise MissingMetadataError("document_type must be 'judgment'")
    marker = re.compile(r"\bJ\s*U\s*D\s*G\s*(?:E\s*)?M\s*E\s*N\s*T\b", re.IGNORECASE)
    if marker.search(raw_text[:50_000]) is None:
        raise InvalidPdfError("PDF does not contain a verifiable JUDGMENT heading")


def validate_pdf(content: bytes, *, min_text_characters: int = 200) -> PdfValidation:
    """Return validated properties for a readable, text-native PDF."""

    extracted = extract_pdf_text(content, min_text_characters=min_text_characters)
    return PdfValidation(
        sha256=extracted.sha256,
        file_size=extracted.file_size,
        page_count=extracted.page_count,
        text_characters=extracted.text_characters,
        mime_type=extracted.mime_type,
    )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL manifest without accepting malformed or non-object rows."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on manifest line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            records.append({field: value.get(field) for field in MANIFEST_FIELDS})
    return records


def write_manifest(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically persist canonical JSONL records for resumable operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            canonical = {field: record.get(field) for field in MANIFEST_FIELDS}
            handle.write(json.dumps(canonical, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _append_failure(path: Path, detail: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": _utc_now(), **detail}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class JudgmentDownloader:
    """Download official PDFs with retries and manifest-backed resumption."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        retries: int = 3,
        max_bytes: int = 50 * 1024 * 1024,
        min_text_characters: int = 200,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries <= 0:
            raise ValueError("retries must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.client = client
        self.retries = retries
        self.max_bytes = max_bytes
        self.min_text_characters = min_text_characters
        self.sleep = sleep

    def _fetch(self, url: str) -> bytes:
        last_error: Exception | None = None
        attempts_made = 0
        for attempt in range(1, self.retries + 1):
            attempts_made = attempt
            try:
                response = self.client.get(url)
                if not _is_official_url(str(response.url)):
                    raise AcquisitionError(
                        f"download redirected to a non-official host: {response.url}"
                    )
                response.raise_for_status()
                content = response.content
                if len(content) > self.max_bytes:
                    raise InvalidPdfError(
                        f"response is {len(content)} bytes; limit is {self.max_bytes}"
                    )
                return content
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in _RETRYABLE_STATUS_CODES:
                    break
            except httpx.RequestError as error:
                last_error = error
            if attempt < self.retries:
                self.sleep(float(2 ** (attempt - 1)))

        assert last_error is not None
        raise AcquisitionError(
            f"download failed after {attempts_made} attempt(s): {last_error}"
        )

    def run(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        failure_log: Path,
        audit_path: Path,
    ) -> AcquisitionAudit:
        records = load_manifest(manifest_path)
        audit = AcquisitionAudit()
        seen_hashes: dict[str, str] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        for record_number, record in enumerate(records, start=1):
            try:
                validate_source_record(record)
                destination = output_dir / str(record["local_filename"])
                if destination.is_file():
                    content = destination.read_bytes()
                else:
                    content = self._fetch(str(record["direct_pdf_url"]))

                extracted = extract_pdf_text(
                    content,
                    min_text_characters=self.min_text_characters,
                )
                validate_document_type(record, extracted.raw_text)
                validation = PdfValidation(
                    sha256=extracted.sha256,
                    file_size=extracted.file_size,
                    page_count=extracted.page_count,
                    text_characters=extracted.text_characters,
                    mime_type=extracted.mime_type,
                )
                existing_filename = seen_hashes.get(validation.sha256)
                if existing_filename is not None:
                    raise DuplicatePdfError(
                        f"content duplicates {existing_filename} ({validation.sha256})"
                    )

                expected_hash = str(record.get("sha256") or "").strip().lower()
                if expected_hash and expected_hash != validation.sha256:
                    raise InvalidPdfError(
                        "existing manifest hash does not match downloaded bytes"
                    )

                if not destination.is_file():
                    temporary_path = destination.with_suffix(destination.suffix + ".part")
                    with temporary_path.open("wb") as handle:
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary_path, destination)

                record.update(
                    retrieval_timestamp=(
                        record.get("retrieval_timestamp") or _utc_now()
                    ),
                    sha256=validation.sha256,
                    file_size=validation.file_size,
                    mime_type=validation.mime_type,
                )
                seen_hashes[validation.sha256] = str(record["local_filename"])
                audit.downloaded.append(
                    {
                        "record_number": record_number,
                        "local_filename": record["local_filename"],
                        "document_type": record["document_type"],
                        "sha256": validation.sha256,
                        "file_size": validation.file_size,
                        "page_count": validation.page_count,
                        "text_characters": validation.text_characters,
                    }
                )
            except MissingMetadataError as error:
                detail = audit.add_failure(
                    "missing_metadata",
                    record_number=record_number,
                    record=record,
                    error=error,
                )
                _append_failure(failure_log, detail)
            except DuplicatePdfError as error:
                detail = audit.add_failure(
                    "duplicates",
                    record_number=record_number,
                    record=record,
                    error=error,
                )
                _append_failure(failure_log, detail)
            except ScannedPdfError as error:
                detail = audit.add_failure(
                    "scanned_pdfs",
                    record_number=record_number,
                    record=record,
                    error=error,
                )
                _append_failure(failure_log, detail)
            except InvalidPdfError as error:
                detail = audit.add_failure(
                    "invalid_pdfs",
                    record_number=record_number,
                    record=record,
                    error=error,
                )
                _append_failure(failure_log, detail)
            except (AcquisitionError, httpx.HTTPError) as error:
                detail = audit.add_failure(
                    "failed_urls",
                    record_number=record_number,
                    record=record,
                    error=error,
                )
                _append_failure(failure_log, detail)

            write_manifest(manifest_path, records)

        _write_json(audit_path, audit.to_dict(manifest_path=manifest_path))
        return audit
