"""Resumable conversion of validated judgment PDFs to canonical JSONL."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_rag.acquisition import (
    InvalidPdfError,
    MissingMetadataError,
    ScannedPdfError,
    extract_pdf_text,
    load_manifest,
    validate_document_type,
    validate_source_record,
)


CANONICAL_FIELDS = (
    "title",
    "case_number",
    "court",
    "judgment_date",
    "source",
    "source_url",
    "raw_text",
)


class ExtractionError(RuntimeError):
    """Raised when a source PDF cannot safely become a canonical record."""


@dataclass(slots=True)
class ExtractionAudit:
    """One extraction run's successful and failed records."""

    extracted: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    missing_metadata: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, manifest_path: Path, output_path: Path) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_path.as_posix(),
            "output": output_path.as_posix(),
            "counts": {
                "extracted": len(self.extracted),
                "failures": len(self.failures),
                "missing_metadata": len(self.missing_metadata),
                "pages": sum(int(item["page_count"]) for item in self.extracted),
                "characters": sum(
                    int(item["text_characters"]) for item in self.extracted
                ),
            },
            "details": {
                "extracted": self.extracted,
                "failures": self.failures,
                "missing_metadata": self.missing_metadata,
            },
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            canonical = {field: record[field] for field in CANONICAL_FIELDS}
            handle.write(json.dumps(canonical, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_failure(path: Path, detail: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **detail}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_processed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExtractionError(
                    f"invalid processed JSONL line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict) or any(
                field not in record for field in CANONICAL_FIELDS
            ):
                raise ExtractionError(
                    f"processed JSONL line {line_number} is not canonical"
                )
            source_url = str(record["source_url"])
            if source_url in records:
                raise ExtractionError(f"duplicate processed source_url: {source_url}")
            records[source_url] = record
    return records


def _canonical_record(
    manifest_record: Mapping[str, Any], raw_text: str
) -> dict[str, Any]:
    return {
        "title": str(manifest_record["title"]),
        "case_number": str(manifest_record["case_number"]),
        "court": str(manifest_record["court"]),
        "judgment_date": str(manifest_record["judgment_date"]),
        "source": str(manifest_record["source_name"]),
        "source_url": str(manifest_record["direct_pdf_url"]),
        "raw_text": raw_text,
    }


def _validate_acquired_metadata(record: Mapping[str, Any]) -> tuple[str, int]:
    validate_source_record(record)
    sha256 = str(record.get("sha256") or "").strip().lower()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise MissingMetadataError("manifest SHA-256 is missing or invalid")
    try:
        file_size = int(record["file_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise MissingMetadataError("manifest file_size is missing or invalid") from error
    if file_size <= 0:
        raise MissingMetadataError("manifest file_size must be positive")
    if record.get("mime_type") != "application/pdf":
        raise MissingMetadataError("manifest MIME type must be application/pdf")
    if not str(record.get("retrieval_timestamp") or "").strip():
        raise MissingMetadataError("manifest retrieval_timestamp is missing")
    return sha256, file_size


def _matches_manifest(
    processed: Mapping[str, Any], manifest_record: Mapping[str, Any]
) -> bool:
    expected = _canonical_record(manifest_record, str(processed.get("raw_text") or ""))
    return bool(expected["raw_text"]) and all(
        processed.get(field) == value for field, value in expected.items()
    )


class PilotCorpusExtractor:
    """Convert pilot PDFs with a manifest/hash-backed resume checkpoint."""

    def __init__(self, *, min_text_characters: int = 200) -> None:
        if min_text_characters <= 0:
            raise ValueError("min_text_characters must be positive")
        self.min_text_characters = min_text_characters

    def run(
        self,
        *,
        manifest_path: Path,
        raw_dir: Path,
        output_path: Path,
        checkpoint_path: Path,
        failure_log: Path,
        audit_path: Path,
        restart: bool = False,
    ) -> ExtractionAudit:
        records = load_manifest(manifest_path)
        manifest_hash = _sha256_file(manifest_path)

        if restart:
            processed_by_url: dict[str, dict[str, Any]] = {}
            successes: dict[str, dict[str, Any]] = {}
            _atomic_jsonl(output_path, [])
        elif checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("manifest_sha256") != manifest_hash:
                raise ExtractionError(
                    "manifest changed since extraction checkpoint; use --restart"
                )
            processed_by_url = _load_processed(output_path)
            successes = dict(checkpoint.get("successes") or {})
        elif output_path.exists():
            raise ExtractionError(
                "processed output exists without a checkpoint; use --restart"
            )
        else:
            processed_by_url = {}
            successes = {}

        audit = ExtractionAudit()
        current_urls = {str(record.get("direct_pdf_url") or "") for record in records}
        processed_by_url = {
            url: record for url, record in processed_by_url.items() if url in current_urls
        }
        successes = {url: detail for url, detail in successes.items() if url in current_urls}

        for record_number, record in enumerate(records, start=1):
            source_url = str(record.get("direct_pdf_url") or "")
            manifest_pdf_hash = str(record.get("sha256") or "").lower()
            resume_detail = successes.get(source_url)
            existing = processed_by_url.get(source_url)

            if (
                resume_detail is not None
                and existing is not None
                and resume_detail.get("sha256") == manifest_pdf_hash
                and _matches_manifest(existing, record)
            ):
                audit.extracted.append({**resume_detail, "resumed": True})
                continue

            try:
                manifest_pdf_hash, manifest_file_size = _validate_acquired_metadata(
                    record
                )
                source_path = raw_dir / str(record["local_filename"])
                if not source_path.is_file():
                    raise ExtractionError(f"source PDF is missing: {source_path}")
                content = source_path.read_bytes()
                extracted = extract_pdf_text(
                    content,
                    min_text_characters=self.min_text_characters,
                )
                validate_document_type(record, extracted.raw_text)
                if extracted.sha256 != manifest_pdf_hash:
                    raise ExtractionError(
                        f"source PDF hash differs from manifest: {source_path}"
                    )
                if manifest_file_size != extracted.file_size:
                    raise ExtractionError(
                        f"source PDF size differs from manifest: {source_path}"
                    )

                canonical = _canonical_record(record, extracted.raw_text)
                detail = {
                    "record_number": record_number,
                    "local_filename": record["local_filename"],
                    "source_url": source_url,
                    "sha256": extracted.sha256,
                    "page_count": extracted.page_count,
                    "text_characters": extracted.text_characters,
                    "form_feed_boundaries": extracted.raw_text.count("\f"),
                    "resumed": False,
                }
                processed_by_url[source_url] = canonical
                successes[source_url] = {**detail, "resumed": False}
                audit.extracted.append(detail)
                ordered = [
                    processed_by_url[str(item["direct_pdf_url"])]
                    for item in records
                    if str(item["direct_pdf_url"]) in processed_by_url
                ]
                _atomic_jsonl(output_path, ordered)
                _atomic_json(
                    checkpoint_path,
                    {
                        "manifest_sha256": manifest_hash,
                        "successes": successes,
                    },
                )
            except MissingMetadataError as error:
                detail = {
                    "record_number": record_number,
                    "title": record.get("title"),
                    "local_filename": record.get("local_filename"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                audit.missing_metadata.append(detail)
                _append_failure(failure_log, detail)
            except (InvalidPdfError, ScannedPdfError, ExtractionError) as error:
                detail = {
                    "record_number": record_number,
                    "title": record.get("title"),
                    "local_filename": record.get("local_filename"),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                audit.failures.append(detail)
                _append_failure(failure_log, detail)

        ordered = [
            processed_by_url[str(item["direct_pdf_url"])]
            for item in records
            if str(item["direct_pdf_url"]) in processed_by_url
        ]
        _atomic_jsonl(output_path, ordered)
        _atomic_json(
            checkpoint_path,
            {"manifest_sha256": manifest_hash, "successes": successes},
        )
        _atomic_json(
            audit_path,
            audit.to_dict(manifest_path=manifest_path, output_path=output_path),
        )
        return audit
