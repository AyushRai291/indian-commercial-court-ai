#!/usr/bin/env python3
"""Resumable ingestion of canonicalized legal cases from a JSONL corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.corpus import normalize_record  # noqa: E402
from legal_rag.database import get_engine, get_session_factory, init_db  # noqa: E402
from legal_rag.models import Case  # noqa: E402
from legal_rag.services.ingestion import insert_case  # noqa: E402


@dataclass
class IngestionProgress:
    """Durable progress and cumulative outcome counters for one input file."""

    input_path: str
    input_sha256: str
    database_fingerprint: str
    last_processed_line: int = 0
    inserted: int = 0
    duplicates: int = 0
    failed: int = 0
    last_persisted_document_hash: str | None = None
    updated_at: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IngestionProgress":
        return cls(
            input_path=str(value["input_path"]),
            input_sha256=str(value.get("input_sha256", "")),
            database_fingerprint=str(value.get("database_fingerprint", "")),
            last_processed_line=int(value.get("last_processed_line", 0)),
            inserted=int(value.get("inserted", 0)),
            duplicates=int(value.get("duplicates", 0)),
            failed=int(value.get("failed", 0)),
            last_persisted_document_hash=(
                str(value["last_persisted_document_hash"])
                if value.get("last_persisted_document_hash")
                else None
            ),
            updated_at=str(value.get("updated_at", "")),
        )


class CheckpointStore:
    """Atomically persist progress for a single, resolved input path."""

    def __init__(
        self, checkpoint_dir: Path, input_path: Path, database_url: str
    ) -> None:
        resolved = str(input_path.resolve())
        self.database_fingerprint = hashlib.sha256(
            database_url.encode("utf-8")
        ).hexdigest()
        checkpoint_identity = f"{resolved}\0{self.database_fingerprint}"
        key = hashlib.sha256(checkpoint_identity.encode("utf-8")).hexdigest()[:16]
        self.path = checkpoint_dir / f"{input_path.stem}-{key}.json"
        self.input_path = resolved
        self.input_sha256 = _sha256_file(input_path)

    def load(self, *, restart: bool = False) -> IngestionProgress:
        if restart or not self.path.exists():
            return IngestionProgress(
                input_path=self.input_path,
                input_sha256=self.input_sha256,
                database_fingerprint=self.database_fingerprint,
            )

        with self.path.open("r", encoding="utf-8") as handle:
            progress = IngestionProgress.from_dict(json.load(handle))
        if progress.input_path != self.input_path:
            raise ValueError(
                f"Checkpoint {self.path} belongs to {progress.input_path}, "
                f"not {self.input_path}"
            )
        if progress.input_sha256 != self.input_sha256:
            raise ValueError(
                f"Input content changed since checkpoint {self.path}; "
                "use --restart to scan the replacement file from line one"
            )
        if progress.database_fingerprint != self.database_fingerprint:
            raise ValueError(
                "Checkpoint database target does not match the configured "
                "DATABASE_URL; use --restart for the new target"
            )
        return progress

    def save(self, progress: IngestionProgress) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        progress.updated_at = datetime.now(timezone.utc).isoformat()
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(progress), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.path)


def _sha256_file(input_path: Path) -> str:
    digest = hashlib.sha256()
    with input_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records_after(input_path: Path, line_number: int) -> Iterator[tuple[int, bytes]]:
    with input_path.open("rb") as handle:
        for current_line, raw_line in enumerate(handle, start=1):
            if current_line > line_number:
                yield current_line, raw_line


def _failure_path(failed_dir: Path, input_path: Path) -> Path:
    key = hashlib.sha256(str(input_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return failed_dir / f"{input_path.stem}-{key}.jsonl"


def _write_failure(
    failure_path: Path,
    *,
    line_number: int,
    error: Exception,
    raw_line: str,
    record: Any | None,
) -> None:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "line_number": line_number,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if record is not None:
        entry["record"] = record
    else:
        entry["raw_line"] = raw_line.rstrip("\r\n")

    with failure_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def ingest_file(
    input_path: Path,
    *,
    database_url: str,
    checkpoint_dir: Path,
    failed_dir: Path,
    default_source: str | None = None,
    restart: bool = False,
) -> IngestionProgress:
    """Ingest ``input_path`` and return its durable cumulative progress."""

    if not input_path.is_file():
        raise FileNotFoundError(f"JSONL input does not exist: {input_path}")

    checkpoint = CheckpointStore(checkpoint_dir, input_path, database_url)
    progress = checkpoint.load(restart=restart)

    engine = get_engine(database_url)
    init_db(engine)
    session_factory = get_session_factory(engine)
    failure_path = _failure_path(failed_dir, input_path)

    if progress.last_persisted_document_hash:
        with session_factory() as session:
            persisted_case_id = session.scalar(
                select(Case.id)
                .where(
                    Case.document_hash == progress.last_persisted_document_hash
                )
                .limit(1)
            )
        if persisted_case_id is None:
            raise ValueError(
                "Checkpoint refers to corpus rows missing from the configured "
                "database; use --restart to rebuild this target"
            )

    if restart:
        # Replace stale progress only after database initialization succeeds.
        # Failure history remains append-only under data/failed.
        checkpoint.save(progress)

    for line_number, raw_line_bytes in _records_after(
        input_path, progress.last_processed_line
    ):
        parsed_record: Any | None = None
        encoding = "utf-8-sig" if line_number == 1 else "utf-8"
        raw_line = raw_line_bytes.decode(encoding, errors="replace")
        try:
            raw_line = raw_line_bytes.decode(encoding)
            if not raw_line.strip():
                raise ValueError("blank JSONL line")
            parsed_record = json.loads(raw_line)
            if not isinstance(parsed_record, dict):
                raise TypeError("each JSONL record must be a JSON object")

            canonical_case = normalize_record(
                parsed_record, default_source=default_source
            )
            with session_factory.begin() as session:
                result = insert_case(session, canonical_case)

            if result.inserted:
                progress.inserted += 1
            else:
                progress.duplicates += 1
            progress.last_persisted_document_hash = canonical_case.document_hash
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:  # one bad source record must not stop the run
            _write_failure(
                failure_path,
                line_number=line_number,
                error=error,
                raw_line=raw_line,
                record=parsed_record,
            )
            progress.failed += 1

        progress.last_processed_line = line_number
        checkpoint.save(progress)

    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize and ingest a JSONL legal corpus into PostgreSQL."
    )
    parser.add_argument("input", type=Path, help="Path to the source JSONL file")
    parser.add_argument(
        "--database-url",
        help="SQLAlchemy database URL (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--source",
        help="Fallback source name when a record does not contain one",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Checkpoint directory (default: data/checkpoints)",
    )
    parser.add_argument(
        "--failed-dir",
        type=Path,
        default=Path("data/failed"),
        help="Failed-record directory (default: data/failed)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start again from line one (database deduplication still applies)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    try:
        progress = ingest_file(
            args.input,
            database_url=args.database_url or settings.database_url,
            checkpoint_dir=args.checkpoint_dir,
            failed_dir=args.failed_dir,
            default_source=args.source,
            restart=args.restart,
        )
    except Exception as error:
        print(f"Ingestion could not start: {error}", file=sys.stderr)
        return 1

    print(
        "Ingestion complete: "
        f"line={progress.last_processed_line} "
        f"inserted={progress.inserted} "
        f"duplicates={progress.duplicates} "
        f"failed={progress.failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
