from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import func, select

from legal_rag.database import Base, get_engine, get_session_factory
from legal_rag.models import Case, Paragraph


def _load_ingestion_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "ingest_corpus.py"
    )
    module_name = "ingest_corpus_script_under_test"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_ingestion_continues_checkpoints_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    ingestion = _load_ingestion_script()
    input_path = tmp_path / "commercial_cases.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"
    failed_dir = tmp_path / "failed"
    database_path = tmp_path / "corpus.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"

    first_case = {
        "title": "Alpha Pvt. Ltd. v. Beta Ltd.",
        "court": "Delhi High Court",
        "judgment_date": "2024-05-16",
        "raw_text": (
            "1. The parties entered into a supply agreement.\n\n"
            "2. The defendant committed a material breach."
        ),
    }
    duplicate_case = {
        "title": "Alternate source title",
        "raw_text": (
            "  1.   The parties entered into a supply agreement.  \n\n"
            "  2. The defendant committed a material breach.  "
        ),
    }
    missing_text = {"title": "Record without judgment text"}
    final_case = {
        "case_title": "Gamma Industries v. Union of India",
        "court_name": "Bombay High Court",
        "date": "2023-11-02",
        "text": "1. The petition is allowed.",
    }

    input_path.write_text(
        "\n".join(
            (
                json.dumps(first_case),
                json.dumps(duplicate_case),
                "{not valid json",
                json.dumps(missing_text),
                json.dumps(final_case),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    progress = ingestion.ingest_file(
        input_path,
        database_url=database_url,
        checkpoint_dir=checkpoint_dir,
        failed_dir=failed_dir,
        default_source="test-feed",
    )

    assert progress.last_processed_line == 5
    assert progress.inserted == 2
    assert progress.duplicates == 1
    assert progress.failed == 2

    failure_files = list(failed_dir.glob("*.jsonl"))
    assert len(failure_files) == 1
    failures = [
        json.loads(line)
        for line in failure_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert [failure["line_number"] for failure in failures] == [3, 4]
    assert failures[0]["error_type"] == "JSONDecodeError"
    assert failures[0]["raw_line"] == "{not valid json"
    assert failures[1]["error_type"] == "NormalizationError"
    assert failures[1]["record"] == missing_text
    assert "no usable judgment text" in failures[1]["error"]

    checkpoint_files = list(checkpoint_dir.glob("*.json"))
    assert len(checkpoint_files) == 1
    checkpoint = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
    assert checkpoint["input_path"] == str(input_path.resolve())
    assert checkpoint["last_processed_line"] == 5
    assert checkpoint["inserted"] == 2
    assert checkpoint["duplicates"] == 1
    assert checkpoint["failed"] == 2
    assert checkpoint["updated_at"]

    engine = get_engine(database_url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Case)) == 2
        assert session.scalar(select(func.count()).select_from(Paragraph)) == 3
        assert set(session.scalars(select(Case.title))) == {
            "Alpha Pvt. Ltd. v. Beta Ltd.",
            "Gamma Industries v. Union of India",
        }

    normalization_calls: list[dict[str, object]] = []
    original_normalize_record = ingestion.normalize_record

    def tracking_normalize_record(record, **kwargs):
        normalization_calls.append(record)
        return original_normalize_record(record, **kwargs)

    monkeypatch.setattr(ingestion, "normalize_record", tracking_normalize_record)
    resumed_progress = ingestion.ingest_file(
        input_path,
        database_url=database_url,
        checkpoint_dir=checkpoint_dir,
        failed_dir=failed_dir,
        default_source="test-feed",
    )

    assert normalization_calls == []
    assert resumed_progress == progress
    assert len(failure_files[0].read_text(encoding="utf-8").splitlines()) == 2
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Case)) == 2
        assert session.scalar(select(func.count()).select_from(Paragraph)) == 3

    engine.dispose()


def test_changed_input_requires_explicit_restart(tmp_path: Path) -> None:
    ingestion = _load_ingestion_script()
    input_path = tmp_path / "replaceable.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"
    failed_dir = tmp_path / "failed"
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'corpus.db').as_posix()}"
    input_path.write_text(
        json.dumps({"title": "First case", "raw_text": "1. First text."}) + "\n",
        encoding="utf-8",
    )

    ingestion.ingest_file(
        input_path,
        database_url=database_url,
        checkpoint_dir=checkpoint_dir,
        failed_dir=failed_dir,
    )
    input_path.write_text(
        json.dumps({"title": "Replacement case", "raw_text": "1. New text."})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Input content changed"):
        ingestion.ingest_file(
            input_path,
            database_url=database_url,
            checkpoint_dir=checkpoint_dir,
            failed_dir=failed_dir,
        )

    restarted = ingestion.ingest_file(
        input_path,
        database_url=database_url,
        checkpoint_dir=checkpoint_dir,
        failed_dir=failed_dir,
        restart=True,
    )
    assert restarted.inserted == 1
    assert restarted.last_processed_line == 1


def test_invalid_utf8_line_is_logged_and_later_records_continue(
    tmp_path: Path,
) -> None:
    ingestion = _load_ingestion_script()
    input_path = tmp_path / "encoding-errors.jsonl"
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'encoding.db').as_posix()}"
    valid_first = json.dumps(
        {"title": "Valid first", "raw_text": "1. First judgment."}
    ).encode("utf-8")
    invalid = b'{"title":"Broken","raw_text":"\xff"}'
    valid_last = json.dumps(
        {"title": "Valid last", "raw_text": "1. Last judgment."}
    ).encode("utf-8")
    input_path.write_bytes(b"\n".join((valid_first, invalid, valid_last)) + b"\n")

    progress = ingestion.ingest_file(
        input_path,
        database_url=database_url,
        checkpoint_dir=tmp_path / "checkpoints",
        failed_dir=tmp_path / "failed",
    )

    assert progress.inserted == 2
    assert progress.failed == 1
    assert progress.last_processed_line == 3
    failure = json.loads(
        next((tmp_path / "failed").glob("*.jsonl"))
        .read_text(encoding="utf-8")
        .strip()
    )
    assert failure["line_number"] == 2
    assert failure["error_type"] == "UnicodeDecodeError"


def test_checkpoint_detects_a_cleared_database(tmp_path: Path) -> None:
    ingestion = _load_ingestion_script()
    input_path = tmp_path / "database-reset.jsonl"
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'reset.db').as_posix()}"
    input_path.write_text(
        json.dumps({"title": "Stored case", "raw_text": "1. Stored text."}) + "\n",
        encoding="utf-8",
    )
    options = {
        "database_url": database_url,
        "checkpoint_dir": tmp_path / "checkpoints",
        "failed_dir": tmp_path / "failed",
    }

    ingestion.ingest_file(input_path, **options)
    engine = get_engine(database_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with pytest.raises(ValueError, match="rows missing"):
        ingestion.ingest_file(input_path, **options)

    rebuilt = ingestion.ingest_file(input_path, restart=True, **options)
    assert rebuilt.inserted == 1
    engine.dispose()
