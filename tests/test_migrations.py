from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    inspect,
    text,
)
from sqlalchemy.engine import Engine

from legal_rag.corpus import generate_paragraph_uid
from legal_rag.database import get_engine
from legal_rag.schema_migrations import (
    BASELINE_REVISION,
    HEAD_REVISION,
    SchemaMigrationError,
    upgrade_database,
    validate_head_schema,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _database_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def _alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location", str(PROJECT_ROOT / "migrations")
    )
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    config.attributes["database_url"] = database_url
    return config


def _version(engine: Engine) -> str:
    with engine.connect() as connection:
        return connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()


def _create_old_create_all_schema(engine: Engine) -> None:
    """Independently reproduce the schema used before Alembic existed."""

    metadata = MetaData()
    cases = Table(
        "cases",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("title", String(1000), nullable=False),
        Column("case_number", String(255)),
        Column("court", String(500)),
        Column("judgment_date", Date),
        Column("source", String(255)),
        Column("source_url", Text),
        Column("raw_text", Text, nullable=False),
        Column("document_hash", String(64), nullable=False, unique=True, index=True),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
    )
    Index("ix_cases_court_judgment_date", cases.c.court, cases.c.judgment_date)
    paragraphs = Table(
        "paragraphs",
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "case_id",
            Integer,
            ForeignKey("cases.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        Column("paragraph_number", Integer, nullable=False),
        Column("page_number", Integer),
        Column("text", Text, nullable=False),
        Column("text_hash", String(64), nullable=False),
        UniqueConstraint(
            "case_id", "text_hash", name="uq_paragraphs_case_id_text_hash"
        ),
    )
    Index(
        "ix_paragraphs_case_id_paragraph_number",
        paragraphs.c.case_id,
        paragraphs.c.paragraph_number,
    )
    statutes = Table(
        "statutes",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("act_name", String(500), nullable=False),
        Column("section", String(100), nullable=False),
        Column("title", String(1000)),
        Column("text", Text, nullable=False),
    )
    Index("ix_statutes_act_name_section", statutes.c.act_name, statutes.c.section)
    metadata.create_all(engine)


def test_alembic_upgrades_empty_database_from_zero_to_head(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "empty.sqlite3")

    command.upgrade(_alembic_config(database_url), "head")
    engine = get_engine(database_url)

    try:
        assert upgrade_database(database_url) == "head"
        assert _version(engine) == HEAD_REVISION
        assert set(inspect(engine).get_table_names()) == {
            "alembic_version",
            "cases",
            "paragraphs",
            "statutes",
        }
        validate_head_schema(engine)
    finally:
        engine.dispose()


def test_alembic_adopts_legacy_database_and_backfills_runtime_uids(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "legacy.sqlite3")
    engine = get_engine(database_url)
    _create_old_create_all_schema(engine)
    document_hash = sha256(b"legacy judgment").hexdigest()
    first_text_hash = sha256(b"first paragraph").hexdigest()
    second_text_hash = sha256(b"second paragraph").hexdigest()

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO cases (
                        id, title, case_number, court, judgment_date, source,
                        source_url, raw_text, document_hash
                    ) VALUES (
                        :id, :title, :case_number, :court, :judgment_date,
                        :source, :source_url, :raw_text, :document_hash
                    )
                    """
                ),
                {
                    "id": 17,
                    "title": "Legacy Industries Ltd. v. Buyer Ltd.",
                    "case_number": "CS(COMM) 17/2024",
                    "court": "Delhi High Court",
                    "judgment_date": "2024-04-12",
                    "source": "legacy-import",
                    "source_url": "https://example.test/legacy/17",
                    "raw_text": "1. First paragraph.\n\n2. Second paragraph.",
                    "document_hash": document_hash,
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO paragraphs (
                        id, case_id, paragraph_number, page_number, text,
                        text_hash
                    ) VALUES (
                        :id, :case_id, :paragraph_number, :page_number, :text,
                        :text_hash
                    )
                    """
                ),
                [
                    {
                        "id": 31,
                        "case_id": 17,
                        "paragraph_number": 1,
                        "page_number": 1,
                        "text": "First paragraph.",
                        "text_hash": first_text_hash,
                    },
                    {
                        "id": 32,
                        "case_id": 17,
                        "paragraph_number": 2,
                        "page_number": 1,
                        "text": "Second paragraph.",
                        "text_hash": second_text_hash,
                    },
                ],
            )
        assert "alembic_version" not in inspect(engine).get_table_names()

        initial_state = upgrade_database(database_url)

        assert initial_state == "legacy"
        assert _version(engine) == HEAD_REVISION
        validate_head_schema(engine)
        with engine.connect() as connection:
            migrated = connection.execute(
                text(
                    """
                    SELECT paragraph_number, text_hash, paragraph_uid
                    FROM paragraphs
                    ORDER BY paragraph_number
                    """
                )
            ).mappings().all()
            uniqueness = connection.execute(
                text(
                    """
                    SELECT COUNT(*) AS total,
                           COUNT(DISTINCT paragraph_uid) AS distinct_uids
                    FROM paragraphs
                    WHERE paragraph_uid IS NOT NULL
                    """
                )
            ).mappings().one()

        assert uniqueness == {"total": 2, "distinct_uids": 2}
        assert [row["paragraph_uid"] for row in migrated] == [
            generate_paragraph_uid(
                document_hash,
                row["paragraph_number"],
                row["text_hash"],
            )
            for row in migrated
        ]
    finally:
        engine.dispose()


def test_unversioned_partial_schema_is_refused_without_being_stamped(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "partial.sqlite3")
    engine = get_engine(database_url)

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE cases (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
                )
            )

        with pytest.raises(SchemaMigrationError, match="expected corpus schema"):
            upgrade_database(database_url)

        assert set(inspect(engine).get_table_names()) == {"cases"}
        assert "alembic_version" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_unknown_alembic_revision_is_refused_without_rewriting_it(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "unknown-version.sqlite3")
    upgrade_database(database_url)
    engine = get_engine(database_url)

    try:
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE alembic_version SET version_num = :revision"),
                {"revision": "foreign_revision"},
            )

        with pytest.raises(SchemaMigrationError, match="Unknown Alembic revision"):
            upgrade_database(database_url)

        assert _version(engine) == "foreign_revision"
    finally:
        engine.dispose()


def test_versioned_baseline_with_wrong_index_shape_is_refused(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "wrong-index.sqlite3")
    engine = get_engine(database_url)
    _create_old_create_all_schema(engine)
    command.stamp(_alembic_config(database_url), BASELINE_REVISION)

    try:
        with engine.begin() as connection:
            connection.execute(text("DROP INDEX ix_cases_court_judgment_date"))
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX ix_cases_court_judgment_date "
                    "ON cases (court, judgment_date)"
                )
            )

        with pytest.raises(SchemaMigrationError, match="unique=True"):
            upgrade_database(database_url)

        assert _version(engine) == BASELINE_REVISION
        assert "paragraph_uid" not in {
            column["name"] for column in inspect(engine).get_columns("paragraphs")
        }
    finally:
        engine.dispose()


def test_invalid_legacy_hash_fails_before_ddl_and_can_be_retried(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path / "retryable.sqlite3")
    engine = get_engine(database_url)
    _create_old_create_all_schema(engine)
    command.stamp(_alembic_config(database_url), BASELINE_REVISION)
    document_hash = sha256(b"retryable legacy judgment").hexdigest()
    corrected_text_hash = sha256(b"corrected paragraph").hexdigest()

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO cases (id, title, raw_text, document_hash) "
                    "VALUES (1, 'Retryable case', 'Legacy text', :document_hash)"
                ),
                {"document_hash": document_hash},
            )
            connection.execute(
                text(
                    "INSERT INTO paragraphs "
                    "(id, case_id, paragraph_number, text, text_hash) "
                    "VALUES (1, 1, 1, 'Legacy paragraph', 'not-a-sha256')"
                )
            )

        with pytest.raises(ValueError, match="paragraph_text_hash"):
            upgrade_database(database_url)

        assert _version(engine) == BASELINE_REVISION
        assert "paragraph_uid" not in {
            column["name"] for column in inspect(engine).get_columns("paragraphs")
        }

        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE paragraphs SET text_hash = :text_hash WHERE id = 1"
                ),
                {"text_hash": corrected_text_hash},
            )

        assert upgrade_database(database_url) == "baseline"
        assert _version(engine) == HEAD_REVISION
        with engine.connect() as connection:
            migrated_uid = connection.execute(
                text("SELECT paragraph_uid FROM paragraphs WHERE id = 1")
            ).scalar_one()
        assert migrated_uid == generate_paragraph_uid(
            document_hash, 1, corrected_text_hash
        )
    finally:
        engine.dispose()
