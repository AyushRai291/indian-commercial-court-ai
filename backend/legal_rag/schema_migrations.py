"""Safe Alembic upgrades for empty, versioned, and legacy corpus databases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.sql import sqltypes

from legal_rag.config import get_settings
from legal_rag.database import get_engine


BASELINE_REVISION = "0001_legacy_schema"
HEAD_REVISION = "0002_paragraph_uid"

DatabaseState = Literal["empty", "legacy", "baseline", "head"]


class SchemaMigrationError(RuntimeError):
    """Raised when a database cannot be identified and migrated safely."""


@dataclass(frozen=True, slots=True)
class _ColumnSpec:
    type_family: Literal["integer", "string", "text", "date", "datetime"]
    nullable: bool
    length: int | None = None


_I = _ColumnSpec("integer", False)
_NULLABLE_I = _ColumnSpec("integer", True)
_TEXT = _ColumnSpec("text", False)
_NULLABLE_TEXT = _ColumnSpec("text", True)

_EXPECTED_LEGACY_COLUMNS: dict[str, dict[str, _ColumnSpec]] = {
    "cases": {
        "id": _I,
        "title": _ColumnSpec("string", False, 1000),
        "case_number": _ColumnSpec("string", True, 255),
        "court": _ColumnSpec("string", True, 500),
        "judgment_date": _ColumnSpec("date", True),
        "source": _ColumnSpec("string", True, 255),
        "source_url": _NULLABLE_TEXT,
        "raw_text": _TEXT,
        "document_hash": _ColumnSpec("string", False, 64),
        "created_at": _ColumnSpec("datetime", False),
    },
    "paragraphs": {
        "id": _I,
        "case_id": _I,
        "paragraph_number": _I,
        "page_number": _NULLABLE_I,
        "text": _TEXT,
        "text_hash": _ColumnSpec("string", False, 64),
    },
    "statutes": {
        "id": _I,
        "act_name": _ColumnSpec("string", False, 500),
        "section": _ColumnSpec("string", False, 100),
        "title": _ColumnSpec("string", True, 1000),
        "text": _TEXT,
    },
}
_EXPECTED_HEAD_COLUMNS = {
    **_EXPECTED_LEGACY_COLUMNS,
    "paragraphs": {
        **_EXPECTED_LEGACY_COLUMNS["paragraphs"],
        "paragraph_uid": _ColumnSpec("string", False, 36),
    },
}
_EXPECTED_LEGACY_INDEXES: dict[
    str, dict[str, tuple[tuple[str, ...], bool]]
] = {
    "cases": {
        "ix_cases_court_judgment_date": (("court", "judgment_date"), False),
        "ix_cases_document_hash": (("document_hash",), True),
    },
    "paragraphs": {
        "ix_paragraphs_case_id": (("case_id",), False),
        "ix_paragraphs_case_id_paragraph_number": (
            ("case_id", "paragraph_number"),
            False,
        ),
    },
    "statutes": {
        "ix_statutes_act_name_section": (("act_name", "section"), False),
    },
}


def _unique_column_sets(inspector, table_name: str) -> set[tuple[str, ...]]:
    unique_sets = {
        tuple(constraint.get("column_names") or ())
        for constraint in inspector.get_unique_constraints(table_name)
    }
    unique_sets.update(
        tuple(index.get("column_names") or ())
        for index in inspector.get_indexes(table_name)
        if index.get("unique")
    )
    return unique_sets


def _matches_type(actual_type, spec: _ColumnSpec) -> bool:
    if spec.type_family == "integer":
        return isinstance(actual_type, sqltypes.Integer)
    if spec.type_family == "string":
        return (
            isinstance(actual_type, sqltypes.String)
            and not isinstance(actual_type, sqltypes.Text)
            and getattr(actual_type, "length", None) == spec.length
        )
    if spec.type_family == "text":
        return isinstance(actual_type, sqltypes.Text)
    if spec.type_family == "date":
        return isinstance(actual_type, sqltypes.Date) and not isinstance(
            actual_type, sqltypes.DateTime
        )
    return isinstance(actual_type, sqltypes.DateTime)


def _validate_columns(
    inspector, expected: dict[str, dict[str, _ColumnSpec]]
) -> None:
    application_tables = set(inspector.get_table_names()) - {"alembic_version"}
    expected_tables = set(expected)
    if application_tables != expected_tables:
        missing = sorted(expected_tables - application_tables)
        unexpected = sorted(application_tables - expected_tables)
        raise SchemaMigrationError(
            "Database is not the expected corpus schema: "
            f"missing tables={missing}, unexpected tables={unexpected}"
        )

    for table_name, expected_columns in expected.items():
        actual_columns = {
            column["name"]: column for column in inspector.get_columns(table_name)
        }
        if set(actual_columns) != set(expected_columns):
            missing = sorted(set(expected_columns) - set(actual_columns))
            unexpected = sorted(set(actual_columns) - set(expected_columns))
            raise SchemaMigrationError(
                f"Table {table_name!r} does not match the expected schema: "
                f"missing columns={missing}, unexpected columns={unexpected}"
            )
        for column_name, spec in expected_columns.items():
            column = actual_columns[column_name]
            if bool(column.get("nullable")) != spec.nullable:
                raise SchemaMigrationError(
                    f"{table_name}.{column_name} nullable={column.get('nullable')!r}; "
                    f"expected {spec.nullable}"
                )
            if not _matches_type(column["type"], spec):
                raise SchemaMigrationError(
                    f"{table_name}.{column_name} has type {column['type']!r}; "
                    f"expected {spec.type_family}"
                    + (f"({spec.length})" if spec.length is not None else "")
                )
            if (
                table_name == "cases"
                and column_name == "created_at"
                and column.get("default") is None
            ):
                raise SchemaMigrationError(
                    "cases.created_at must have a server default"
                )
        primary_key = tuple(
            inspector.get_pk_constraint(table_name).get("constrained_columns") or ()
        )
        if primary_key != ("id",):
            raise SchemaMigrationError(
                f"Table {table_name!r} must have primary key ('id',), "
                f"found {primary_key}"
            )


def _validate_common_constraints(inspector, *, head: bool) -> None:
    if ("document_hash",) not in _unique_column_sets(inspector, "cases"):
        raise SchemaMigrationError("cases.document_hash must be globally unique")
    if ("case_id", "text_hash") not in _unique_column_sets(
        inspector, "paragraphs"
    ):
        raise SchemaMigrationError(
            "paragraphs must be unique on (case_id, text_hash)"
        )

    expected_foreign_key = False
    for foreign_key in inspector.get_foreign_keys("paragraphs"):
        if (
            tuple(foreign_key.get("constrained_columns") or ()) == ("case_id",)
            and foreign_key.get("referred_table") == "cases"
            and tuple(foreign_key.get("referred_columns") or ()) == ("id",)
            and str((foreign_key.get("options") or {}).get("ondelete", "")).upper()
            == "CASCADE"
        ):
            expected_foreign_key = True
            break
    if not expected_foreign_key:
        raise SchemaMigrationError(
            "paragraphs.case_id must reference cases.id with ON DELETE CASCADE"
        )

    expected_indexes = {
        table_name: dict(indexes)
        for table_name, indexes in _EXPECTED_LEGACY_INDEXES.items()
    }
    if head:
        expected_indexes["paragraphs"]["ix_paragraphs_paragraph_uid"] = (
            ("paragraph_uid",),
            True,
        )
    for table_name, required_indexes in expected_indexes.items():
        actual_indexes = {
            index["name"]: index for index in inspector.get_indexes(table_name)
        }
        for index_name, (columns, unique) in required_indexes.items():
            index = actual_indexes.get(index_name)
            if index is None:
                raise SchemaMigrationError(
                    f"Table {table_name!r} is missing index {index_name!r}"
                )
            actual_columns = tuple(index.get("column_names") or ())
            actual_unique = bool(index.get("unique"))
            if actual_columns != columns or actual_unique != unique:
                raise SchemaMigrationError(
                    f"Index {index_name!r} has columns {actual_columns!r} and "
                    f"unique={actual_unique}; expected columns {columns!r} and "
                    f"unique={unique}"
                )


def validate_legacy_schema(engine: Engine) -> None:
    """Require an exact unversioned/baseline schema created by old ``create_all``."""

    inspector = inspect(engine)
    _validate_columns(inspector, _EXPECTED_LEGACY_COLUMNS)
    _validate_common_constraints(inspector, head=False)


def validate_head_schema(engine: Engine) -> None:
    """Require the expected current schema and durable UID constraint."""

    inspector = inspect(engine)
    _validate_columns(inspector, _EXPECTED_HEAD_COLUMNS)
    _validate_common_constraints(inspector, head=True)
    paragraph_columns = {
        column["name"]: column for column in inspector.get_columns("paragraphs")
    }
    if paragraph_columns["paragraph_uid"].get("nullable", True):
        raise SchemaMigrationError("paragraphs.paragraph_uid must be non-null")
    if ("paragraph_uid",) not in _unique_column_sets(inspector, "paragraphs"):
        raise SchemaMigrationError("paragraphs.paragraph_uid must be globally unique")


def _database_state(engine: Engine) -> DatabaseState:
    inspector = inspect(engine)
    all_tables = set(inspector.get_table_names())
    application_tables = all_tables - {"alembic_version"}
    has_version_table = "alembic_version" in all_tables

    if not has_version_table:
        if not application_tables:
            return "empty"
        validate_legacy_schema(engine)
        return "legacy"

    with engine.connect() as connection:
        revisions = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalars().all()
    if not revisions:
        if not application_tables:
            return "empty"
        validate_legacy_schema(engine)
        return "legacy"
    if len(revisions) != 1:
        raise SchemaMigrationError(
            "alembic_version must contain exactly one revision; "
            f"found {revisions!r}"
        )

    revision = revisions[0]
    if revision == BASELINE_REVISION:
        validate_legacy_schema(engine)
        return "baseline"
    if revision == HEAD_REVISION:
        validate_head_schema(engine)
        return "head"
    raise SchemaMigrationError(f"Unknown Alembic revision: {revision!r}")


def _alembic_config(database_url: str) -> Config:
    project_root = Path(__file__).resolve().parents[2]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    config.attributes["database_url"] = database_url
    return config


def upgrade_database(database_url: str | None = None) -> DatabaseState:
    """Upgrade a known corpus database to head and return its initial state.

    An unversioned database is accepted by the baseline revision only after its
    complete legacy table/column/constraint shape has been validated. Unknown
    and partial schemas fail closed instead of being versioned.
    """

    url = database_url or get_settings().database_url
    engine = get_engine(url)
    initial_state = _database_state(engine)
    config = _alembic_config(url)

    if initial_state != "head":
        command.upgrade(config, "head")

    validate_head_schema(engine)
    return initial_state
