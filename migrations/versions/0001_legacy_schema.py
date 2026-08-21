"""Create or validate the legacy corpus schema.

Revision ID: 0001_legacy_schema
Revises:
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from alembic import op
import sqlalchemy as sa


revision: str = "0001_legacy_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_EXPECTED_COLUMNS: Mapping[
    str, Mapping[str, tuple[type[sa.types.TypeEngine[Any]], int | None, bool]]
] = {
    "cases": {
        "id": (sa.Integer, None, False),
        "title": (sa.String, 1000, False),
        "case_number": (sa.String, 255, True),
        "court": (sa.String, 500, True),
        "judgment_date": (sa.Date, None, True),
        "source": (sa.String, 255, True),
        "source_url": (sa.Text, None, True),
        "raw_text": (sa.Text, None, False),
        "document_hash": (sa.String, 64, False),
        "created_at": (sa.DateTime, None, False),
    },
    "paragraphs": {
        "id": (sa.Integer, None, False),
        "case_id": (sa.Integer, None, False),
        "paragraph_number": (sa.Integer, None, False),
        "page_number": (sa.Integer, None, True),
        "text": (sa.Text, None, False),
        "text_hash": (sa.String, 64, False),
    },
    "statutes": {
        "id": (sa.Integer, None, False),
        "act_name": (sa.String, 500, False),
        "section": (sa.String, 100, False),
        "title": (sa.String, 1000, True),
        "text": (sa.Text, None, False),
    },
}

_EXPECTED_INDEXES: Mapping[
    str, Mapping[str, tuple[tuple[str, ...], bool]]
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


def _schema_error(detail: str) -> RuntimeError:
    return RuntimeError(
        "Existing database is not the expected unversioned legacy corpus "
        f"schema: {detail}. Refusing to mark it as migrated."
    )


def _validate_columns(inspector: sa.Inspector, table_name: str) -> None:
    expected = _EXPECTED_COLUMNS[table_name]
    actual = {column["name"]: column for column in inspector.get_columns(table_name)}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise _schema_error(
            f"table {table_name!r} has missing columns {missing} and extra columns {extra}"
        )

    for column_name, (expected_type, expected_length, expected_nullable) in expected.items():
        column = actual[column_name]
        actual_type = column["type"]
        if not isinstance(actual_type, expected_type):
            raise _schema_error(
                f"column {table_name}.{column_name} has type {actual_type!s}, "
                f"expected {expected_type.__name__}"
            )
        if expected_length is not None and getattr(actual_type, "length", None) != expected_length:
            raise _schema_error(
                f"column {table_name}.{column_name} has length "
                f"{getattr(actual_type, 'length', None)!r}, expected {expected_length}"
            )
        if bool(column["nullable"]) != expected_nullable:
            raise _schema_error(
                f"column {table_name}.{column_name} has nullable={column['nullable']!r}, "
                f"expected {expected_nullable}"
            )
        if (
            table_name == "cases"
            and column_name == "created_at"
            and column.get("default") is None
        ):
            raise _schema_error(
                "column cases.created_at is missing its server default"
            )

    primary_key = inspector.get_pk_constraint(table_name).get(
        "constrained_columns", []
    )
    if primary_key != ["id"]:
        raise _schema_error(
            f"table {table_name!r} has primary key {primary_key!r}, expected ['id']"
        )


def _validate_indexes(inspector: sa.Inspector, table_name: str) -> None:
    indexes = {index["name"]: index for index in inspector.get_indexes(table_name)}
    for index_name, (expected_columns, expected_unique) in _EXPECTED_INDEXES[
        table_name
    ].items():
        index = indexes.get(index_name)
        if index is None:
            raise _schema_error(
                f"table {table_name!r} is missing index {index_name!r}"
            )
        columns = tuple(index.get("column_names") or ())
        if columns != expected_columns or bool(index.get("unique")) != expected_unique:
            raise _schema_error(
                f"index {index_name!r} has columns {columns!r} and "
                f"unique={bool(index.get('unique'))}, expected columns "
                f"{expected_columns!r} and unique={expected_unique}"
            )


def _validate_legacy_schema(inspector: sa.Inspector) -> None:
    for table_name in _EXPECTED_COLUMNS:
        _validate_columns(inspector, table_name)
        _validate_indexes(inspector, table_name)

    paragraph_uniques = {
        constraint.get("name"): tuple(constraint.get("column_names") or ())
        for constraint in inspector.get_unique_constraints("paragraphs")
    }
    if paragraph_uniques.get("uq_paragraphs_case_id_text_hash") != (
        "case_id",
        "text_hash",
    ):
        raise _schema_error(
            "paragraphs is missing unique constraint "
            "'uq_paragraphs_case_id_text_hash'"
        )

    foreign_keys = inspector.get_foreign_keys("paragraphs")
    expected_foreign_key = any(
        tuple(key.get("constrained_columns") or ()) == ("case_id",)
        and key.get("referred_table") == "cases"
        and tuple(key.get("referred_columns") or ()) == ("id",)
        and str((key.get("options") or {}).get("ondelete", "")).upper()
        == "CASCADE"
        for key in foreign_keys
    )
    if not expected_foreign_key:
        raise _schema_error(
            "paragraphs is missing the expected case_id -> cases.id ON DELETE CASCADE foreign key"
        )


def _create_legacy_schema() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=1000), nullable=False),
        sa.Column("case_number", sa.String(length=255), nullable=True),
        sa.Column("court", sa.String(length=500), nullable=True),
        sa.Column("judgment_date", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("document_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cases_court_judgment_date",
        "cases",
        ["court", "judgment_date"],
        unique=False,
    )
    op.create_index(
        "ix_cases_document_hash", "cases", ["document_hash"], unique=True
    )

    op.create_table(
        "paragraphs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("paragraph_number", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "case_id",
            "text_hash",
            name="uq_paragraphs_case_id_text_hash",
        ),
    )
    op.create_index(
        "ix_paragraphs_case_id", "paragraphs", ["case_id"], unique=False
    )
    op.create_index(
        "ix_paragraphs_case_id_paragraph_number",
        "paragraphs",
        ["case_id", "paragraph_number"],
        unique=False,
    )

    op.create_table(
        "statutes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("act_name", sa.String(length=500), nullable=False),
        sa.Column("section", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=1000), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_statutes_act_name_section",
        "statutes",
        ["act_name", "section"],
        unique=False,
    )


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    expected_tables = set(_EXPECTED_COLUMNS)
    present_tables = set(inspector.get_table_names()) - {"alembic_version"}

    if not present_tables:
        _create_legacy_schema()
        return

    if present_tables != expected_tables:
        missing = sorted(expected_tables - present_tables)
        unexpected = sorted(present_tables - expected_tables)
        raise _schema_error(
            f"database tables do not match; missing tables are {missing} and "
            f"unexpected tables are {unexpected}"
        )

    _validate_legacy_schema(inspector)


def downgrade() -> None:
    op.drop_table("statutes")
    op.drop_table("paragraphs")
    op.drop_table("cases")
