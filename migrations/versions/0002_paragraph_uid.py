"""Add stable UUIDv5 paragraph identities.

Revision ID: 0002_paragraph_uid
Revises: 0001_legacy_schema
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

from legal_rag.corpus.identity import generate_paragraph_uid


revision: str = "0002_paragraph_uid"
down_revision: str | None = "0001_legacy_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    cases = sa.table(
        "cases",
        sa.column("id", sa.Integer()),
        sa.column("document_hash", sa.String(length=64)),
    )
    paragraphs = sa.table(
        "paragraphs",
        sa.column("id", sa.Integer()),
        sa.column("case_id", sa.Integer()),
        sa.column("paragraph_number", sa.Integer()),
        sa.column("text_hash", sa.String(length=64)),
    )
    paragraph_count = bind.scalar(
        sa.select(sa.func.count()).select_from(paragraphs)
    )
    rows = bind.execute(
        sa.select(
            paragraphs.c.id,
            paragraphs.c.paragraph_number,
            paragraphs.c.text_hash,
            cases.c.document_hash,
        ).select_from(
            paragraphs.join(cases, paragraphs.c.case_id == cases.c.id)
        )
    ).mappings().all()
    if len(rows) != paragraph_count:
        raise RuntimeError(
            "Cannot backfill paragraph_uid because one or more paragraphs "
            "do not reference an existing case"
        )

    generated_uids = [
        (
            row["id"],
            generate_paragraph_uid(
                row["document_hash"],
                row["paragraph_number"],
                row["text_hash"],
            ),
        )
        for row in rows
    ]
    if len({paragraph_uid for _id, paragraph_uid in generated_uids}) != len(
        generated_uids
    ):
        raise RuntimeError(
            "Cannot backfill paragraph_uid because deterministic UIDs are not unique"
        )

    # SQLite DDL is not transactional. Perform every data validation above before
    # the first schema change so a bad legacy row leaves a retryable baseline.
    op.add_column(
        "paragraphs",
        sa.Column("paragraph_uid", sa.String(length=36), nullable=True),
    )
    paragraphs_with_uid = sa.table(
        "paragraphs",
        sa.column("id", sa.Integer()),
        sa.column("paragraph_uid", sa.String(length=36)),
    )

    for paragraph_id, paragraph_uid in generated_uids:
        bind.execute(
            paragraphs_with_uid.update()
            .where(paragraphs_with_uid.c.id == paragraph_id)
            .values(paragraph_uid=paragraph_uid)
        )

    missing_uids = bind.scalar(
        sa.select(sa.func.count())
        .select_from(paragraphs_with_uid)
        .where(paragraphs_with_uid.c.paragraph_uid.is_(None))
    )
    if missing_uids:
        raise RuntimeError(
            f"Unable to backfill paragraph_uid for {missing_uids} paragraph(s); "
            "verify that every paragraph references an existing case"
        )

    with op.batch_alter_table("paragraphs") as batch_op:
        batch_op.alter_column(
            "paragraph_uid",
            existing_type=sa.String(length=36),
            nullable=False,
        )
        batch_op.create_index(
            "ix_paragraphs_paragraph_uid",
            ["paragraph_uid"],
            unique=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("paragraphs") as batch_op:
        batch_op.drop_index("ix_paragraphs_paragraph_uid")
        batch_op.drop_column("paragraph_uid")
