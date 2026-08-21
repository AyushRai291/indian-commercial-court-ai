"""Relational models for cases, paragraphs, and statutes."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from legal_rag.database import Base


class Case(Base):
    """A normalized commercial-court judgment."""

    __tablename__ = "cases"
    __table_args__ = (
        Index("ix_cases_court_judgment_date", "court", "judgment_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(1000), nullable=False)
    case_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    court: Mapped[str | None] = mapped_column(String(500), nullable=True)
    judgment_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    document_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    paragraphs: Mapped[list["Paragraph"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Paragraph.id",
    )


class Paragraph(Base):
    """A deduplicated paragraph belonging to a case."""

    __tablename__ = "paragraphs"
    __table_args__ = (
        UniqueConstraint(
            "case_id", "text_hash", name="uq_paragraphs_case_id_text_hash"
        ),
        Index("ix_paragraphs_case_id_paragraph_number", "case_id", "paragraph_number"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paragraph_uid: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, index=True
    )
    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    paragraph_number: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    case: Mapped[Case] = relationship(back_populates="paragraphs")


class Statute(Base):
    """A statutory provision available to the legal corpus."""

    __tablename__ = "statutes"
    __table_args__ = (
        Index("ix_statutes_act_name_section", "act_name", "section"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    act_name: Mapped[str] = mapped_column(String(500), nullable=False)
    section: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
