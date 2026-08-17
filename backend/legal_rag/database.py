"""SQLAlchemy engine, metadata, and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from legal_rag.config import get_settings


class Base(DeclarativeBase):
    """Declarative base shared by all corpus models."""


@lru_cache(maxsize=8)
def _engine_for_url(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_engine(database_url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy engine for ``database_url``."""

    return _engine_for_url(database_url or get_settings().database_url)


def get_session_factory(
    engine: Engine | None = None,
) -> sessionmaker[Session]:
    """Create a session factory bound to the supplied (or default) engine."""

    return sessionmaker(
        bind=engine or get_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(
    session_factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    """Provide a transaction boundary that commits or rolls back as needed."""

    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> None:
    """Create all corpus tables if they do not yet exist."""

    # Importing the models registers their tables on Base.metadata.
    from legal_rag import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine or get_engine())
