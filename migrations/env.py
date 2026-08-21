"""Alembic runtime configuration."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection


# Make the src-layout application package importable for direct Alembic CLI use.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_ROOT = _PROJECT_ROOT / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from legal_rag.database import Base  # noqa: E402
from legal_rag import models as _models  # noqa: E402,F401


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the URL, allowing callers to override the process environment."""

    attribute_url = config.attributes.get("database_url")
    if attribute_url:
        return str(attribute_url)

    environment_url = os.getenv("DATABASE_URL")
    if environment_url:
        return environment_url

    configured_url = config.get_main_option("sqlalchemy.url")
    if not configured_url:
        raise RuntimeError(
            "No database URL configured; set DATABASE_URL or sqlalchemy.url"
        )
    return configured_url


def _configure_migration_context(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite",
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Reject SQL-only migrations because legacy validation needs inspection."""

    raise RuntimeError(
        "Offline Alembic migrations are not supported: the baseline revision "
        "must inspect and validate any existing legacy schema"
    )


def run_migrations_online() -> None:
    """Run migrations against either a supplied connection or a configured URL."""

    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        _configure_migration_context(supplied_connection)
        return

    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _configure_migration_context(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
