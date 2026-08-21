#!/usr/bin/env python3
"""Safely migrate an empty, versioned, or validated legacy corpus database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.schema_migrations import SchemaMigrationError, upgrade_database  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and upgrade the corpus database to Alembic head."
    )
    parser.add_argument(
        "--database-url",
        help="SQLAlchemy URL (defaults to DATABASE_URL)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database_url = args.database_url or get_settings().database_url
    try:
        initial_state = upgrade_database(database_url)
    except SchemaMigrationError as error:
        print(f"Migration refused: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Migration failed: {error}", file=sys.stderr)
        return 1

    print(f"Database upgraded to head (initial state: {initial_state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
