#!/usr/bin/env python3
"""Validate gold queries against authoritative PostgreSQL corpus metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.evaluation import (  # noqa: E402
    GoldValidationError,
    load_gold_queries,
    validate_gold_queries,
    write_review_markdown,
)
from legal_rag.schema_migrations import upgrade_database  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate gold retrieval queries against PostgreSQL."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("data/evaluation/gold_queries.jsonl"),
        help="Gold JSONL path (default: %(default)s)",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help="Optionally write deterministic human-review Markdown",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=40,
        help="Required number of query records (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    database_url = args.database_url or get_settings().database_url
    try:
        queries = load_gold_queries(args.dataset)
        upgrade_database(database_url)
        session_factory = get_session_factory(get_engine(database_url))
        with session_factory() as session:
            statistics = validate_gold_queries(
                queries, session, expected_count=args.expected_count
            )
            if args.review_output is not None:
                write_review_markdown(
                    args.review_output,
                    queries,
                    session,
                    expected_count=args.expected_count,
                )
    except GoldValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(statistics.to_dict(), indent=2, sort_keys=True))
    if args.review_output is not None:
        print(f"Review Markdown written to {args.review_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
