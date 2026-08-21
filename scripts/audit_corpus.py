#!/usr/bin/env python3
"""Audit PostgreSQL corpus quality and optional Qdrant identity coverage."""

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
from legal_rag.corpus_audit import (  # noqa: E402
    build_corpus_audit,
    compare_vector_coverage,
    paragraph_uids,
    write_audit,
)
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.schema_migrations import upgrade_database  # noqa: E402
from legal_rag.vector import QdrantParagraphIndex  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit corpus quality and PostgreSQL/Qdrant identity coverage."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/judgments_pilot_corpus_audit.json"),
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--very-short-characters", type=int, default=20)
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Write relational quality metrics without vector coverage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    database_url = args.database_url or settings.database_url
    upgrade_database(database_url)
    session_factory = get_session_factory(get_engine(database_url))

    with session_factory() as session:
        audit = build_corpus_audit(
            session, very_short_characters=args.very_short_characters
        )
        postgres_uids = paragraph_uids(session)

    if not args.skip_qdrant:
        paragraph_index = QdrantParagraphIndex(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key,
            collection_name=args.collection or settings.qdrant_collection,
        )
        paragraph_index.validate_collection(settings.embedding_dimension)
        audit["qdrant_coverage"] = compare_vector_coverage(
            postgres_uids,
            [str(value) for value in paragraph_index.list_point_ids()],
        )

    write_audit(args.output, audit)
    summary = {**audit["counts"]}
    if "qdrant_coverage" in audit:
        summary.update(audit["qdrant_coverage"])
    print(json.dumps(summary, sort_keys=True))

    coverage = audit.get("qdrant_coverage")
    if coverage and any(
        coverage[field]
        for field in ("missing_points", "stale_orphan_points", "duplicate_points")
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
