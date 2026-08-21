#!/usr/bin/env python3
"""Build BM25 from PostgreSQL and print the top lexical paragraph matches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.retrieval import (  # noqa: E402
    BM25ParagraphRetriever,
    build_retrieval_filters,
)
from legal_rag.schema_migrations import upgrade_database  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the legal paragraphs most relevant under BM25.",
    )
    parser.add_argument("query", nargs="+", help="Lexical search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of matches to print (default: %(default)s)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL (defaults to DATABASE_URL)",
    )
    parser.add_argument("--court", default=None, help="Exact court metadata filter")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Exact judgment calendar year filter",
    )
    parser.add_argument(
        "--case-number",
        default=None,
        help="Exact canonical case-number filter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip()
    if not query:
        parser.error("query must not be empty")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        filters = build_retrieval_filters(
            court=args.court,
            year=args.year,
            case_number=args.case_number,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    settings = get_settings()
    database_url = args.database_url or settings.database_url
    upgrade_database(database_url)
    session_factory = get_session_factory(get_engine(database_url))

    build_started = perf_counter()
    with session_factory() as session:
        retriever = BM25ParagraphRetriever.from_session(session)
    build_seconds = perf_counter() - build_started

    query_started = perf_counter()
    results = retriever.search(query, top_k=args.top_k, filters=filters)
    query_seconds = perf_counter() - query_started
    if filters is not None:
        print(
            f"Filters: court={filters.court or '-'} year={filters.year or '-'} "
            f"case_number={filters.case_number or '-'}"
        )
    print(
        f"Indexed {retriever.indexed_paragraphs} paragraphs in "
        f"{build_seconds * 1000:.1f} ms; query {query_seconds * 1000:.2f} ms"
    )
    if not results:
        print("No matching paragraphs found.")
        return 0

    for result in results:
        judgment_date = result.judgment_date or "unknown date"
        case_number = result.case_number or "unknown case number"
        court = result.court or "unknown court"
        page = result.page_number if result.page_number is not None else "unknown"
        snippet = " ".join(result.text.split())
        print(f"{result.rank}. score={result.score:.6f}")
        print(f"   {result.title} | {case_number} | {court} | {judgment_date}")
        print(
            f"   paragraph={result.paragraph_number} page={page} "
            f"uid={result.paragraph_uid}"
        )
        print(f"   {snippet}")
        if result.rank != len(results):
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
