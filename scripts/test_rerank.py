#!/usr/bin/env python3
"""Rerank hybrid legal-paragraph candidates with a cross-encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import (  # noqa: E402
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    get_settings,
)
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.embeddings import SentenceTransformerEmbeddingProvider  # noqa: E402
from legal_rag.retrieval import (  # noqa: E402
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_RRF_K,
    BM25ParagraphRetriever,
    CrossEncoderReranker,
    DenseParagraphRetriever,
    HybridParagraphRetriever,
    SentenceTransformerCrossEncoderScorer,
    build_retrieval_filters,
)
from legal_rag.schema_migrations import upgrade_database  # noqa: E402
from legal_rag.vector import QdrantParagraphIndex  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank hybrid legal-paragraph candidates with a cross-encoder.",
    )
    parser.add_argument("query", nargs="+", help="Natural-language search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Final reranked matches (default: {DEFAULT_RERANKER_TOP_K})",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help=f"Hybrid candidates to rerank (default: {DEFAULT_RERANKER_CANDIDATE_K})",
    )
    parser.add_argument(
        "--bm25-depth",
        type=int,
        default=DEFAULT_CANDIDATE_DEPTH,
        help="BM25 candidates requested before fusion (default: %(default)s)",
    )
    parser.add_argument(
        "--dense-depth",
        type=int,
        default=DEFAULT_CANDIDATE_DEPTH,
        help="Dense candidates requested before fusion (default: %(default)s)",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF rank constant (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Cross-encoder inference batch size (default: {DEFAULT_RERANKER_BATCH_SIZE})",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL (defaults to DATABASE_URL)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection (defaults to QDRANT_COLLECTION)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Dense embedding model (defaults to EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help=f"Cross-encoder model (default: {DEFAULT_RERANKER_MODEL})",
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


def _configured_dimension(settings: Any) -> int | None:
    value = getattr(settings, "embedding_dimension", None)
    return int(value) if value is not None else None


def _qdrant_api_key(settings: Any) -> str | None:
    value = getattr(settings, "qdrant_api_key", None)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def _rank(value: int | None) -> str:
    return "-" if value is None else str(value)


def _snippet(text: str, limit: int = 360) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _positive(parser: argparse.ArgumentParser, option: str, value: int) -> int:
    if value <= 0:
        parser.error(f"{option} must be positive")
    return value


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip()
    if not query:
        parser.error("query must not be empty")

    settings = get_settings()
    top_k = _positive(
        parser,
        "--top-k",
        settings.reranker_top_k if args.top_k is None else args.top_k,
    )
    candidate_k = _positive(
        parser,
        "--candidate-k",
        settings.reranker_candidate_k
        if args.candidate_k is None
        else args.candidate_k,
    )
    batch_size = _positive(
        parser,
        "--batch-size",
        settings.reranker_batch_size
        if args.batch_size is None
        else args.batch_size,
    )
    for option, value in (
        ("--bm25-depth", args.bm25_depth),
        ("--dense-depth", args.dense_depth),
        ("--rrf-k", args.rrf_k),
    ):
        _positive(parser, option, value)
    try:
        filters = build_retrieval_filters(
            court=args.court,
            year=args.year,
            case_number=args.case_number,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    database_url = args.database_url or settings.database_url
    upgrade_database(database_url)
    session_factory = get_session_factory(get_engine(database_url))

    build_started = perf_counter()
    with session_factory() as session:
        bm25_retriever = BM25ParagraphRetriever.from_session(session)
    build_seconds = perf_counter() - build_started

    provider = SentenceTransformerEmbeddingProvider(
        args.embedding_model or settings.embedding_model,
        expected_dimension=_configured_dimension(settings),
    )
    paragraph_index = QdrantParagraphIndex(
        url=str(settings.qdrant_url),
        api_key=_qdrant_api_key(settings),
        collection_name=args.collection or settings.qdrant_collection,
    )
    paragraph_index.validate_collection(provider.dimension)
    hybrid_retriever = HybridParagraphRetriever(
        bm25_retriever,
        DenseParagraphRetriever(paragraph_index, provider),
        bm25_candidate_depth=args.bm25_depth,
        dense_candidate_depth=args.dense_depth,
        rrf_k=args.rrf_k,
    )
    scorer = SentenceTransformerCrossEncoderScorer(
        args.reranker_model or settings.reranker_model,
    )
    reranker = CrossEncoderReranker(
        hybrid_retriever,
        scorer,
        candidate_k=candidate_k,
        batch_size=batch_size,
    )
    results, diagnostics = reranker.search_with_diagnostics(
        query,
        top_k=top_k,
        filters=filters,
    )

    if filters is not None:
        print(
            f"Filters: court={filters.court or '-'} year={filters.year or '-'} "
            f"case_number={filters.case_number or '-'}"
        )
    print(
        f"BM25 indexed {bm25_retriever.indexed_paragraphs} paragraphs in "
        f"{build_seconds * 1000:.1f} ms; hybrid candidates="
        f"{diagnostics.hybrid_candidates} unique={diagnostics.unique_candidates} "
        f"hybrid={diagnostics.hybrid_seconds * 1000:.1f} ms; "
        f"cross-encoder load={diagnostics.model_load_seconds * 1000:.1f} ms "
        f"inference={diagnostics.inference_seconds * 1000:.1f} ms; "
        f"total={diagnostics.total_seconds * 1000:.1f} ms"
    )
    if not results:
        print("No matching paragraphs found.")
        return 0

    for result in results:
        judgment_date = result.judgment_date or "unknown date"
        case_number = result.case_number or "unknown case number"
        court = result.court or "unknown court"
        page = result.page_number if result.page_number is not None else "unknown"
        print(
            f"{result.reranked_rank}. cross_encoder="
            f"{result.cross_encoder_score:.6f} hybrid_rank={result.hybrid_rank} "
            f"rrf={result.rrf_score:.8f} "
            f"bm25(rank={_rank(result.bm25_rank)}, score={_score(result.bm25_score)}) "
            f"dense(rank={_rank(result.dense_rank)}, score={_score(result.dense_score)})"
        )
        print(f"   {result.title} | {case_number} | {court} | {judgment_date}")
        print(
            f"   paragraph={result.paragraph_number} page={page} "
            f"uid={result.paragraph_uid}"
        )
        print(f"   {_snippet(result.text)}")
        if result.reranked_rank != len(results):
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
