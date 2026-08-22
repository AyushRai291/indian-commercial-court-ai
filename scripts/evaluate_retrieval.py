#!/usr/bin/env python3
"""Evaluate the four frozen retrieval configurations against gold queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from sqlalchemy import func, select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import get_settings  # noqa: E402
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.embeddings import SentenceTransformerEmbeddingProvider  # noqa: E402
from legal_rag.evaluation import (  # noqa: E402
    EvaluationSystem,
    aggregate_records,
    build_diagnostics,
    evaluate_system,
    load_gold_queries,
    validate_evaluation_artifacts,
    validate_gold_queries,
    write_evaluation_artifacts,
)
from legal_rag.models import Paragraph  # noqa: E402
from legal_rag.retrieval import (  # noqa: E402
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_RRF_K,
    BM25ParagraphRetriever,
    CrossEncoderReranker,
    DenseParagraphRetriever,
    HybridParagraphRetriever,
    SentenceTransformerCrossEncoderScorer,
)
from legal_rag.schema_migrations import upgrade_database  # noqa: E402
from legal_rag.vector import QdrantParagraphIndex  # noqa: E402


EXPECTED_QUERIES = 40
EXPECTED_PARAGRAPHS = 18_822
FINAL_TOP_K = 10
BM25_K1 = 1.5
BM25_B = 0.75


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure BM25, dense, RRF, and reranked retrieval on gold queries."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/gold_queries.jsonl"),
        help="Frozen gold JSONL path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/evaluation/results"),
        help="Tracked artifact directory (default: %(default)s)",
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
        "--validate-only",
        action="store_true",
        help="Validate existing result artifacts without running retrieval",
    )
    return parser


def _qdrant_api_key(settings: Any) -> str | None:
    value = getattr(settings, "qdrant_api_key", None)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _qdrant_point_count(index: QdrantParagraphIndex) -> int:
    collection = index.client.get_collection(index.collection_name)
    count = getattr(collection, "points_count", None)
    if count is None:
        raise RuntimeError("Qdrant did not report a collection point count")
    return int(count)


def _metric_definitions() -> dict[str, str]:
    return {
        "relevance_identity": "exact paragraph_uid",
        "binary_relevance": "every gold grade greater than zero is relevant",
        "recall_at_5": "unique relevant UIDs in top 5 / unique gold relevant UIDs; macro mean",
        "recall_at_10": "unique relevant UIDs in top 10 / unique gold relevant UIDs; macro mean",
        "mrr": "reciprocal rank of first relevant UID within returned top 10; macro mean",
        "ndcg_at_10": "gain=2^rel-1, discount=log2(rank+1), ideal grades descending; macro mean",
    }


def run_benchmark(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Construct every expensive component once and execute the frozen benchmark."""

    benchmark_started = perf_counter()
    settings = get_settings()
    database_url = args.database_url or settings.database_url
    collection_name = args.collection or settings.qdrant_collection
    queries = load_gold_queries(args.dataset)
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(
            f"expected exactly {EXPECTED_QUERIES} gold queries, found {len(queries)}"
        )

    _print_progress("Validating frozen gold set and corpus schema...")
    upgrade_database(database_url)
    session_factory = get_session_factory(get_engine(database_url))
    bm25_build_started = perf_counter()
    with session_factory() as session:
        validate_gold_queries(queries, session, expected_count=EXPECTED_QUERIES)
        postgres_count = int(session.scalar(select(func.count()).select_from(Paragraph)) or 0)
        bm25 = BM25ParagraphRetriever.from_session(
            session,
            k1=BM25_K1,
            b=BM25_B,
        )
    bm25_build_seconds = perf_counter() - bm25_build_started
    if postgres_count != EXPECTED_PARAGRAPHS:
        raise RuntimeError(
            f"expected {EXPECTED_PARAGRAPHS} PostgreSQL paragraphs, found {postgres_count}"
        )
    if bm25.indexed_paragraphs != postgres_count:
        raise RuntimeError("BM25 indexed paragraph count does not match PostgreSQL")

    _print_progress("Loading the shared dense model and validating Qdrant...")
    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        expected_dimension=settings.embedding_dimension,
    )
    paragraph_index = QdrantParagraphIndex(
        url=str(settings.qdrant_url),
        api_key=_qdrant_api_key(settings),
        collection_name=collection_name,
    )
    paragraph_index.validate_collection(provider.dimension)
    qdrant_count = _qdrant_point_count(paragraph_index)
    if qdrant_count != postgres_count:
        raise RuntimeError(
            f"Qdrant has {qdrant_count} points but PostgreSQL has {postgres_count} paragraphs"
        )

    dense = DenseParagraphRetriever(paragraph_index, provider)
    hybrid = HybridParagraphRetriever(
        bm25,
        dense,
        bm25_candidate_depth=DEFAULT_CANDIDATE_DEPTH,
        dense_candidate_depth=DEFAULT_CANDIDATE_DEPTH,
        rrf_k=DEFAULT_RRF_K,
    )
    scorer = SentenceTransformerCrossEncoderScorer(settings.reranker_model)
    reranker = CrossEncoderReranker(
        hybrid,
        scorer,
        candidate_k=50,
        batch_size=settings.reranker_batch_size,
    )
    setup_seconds = perf_counter() - benchmark_started

    def bm25_search(query: str):
        results = bm25.search(query, top_k=FINAL_TOP_K)
        return results, {"returned_results": len(results)}

    def dense_search(query: str):
        results = dense.search(query, top_k=FINAL_TOP_K)
        return results, {"returned_results": len(results)}

    def hybrid_search(query: str):
        results, diagnostics = hybrid.search_with_diagnostics(
            query,
            top_k=FINAL_TOP_K,
            bm25_candidate_depth=DEFAULT_CANDIDATE_DEPTH,
            dense_candidate_depth=DEFAULT_CANDIDATE_DEPTH,
            rrf_k=DEFAULT_RRF_K,
        )
        return results, asdict(diagnostics)

    def reranker_search(query: str):
        results, diagnostics = reranker.search_with_diagnostics(
            query,
            top_k=FINAL_TOP_K,
            candidate_k=50,
            bm25_candidate_depth=DEFAULT_CANDIDATE_DEPTH,
            dense_candidate_depth=DEFAULT_CANDIDATE_DEPTH,
            rrf_k=DEFAULT_RRF_K,
            batch_size=settings.reranker_batch_size,
        )
        return results, asdict(diagnostics)

    systems = (
        EvaluationSystem("bm25", bm25_search),
        EvaluationSystem("dense", dense_search),
        EvaluationSystem("hybrid_rrf", hybrid_search),
        EvaluationSystem("hybrid_reranker", reranker_search),
    )
    records: list[dict[str, Any]] = []
    system_runtimes: dict[str, float] = {}
    evaluation_started = perf_counter()
    for system in systems:
        _print_progress(f"Evaluating {system.name} over {len(queries)} queries...")
        system_records, runtime = evaluate_system(
            queries,
            system,
            progress=_print_progress,
        )
        records.extend(system_records)
        system_runtimes[system.name] = runtime
    evaluation_seconds = perf_counter() - evaluation_started
    cross_encoder_cold_load = sum(
        float(record["diagnostics"].get("model_load_seconds", 0.0))
        for record in records
        if record["system"] == "hybrid_reranker"
    )

    configuration = {
        "corpus_paragraph_count": postgres_count,
        "qdrant_point_count": qdrant_count,
        "gold_query_count": len(queries),
        "gold_dataset": args.dataset.as_posix(),
        "gold_dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "bm25": {"k1": BM25_K1, "b": BM25_B, "top_k": FINAL_TOP_K},
        "dense": {
            "model": settings.embedding_model,
            "dimension": provider.dimension,
            "collection": collection_name,
            "top_k": FINAL_TOP_K,
        },
        "hybrid_rrf": {
            "bm25_candidate_depth": DEFAULT_CANDIDATE_DEPTH,
            "dense_candidate_depth": DEFAULT_CANDIDATE_DEPTH,
            "rrf_k": DEFAULT_RRF_K,
            "top_k": FINAL_TOP_K,
        },
        "hybrid_reranker": {
            "model": settings.reranker_model,
            "candidate_k": 50,
            "batch_size": settings.reranker_batch_size,
            "top_k": FINAL_TOP_K,
        },
        "final_top_k": FINAL_TOP_K,
        "metadata_filters": None,
        "metric_definitions": _metric_definitions(),
    }
    aggregate_document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "metrics": aggregate_records(records),
        "diagnostics": build_diagnostics(records),
        "runtime_seconds": {
            **system_runtimes,
            "bm25_index_build": bm25_build_seconds,
            "cross_encoder_cold_load": cross_encoder_cold_load,
            "setup": setup_seconds,
            "evaluation_total": evaluation_seconds,
            "benchmark_total": perf_counter() - benchmark_started,
        },
    }
    paths = write_evaluation_artifacts(
        args.output_directory,
        aggregate_document,
        records,
        expected_query_ids=[query.query_id for query in queries],
    )
    _print_progress(json.dumps(aggregate_document["metrics"], indent=2))
    return paths


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    queries = load_gold_queries(args.dataset)
    expected_ids = [query.query_id for query in queries]
    metrics_path = args.output_directory / "retrieval_metrics.json"
    per_query_path = args.output_directory / "retrieval_per_query.jsonl"
    if args.validate_only:
        validate_evaluation_artifacts(
            metrics_path,
            per_query_path,
            expected_query_ids=expected_ids,
        )
        print(
            f"Validated {len(expected_ids) * 4} query-system records in "
            f"{args.output_directory}"
        )
        return 0

    paths = run_benchmark(args)
    print("Evaluation artifacts written:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
