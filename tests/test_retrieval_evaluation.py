from __future__ import annotations

from pathlib import Path

import pytest

from legal_rag.evaluation import (
    EvaluationArtifactError,
    EvaluationSystem,
    GoldParagraphLabel,
    GoldQuery,
    aggregate_records,
    build_diagnostics,
    evaluate_system,
    validate_evaluation_artifacts,
    validate_evaluation_data,
    write_evaluation_artifacts,
)
from legal_rag.retrieval import ParagraphSearchResult


def _query(query_id: str, gold_uid: str) -> GoldQuery:
    return GoldQuery(
        query_id=query_id,
        query=f"Question {query_id}",
        query_type="legal_principle",
        difficulty="easy",
        notes="Synthetic evaluation fixture.",
        relevant_paragraphs=(
            GoldParagraphLabel(
                paragraph_uid=gold_uid,
                relevance=3,
                case_name="Synthetic Case",
                case_number=None,
                paragraph_number=1,
                page_number=1,
                reason="Direct synthetic answer.",
            ),
        ),
    )


def _result(uid: str) -> ParagraphSearchResult:
    return ParagraphSearchResult(
        paragraph_uid=uid,
        text="Synthetic paragraph",
        case_id=1,
        title="Synthetic Case",
        case_number=None,
        court=None,
        judgment_date=None,
        source_url=None,
        paragraph_number=1,
        page_number=1,
        score=1.0,
        rank=1,
    )


def _complete_records():
    queries = (_query("Q001", "gold-1"), _query("Q002", "gold-2"))
    records = []
    for system in ("bm25", "dense", "hybrid_rrf", "hybrid_reranker"):
        system_records, _ = evaluate_system(
            queries,
            EvaluationSystem(system, lambda query: ([_result("gold-1" if "Q001" in query else "miss")], {})),
        )
        records.extend(system_records)
    aggregate = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "configuration": {
            "corpus_paragraph_count": 2,
            "gold_query_count": 2,
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense": {"model": "fake"},
            "hybrid_rrf": {
                "bm25_candidate_depth": 50,
                "dense_candidate_depth": 50,
                "rrf_k": 60,
            },
            "hybrid_reranker": {"model": "fake", "candidate_k": 50},
            "final_top_k": 10,
        },
        "metrics": aggregate_records(records),
        "diagnostics": build_diagnostics(records),
        "runtime_seconds": {
            "bm25": 0.0,
            "dense": 0.0,
            "hybrid_rrf": 0.0,
            "hybrid_reranker": 0.0,
            "cross_encoder_cold_load": 0.0,
            "evaluation_total": 0.0,
            "benchmark_total": 0.0,
        },
    }
    return queries, records, aggregate


def test_complete_artifacts_round_trip_and_recompute(tmp_path: Path) -> None:
    queries, records, aggregate = _complete_records()
    query_ids = [query.query_id for query in queries]

    validate_evaluation_data(aggregate, records, expected_query_ids=query_ids)
    metrics_path, per_query_path, report_path = write_evaluation_artifacts(
        tmp_path,
        aggregate,
        records,
        expected_query_ids=query_ids,
    )
    validate_evaluation_artifacts(
        metrics_path,
        per_query_path,
        expected_query_ids=query_ids,
    )

    assert metrics_path.exists()
    assert len(per_query_path.read_text(encoding="utf-8").splitlines()) == 8
    assert "| BM25 |" in report_path.read_text(encoding="utf-8")


def test_artifact_validation_rejects_missing_system_record() -> None:
    queries, records, aggregate = _complete_records()

    with pytest.raises(EvaluationArtifactError, match="incomplete query-system coverage"):
        validate_evaluation_data(
            aggregate,
            records[:-1],
            expected_query_ids=[query.query_id for query in queries],
        )


def test_artifact_validation_rejects_duplicate_uid_in_ranking() -> None:
    queries, records, aggregate = _complete_records()
    records[0]["retrieved"].append(dict(records[0]["retrieved"][0], rank=2))

    with pytest.raises(EvaluationArtifactError, match="duplicate retrieved UIDs"):
        validate_evaluation_data(
            aggregate,
            records,
            expected_query_ids=[query.query_id for query in queries],
        )
