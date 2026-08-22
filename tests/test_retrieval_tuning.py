from __future__ import annotations

from copy import deepcopy

import pytest

from legal_rag.config import DEFAULT_RERANKER_CANDIDATE_K
from legal_rag.evaluation import (
    GoldParagraphLabel,
    GoldQuery,
    TuningArtifactError,
    aggregate_query_categories,
    candidate_recall,
    classify_reranker_failure,
    generate_hybrid_experiment_configs,
    generate_reranker_experiment_configs,
    select_best_experiment,
    validate_tuning_data,
)
from legal_rag.retrieval import DEFAULT_RRF_K


def _query(
    query_id: str,
    *,
    query_type: str = "legal_principle",
    difficulty: str = "easy",
    case_name: str = "Case A",
) -> GoldQuery:
    return GoldQuery(
        query_id=query_id,
        query=f"Question {query_id}",
        query_type=query_type,
        difficulty=difficulty,
        notes="Synthetic tuning fixture.",
        relevant_paragraphs=(
            GoldParagraphLabel(
                paragraph_uid=f"gold-{query_id}",
                relevance=3,
                case_name=case_name,
                case_number=None,
                paragraph_number=1,
                page_number=1,
                reason="Synthetic direct evidence.",
            ),
        ),
    )


def _metrics(value: float) -> dict[str, float]:
    return {
        "recall_at_5": value,
        "recall_at_10": value,
        "mrr": value,
        "ndcg_at_10": value,
    }


def test_experiment_grid_is_fixed_unique_and_complete() -> None:
    hybrid = generate_hybrid_experiment_configs()

    assert len(hybrid) == 10
    assert len({item.experiment_id for item in hybrid}) == 10
    assert {
        item.rrf_k
        for item in hybrid
        if item.bm25_candidate_depth == item.dense_candidate_depth == 50
    } == {10, 20, 40, 60, 80, 100}
    assert {
        (item.bm25_candidate_depth, item.dense_candidate_depth)
        for item in hybrid
        if item.rrf_k == 60
    } == {(30, 30), (40, 40), (50, 50), (50, 30), (30, 50)}

    reranker = generate_reranker_experiment_configs("selected")
    assert [item.candidate_k for item in reranker] == [30, 40, 50]
    assert len({item.experiment_id for item in reranker}) == 3


def test_measured_retrieval_defaults_are_centralized() -> None:
    assert DEFAULT_RRF_K == 10
    assert DEFAULT_RERANKER_CANDIDATE_K == 30


def test_candidate_recall_uses_unique_union_at_requested_depth() -> None:
    bm25 = ["a", "a", "x", "b"]
    dense = ["y", "c", "a", "z"]

    assert candidate_recall([bm25], {"a", "b", "c"}, depth=3) == pytest.approx(1 / 3)
    assert candidate_recall([bm25, dense], {"a", "b", "c"}, depth=3) == pytest.approx(2 / 3)
    assert candidate_recall([bm25, dense], {"a", "b", "c"}, depth=4) == 1.0


@pytest.mark.parametrize(
    ("candidates", "ranking", "expected"),
    [
        (["gold"], ["gold"], "success"),
        (["other"], ["other"], "missing_candidates"),
        (["gold", "other"], ["other"], "bad_reranking"),
    ],
)
def test_reranker_failure_classification(
    candidates: list[str], ranking: list[str], expected: str
) -> None:
    assert classify_reranker_failure(candidates, ranking, {"gold"}) == expected


def test_query_category_aggregation_is_macro_and_uses_frozen_facets() -> None:
    queries = (
        _query("Q001"),
        _query("Q002", query_type="procedural", difficulty="hard", case_name="Case B"),
    )
    records = []
    for system in ("bm25", "dense", "hybrid_rrf", "hybrid_reranker"):
        for query, value in zip(queries, (1.0, 0.0)):
            records.append(
                {
                    "query_id": query.query_id,
                    "system": system,
                    "recall_at_5": value,
                    "recall_at_10": value,
                    "reciprocal_rank": value,
                    "ndcg_at_10": value,
                }
            )

    breakdown = aggregate_query_categories(queries, records)

    assert breakdown["difficulty"]["easy"]["systems"]["bm25"]["ndcg_at_10"] == 1.0
    assert breakdown["difficulty"]["hard"]["systems"]["dense"]["ndcg_at_10"] == 0.0
    assert breakdown["judgment_scope"]["single_judgment"]["query_count"] == 2


def test_selection_rule_is_deterministic_and_uses_declared_priority() -> None:
    experiments = [
        {
            "experiment_id": "slower",
            "configuration": {"bm25_candidate_depth": 50, "dense_candidate_depth": 50},
            "metrics": {**_metrics(0.5), "recall_at_10": 0.6},
            "runtime_seconds": 2.0,
        },
        {
            "experiment_id": "winner",
            "configuration": {"bm25_candidate_depth": 30, "dense_candidate_depth": 30},
            "metrics": {**_metrics(0.5), "recall_at_10": 0.6},
            "runtime_seconds": 1.0,
        },
        {
            "experiment_id": "lower_ndcg",
            "configuration": {"bm25_candidate_depth": 10, "dense_candidate_depth": 10},
            "metrics": {**_metrics(0.9), "ndcg_at_10": 0.49},
            "runtime_seconds": 0.1,
        },
    ]

    assert select_best_experiment(experiments)["experiment_id"] == "winner"


def _valid_artifacts():
    baseline = {
        "bm25": _metrics(1.0),
        "dense": _metrics(0.0),
        "hybrid_rrf": _metrics(1.0),
        "hybrid_reranker": _metrics(1.0),
    }
    experiments = [
        {
            "experiment_id": "hybrid",
            "family": "hybrid",
            "configuration": {},
            "metrics": _metrics(1.0),
            "runtime_seconds": 0.1,
        },
        {
            "experiment_id": "reranker",
            "family": "reranker",
            "configuration": {},
            "metrics": _metrics(1.0),
            "runtime_seconds": 0.2,
        },
    ]
    ablation = {
        "gold_dataset_sha256": "frozen",
        "day11_baseline": baseline,
        "experiments": experiments,
        "selected": {
            "hybrid_experiment_id": "hybrid",
            "reranker_experiment_id": "reranker",
        },
    }
    analysis = {
        "queries": [
            {
                "query_id": "Q001",
                "gold_paragraphs": [{"paragraph_uid": "gold-1", "relevance": 3}],
                "selected_hybrid_top_10": ["gold-1"],
                "selected_reranker_top_10": ["gold-1"],
            },
            {
                "query_id": "Q002",
                "gold_paragraphs": [{"paragraph_uid": "gold-2", "relevance": 2}],
                "selected_hybrid_top_10": ["gold-2"],
                "selected_reranker_top_10": ["gold-2"],
            },
        ]
    }
    return baseline, ablation, analysis


def test_artifact_validation_recomputes_final_metrics_and_baseline() -> None:
    baseline, ablation, analysis = _valid_artifacts()

    validate_tuning_data(
        ablation,
        analysis,
        day11_metrics=baseline,
        expected_query_ids=["Q001", "Q002"],
        expected_gold_sha256="frozen",
    )


def test_artifact_validation_rejects_duplicate_experiment_ids() -> None:
    baseline, ablation, analysis = _valid_artifacts()
    broken = deepcopy(ablation)
    broken["experiments"][1]["experiment_id"] = "hybrid"

    with pytest.raises(TuningArtifactError, match="duplicate experiment IDs"):
        validate_tuning_data(
            broken,
            analysis,
            day11_metrics=baseline,
            expected_query_ids=["Q001", "Q002"],
            expected_gold_sha256="frozen",
        )


def test_artifact_validation_rejects_duplicate_configurations() -> None:
    baseline, ablation, analysis = _valid_artifacts()
    broken = deepcopy(ablation)
    broken["experiments"][1]["family"] = "hybrid"

    with pytest.raises(TuningArtifactError, match="duplicate experiment configurations"):
        validate_tuning_data(
            broken,
            analysis,
            day11_metrics=baseline,
            expected_query_ids=["Q001", "Q002"],
            expected_gold_sha256="frozen",
        )
