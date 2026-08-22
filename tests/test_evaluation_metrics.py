from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from legal_rag.evaluation.metrics import (
    RetrievalMetrics,
    evaluate_ranking,
    macro_average,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_uses_unique_exact_uids_and_top_k_positions() -> None:
    gold = {"a": 3, "b": 2, "c": 1}
    ranking = ["a", "a", "same-case-but-not-gold", "b", "x", "c"]

    assert recall_at_k(ranking, gold, k=5) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, gold, k=10) == 1.0


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [
        (["relevant", "other"], 1.0),
        (["a", "b", "c", "relevant"], 0.25),
        (["a", "b", "c", "d"], 0.0),
        (["a"] * 10 + ["relevant"], 0.0),
    ],
)
def test_mrr_is_bounded_to_returned_top_ten(
    ranking: list[str], expected: float
) -> None:
    assert reciprocal_rank(ranking, {"relevant": 3}, max_rank=10) == expected


def test_ndcg_uses_exponential_gain_and_log2_discount() -> None:
    gold = {"grade-three": 3, "grade-two": 2}
    ranking = ["grade-two", "grade-three"]
    expected_dcg = 3.0 + 7.0 / math.log2(3)
    expected_ideal = 7.0 + 3.0 / math.log2(3)

    assert ndcg_at_k(ranking, gold, k=10) == pytest.approx(
        expected_dcg / expected_ideal
    )


def test_perfect_ranking_has_full_metrics_when_gold_fits_cutoffs() -> None:
    metrics = evaluate_ranking(["a", "b", "c"], {"a": 3, "b": 2, "c": 1})

    assert metrics == RetrievalMetrics(1.0, 1.0, 1.0, 1.0)


def test_perfect_ranking_recall_at_five_respects_more_than_five_gold_labels() -> None:
    gold = {f"p{index}": 1 for index in range(1, 8)}
    metrics = evaluate_ranking(list(gold), gold)

    assert metrics.recall_at_5 == pytest.approx(5 / 7)
    assert metrics.recall_at_10 == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.ndcg_at_10 == 1.0


def test_empty_and_irrelevant_rankings_score_zero() -> None:
    gold = {"gold": 3}

    assert evaluate_ranking([], gold) == RetrievalMetrics(0.0, 0.0, 0.0, 0.0)
    assert evaluate_ranking(["other", "same-case-different-uid"], gold) == (
        RetrievalMetrics(0.0, 0.0, 0.0, 0.0)
    )


@dataclass(frozen=True)
class _Result:
    paragraph_uid: str


def test_metric_functions_accept_retrieval_results_by_exact_uid() -> None:
    results = [_Result("not-gold"), _Result(" gold "), _Result("gold")]

    assert reciprocal_rank(results, {"gold": 1}) == pytest.approx(1 / 3)
    assert recall_at_k(results, {"gold": 1}, k=2) == 0.0
    assert recall_at_k(results, {"gold": 1}, k=3) == 1.0


def test_macro_average_is_unweighted_across_queries() -> None:
    average = macro_average(
        [
            RetrievalMetrics(1.0, 1.0, 1.0, 1.0),
            RetrievalMetrics(0.0, 0.5, 0.25, 0.2),
        ]
    )

    assert average.recall_at_5 == 0.5
    assert average.recall_at_10 == 0.75
    assert average.reciprocal_rank == 0.625
    assert average.ndcg_at_10 == 0.6
