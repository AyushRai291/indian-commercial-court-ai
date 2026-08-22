"""Pure paragraph-level retrieval metrics for graded gold labels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, Union


class ParagraphIdentity(Protocol):
    """Structural identity exposed by every repository retrieval result."""

    paragraph_uid: str


RankedIdentity = Union[str, ParagraphIdentity]


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Per-query or macro-averaged retrieval measurements."""

    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_10: float

    def to_dict(self) -> dict[str, float]:
        """Return a stable JSON representation."""

        return asdict(self)


def _uid(value: RankedIdentity) -> str:
    paragraph_uid = value if isinstance(value, str) else value.paragraph_uid
    if not isinstance(paragraph_uid, str) or not paragraph_uid.strip():
        raise ValueError("paragraph_uid must be a non-empty string")
    return paragraph_uid


def unique_ranked_uids(
    ranking: Sequence[RankedIdentity],
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Preserve first rank while removing duplicate paragraph identities."""

    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit <= 0:
            raise ValueError("limit must be positive")
    ordered: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(ranking, start=1):
        if limit is not None and position > limit:
            break
        paragraph_uid = _uid(value)
        if paragraph_uid in seen:
            continue
        seen.add(paragraph_uid)
        ordered.append(paragraph_uid)
    return tuple(ordered)


def _gold_relevance(relevance: Mapping[str, int]) -> dict[str, int]:
    if not relevance:
        raise ValueError("gold relevance must contain at least one paragraph")
    validated: dict[str, int] = {}
    for raw_uid, grade in relevance.items():
        paragraph_uid = _uid(raw_uid)
        if isinstance(grade, bool) or not isinstance(grade, int):
            raise TypeError("relevance grades must be integers")
        if grade <= 0:
            raise ValueError("gold relevance grades must be positive")
        if paragraph_uid in validated:
            raise ValueError(f"duplicate gold paragraph_uid: {paragraph_uid}")
        validated[paragraph_uid] = grade
    return validated


def recall_at_k(
    ranking: Sequence[RankedIdentity],
    relevance: Mapping[str, int],
    *,
    k: int,
) -> float:
    """Return binary paragraph recall at ``k`` over all positive gold labels."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    gold = _gold_relevance(relevance)
    retrieved = set(unique_ranked_uids(ranking, limit=k))
    return len(retrieved.intersection(gold)) / len(gold)


def reciprocal_rank(
    ranking: Sequence[RankedIdentity],
    relevance: Mapping[str, int],
    *,
    max_rank: int = 10,
) -> float:
    """Return reciprocal rank of the first positive gold hit within ``max_rank``."""

    gold = _gold_relevance(relevance)
    for rank, paragraph_uid in enumerate(
        unique_ranked_uids(ranking, limit=max_rank), start=1
    ):
        if paragraph_uid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranking: Sequence[RankedIdentity],
    relevance: Mapping[str, int],
    *,
    k: int = 10,
) -> float:
    """Return graded nDCG using exponential gain and log-base-two discount."""

    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    gold = _gold_relevance(relevance)
    ranked_uids = unique_ranked_uids(ranking, limit=k)
    dcg = sum(
        ((2**gold.get(paragraph_uid, 0)) - 1) / math.log2(rank + 1)
        for rank, paragraph_uid in enumerate(ranked_uids, start=1)
    )
    ideal_grades = sorted(gold.values(), reverse=True)[:k]
    ideal_dcg = sum(
        ((2**grade) - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal_dcg


def evaluate_ranking(
    ranking: Sequence[RankedIdentity],
    relevance: Mapping[str, int],
) -> RetrievalMetrics:
    """Calculate the frozen Day 11 metrics for one top-10 ranking."""

    return RetrievalMetrics(
        recall_at_5=recall_at_k(ranking, relevance, k=5),
        recall_at_10=recall_at_k(ranking, relevance, k=10),
        reciprocal_rank=reciprocal_rank(ranking, relevance, max_rank=10),
        ndcg_at_10=ndcg_at_k(ranking, relevance, k=10),
    )


def macro_average(metrics: Sequence[RetrievalMetrics]) -> RetrievalMetrics:
    """Return an unweighted arithmetic mean across query-level metrics."""

    if not metrics:
        raise ValueError("metrics must contain at least one query")
    count = len(metrics)
    return RetrievalMetrics(
        recall_at_5=sum(item.recall_at_5 for item in metrics) / count,
        recall_at_10=sum(item.recall_at_10 for item in metrics) / count,
        reciprocal_rank=sum(item.reciprocal_rank for item in metrics) / count,
        ndcg_at_10=sum(item.ndcg_at_10 for item in metrics) / count,
    )
