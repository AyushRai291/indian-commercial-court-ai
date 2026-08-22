"""Pure helpers for controlled retrieval ablations and error analysis."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final

from legal_rag.evaluation.gold import GoldQuery
from legal_rag.evaluation.metrics import RetrievalMetrics, evaluate_ranking, macro_average


RRF_VALUES: Final[tuple[int, ...]] = (10, 20, 40, 60, 80, 100)
CANDIDATE_DEPTHS: Final[tuple[tuple[int, int], ...]] = (
    (30, 30),
    (40, 40),
    (50, 50),
    (50, 30),
    (30, 50),
)
RERANKER_DEPTHS: Final[tuple[int, ...]] = (30, 40, 50)
BASELINE_METRIC_KEYS: Final[tuple[str, ...]] = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
)


class TuningArtifactError(ValueError):
    """Raised when a Day 12 experiment or artifact is incomplete."""


@dataclass(frozen=True, slots=True)
class HybridExperimentConfig:
    """One controlled RRF configuration over cached native rankings."""

    experiment_id: str
    bm25_candidate_depth: int
    dense_candidate_depth: int
    rrf_k: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RerankerExperimentConfig:
    """One cross-encoder depth over a selected hybrid configuration."""

    experiment_id: str
    hybrid_experiment_id: str
    candidate_k: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_hybrid_experiment_configs() -> tuple[HybridExperimentConfig, ...]:
    """Return the fixed RRF and candidate-depth grid without duplicates."""

    raw = [(50, 50, value) for value in RRF_VALUES]
    raw.extend((bm25_depth, dense_depth, 60) for bm25_depth, dense_depth in CANDIDATE_DEPTHS)
    unique: dict[tuple[int, int, int], HybridExperimentConfig] = {}
    for bm25_depth, dense_depth, rrf_k in raw:
        key = (bm25_depth, dense_depth, rrf_k)
        unique.setdefault(
            key,
            HybridExperimentConfig(
                experiment_id=f"rrf_b{bm25_depth}_d{dense_depth}_k{rrf_k}",
                bm25_candidate_depth=bm25_depth,
                dense_candidate_depth=dense_depth,
                rrf_k=rrf_k,
            ),
        )
    return tuple(unique.values())


def generate_reranker_experiment_configs(
    hybrid_experiment_id: str,
) -> tuple[RerankerExperimentConfig, ...]:
    """Return the fixed 30/40/50 candidate-depth reranker experiments."""

    if not isinstance(hybrid_experiment_id, str) or not hybrid_experiment_id.strip():
        raise ValueError("hybrid_experiment_id must not be empty")
    return tuple(
        RerankerExperimentConfig(
            experiment_id=f"rerank_{hybrid_experiment_id}_c{depth}",
            hybrid_experiment_id=hybrid_experiment_id,
            candidate_k=depth,
        )
        for depth in RERANKER_DEPTHS
    )


def gold_relevance(query: GoldQuery) -> dict[str, int]:
    """Return one query's exact paragraph UID-to-grade mapping."""

    return {label.paragraph_uid: label.relevance for label in query.relevant_paragraphs}


def metrics_dict(metrics: RetrievalMetrics) -> dict[str, float]:
    """Use the aggregate artifact's explicit MRR field name."""

    return {
        "recall_at_5": metrics.recall_at_5,
        "recall_at_10": metrics.recall_at_10,
        "mrr": metrics.reciprocal_rank,
        "ndcg_at_10": metrics.ndcg_at_10,
    }


def aggregate_rankings(
    queries: Sequence[GoldQuery],
    rankings: Mapping[str, Sequence[str]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Calculate macro and query-level metrics for exact-UID rankings."""

    expected = {query.query_id for query in queries}
    if set(rankings) != expected:
        raise ValueError("rankings must contain every query exactly once")
    per_query: dict[str, dict[str, float]] = {}
    metric_values: list[RetrievalMetrics] = []
    for query in queries:
        metrics = evaluate_ranking(rankings[query.query_id][:10], gold_relevance(query))
        metric_values.append(metrics)
        per_query[query.query_id] = metrics_dict(metrics)
    return metrics_dict(macro_average(metric_values)), per_query


def candidate_recall(
    rankings: Sequence[Sequence[str]],
    relevant_uids: Sequence[str] | set[str],
    *,
    depth: int,
) -> float:
    """Return recall of the unique union of ranked candidate lists at depth."""

    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth <= 0:
        raise ValueError("depth must be positive")
    gold = set(relevant_uids)
    if not gold:
        raise ValueError("relevant_uids must not be empty")
    candidates = {uid for ranking in rankings for uid in ranking[:depth]}
    return len(candidates.intersection(gold)) / len(gold)


def classify_reranker_failure(
    candidate_uids: Sequence[str],
    reranked_top_ten: Sequence[str],
    relevant_uids: Sequence[str] | set[str],
) -> str:
    """Distinguish retrieval success, candidate absence, and ordering failure."""

    gold = set(relevant_uids)
    if not gold:
        raise ValueError("relevant_uids must not be empty")
    if gold.intersection(reranked_top_ten[:10]):
        return "success"
    if gold.intersection(candidate_uids):
        return "bad_reranking"
    return "missing_candidates"


def _record_metrics(record: Mapping[str, Any]) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_5=float(record["recall_at_5"]),
        recall_at_10=float(record["recall_at_10"]),
        reciprocal_rank=float(record.get("mrr", record.get("reciprocal_rank"))),
        ndcg_at_10=float(record["ndcg_at_10"]),
    )


def aggregate_query_categories(
    queries: Sequence[GoldQuery],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Macro-average existing system records across four frozen query facets."""

    queries_by_id = {query.query_id: query for query in queries}
    grouped: dict[str, dict[str, dict[str, list[RetrievalMetrics]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in records:
        query_id = str(record["query_id"])
        query = queries_by_id.get(query_id)
        if query is None:
            raise ValueError(f"unknown query_id in records: {query_id}")
        system = str(record["system"])
        judgments = {
            (label.case_name, label.case_number) for label in query.relevant_paragraphs
        }
        facets = {
            "query_type": query.query_type,
            "difficulty": query.difficulty,
            "gold_paragraph_count": str(len(query.relevant_paragraphs)),
            "judgment_scope": "multi_judgment" if len(judgments) > 1 else "single_judgment",
        }
        metrics = _record_metrics(record)
        for facet, category in facets.items():
            grouped[facet][category][system].append(metrics)

    result: dict[str, Any] = {}
    for facet, categories in grouped.items():
        result[facet] = {}
        for category, systems in sorted(categories.items()):
            query_ids = sorted(
                query.query_id
                for query in queries
                if (
                    query.query_type if facet == "query_type" else
                    query.difficulty if facet == "difficulty" else
                    str(len(query.relevant_paragraphs)) if facet == "gold_paragraph_count" else
                    (
                        "multi_judgment"
                        if len({(label.case_name, label.case_number) for label in query.relevant_paragraphs}) > 1
                        else "single_judgment"
                    )
                ) == category
            )
            result[facet][category] = {
                "query_count": len(query_ids),
                "query_ids": query_ids,
                "systems": {
                    system: metrics_dict(macro_average(values))
                    for system, values in sorted(systems.items())
                },
            }
    return result


def select_best_experiment(
    experiments: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Apply the frozen nDCG, Recall@10, MRR, runtime, simplicity rule."""

    if not experiments:
        raise ValueError("experiments must not be empty")
    ids = [str(experiment.get("experiment_id", "")) for experiment in experiments]
    if any(not experiment_id for experiment_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("experiment IDs must be non-empty and unique")

    def key(experiment: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = experiment["metrics"]
        configuration = experiment["configuration"]
        complexity = (
            int(configuration.get("bm25_candidate_depth", 0))
            + int(configuration.get("dense_candidate_depth", 0))
            + int(configuration.get("candidate_k", 0))
        )
        return (
            -float(metrics["ndcg_at_10"]),
            -float(metrics["recall_at_10"]),
            -float(metrics["mrr"]),
            float(experiment.get("runtime_seconds", 0.0)),
            complexity,
            str(experiment["experiment_id"]),
        )

    return min(experiments, key=key)


def _validate_metrics(metrics: object, *, context: str) -> None:
    if not isinstance(metrics, Mapping):
        raise TuningArtifactError(f"{context} metrics must be an object")
    if set(metrics) != set(BASELINE_METRIC_KEYS):
        raise TuningArtifactError(f"{context} metrics have incorrect fields")
    for field in BASELINE_METRIC_KEYS:
        value = metrics[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TuningArtifactError(f"{context}/{field} must be numeric")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise TuningArtifactError(f"{context}/{field} must be within [0,1]")


def validate_tuning_data(
    ablation: Mapping[str, Any],
    error_analysis: Mapping[str, Any],
    *,
    day11_metrics: Mapping[str, Any],
    expected_query_ids: Sequence[str],
    expected_gold_sha256: str,
) -> None:
    """Fail closed on incomplete experiments, stale baselines, or bad final metrics."""

    if ablation.get("gold_dataset_sha256") != expected_gold_sha256:
        raise TuningArtifactError("gold dataset hash does not match the frozen dataset")
    baseline = ablation.get("day11_baseline")
    if baseline != day11_metrics:
        raise TuningArtifactError("Day 11 baseline metrics changed")
    experiments = ablation.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise TuningArtifactError("experiments must be a non-empty array")
    experiment_ids: list[str] = []
    configuration_fingerprints: list[str] = []
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise TuningArtifactError("experiment must be an object")
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise TuningArtifactError("experiment_id must not be empty")
        experiment_ids.append(experiment_id)
        configuration = experiment.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TuningArtifactError(f"{experiment_id} configuration must be an object")
        configuration_fingerprints.append(
            f"{experiment.get('family', '')}:"
            + json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        )
        _validate_metrics(experiment.get("metrics"), context=experiment_id)
        runtime = experiment.get("runtime_seconds")
        if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)) or runtime < 0:
            raise TuningArtifactError(f"{experiment_id} runtime must be finite and non-negative")
    if len(experiment_ids) != len(set(experiment_ids)):
        raise TuningArtifactError("duplicate experiment IDs")
    if len(configuration_fingerprints) != len(set(configuration_fingerprints)):
        raise TuningArtifactError("duplicate experiment configurations")
    selected = ablation.get("selected")
    if not isinstance(selected, Mapping):
        raise TuningArtifactError("selected configuration is missing")
    for field in ("hybrid_experiment_id", "reranker_experiment_id"):
        if selected.get(field) not in experiment_ids:
            raise TuningArtifactError(f"selected {field} was not tested")

    query_records = error_analysis.get("queries")
    if not isinstance(query_records, list):
        raise TuningArtifactError("error-analysis queries must be an array")
    actual_ids = [record.get("query_id") for record in query_records if isinstance(record, Mapping)]
    if len(actual_ids) != len(query_records) or set(actual_ids) != set(expected_query_ids):
        raise TuningArtifactError("error analysis does not contain all expected queries")
    if len(actual_ids) != len(set(actual_ids)):
        raise TuningArtifactError("error analysis has duplicate query IDs")

    selected_id = str(selected["reranker_experiment_id"])
    selected_experiment = next(
        experiment for experiment in experiments if experiment["experiment_id"] == selected_id
    )
    selected_hybrid_id = str(selected["hybrid_experiment_id"])
    selected_hybrid_experiment = next(
        experiment for experiment in experiments if experiment["experiment_id"] == selected_hybrid_id
    )
    recomputed_hybrid: list[RetrievalMetrics] = []
    recomputed_reranker: list[RetrievalMetrics] = []
    for record in query_records:
        gold = {
            str(label["paragraph_uid"]): int(label["relevance"])
            for label in record["gold_paragraphs"]
        }
        recomputed_hybrid.append(evaluate_ranking(record["selected_hybrid_top_10"], gold))
        recomputed_reranker.append(evaluate_ranking(record["selected_reranker_top_10"], gold))
    for label, actual, experiment in (
        ("hybrid", metrics_dict(macro_average(recomputed_hybrid)), selected_hybrid_experiment),
        ("reranker", metrics_dict(macro_average(recomputed_reranker)), selected_experiment),
    ):
        for field in BASELINE_METRIC_KEYS:
            if not math.isclose(
                actual[field],
                float(experiment["metrics"][field]),
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise TuningArtifactError(f"selected {label} {field} does not recompute")


def validate_tuning_artifacts(
    ablation_path: Path | str,
    error_analysis_path: Path | str,
    *,
    day11_metrics: Mapping[str, Any],
    expected_query_ids: Sequence[str],
    expected_gold_sha256: str,
) -> None:
    """Load and validate persisted Day 12 machine-readable artifacts."""

    try:
        ablation = json.loads(Path(ablation_path).read_text(encoding="utf-8"))
        error_analysis = json.loads(Path(error_analysis_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TuningArtifactError(f"could not load tuning artifacts: {error}") from error
    if not isinstance(ablation, Mapping) or not isinstance(error_analysis, Mapping):
        raise TuningArtifactError("tuning artifacts must be JSON objects")
    validate_tuning_data(
        ablation,
        error_analysis,
        day11_metrics=day11_metrics,
        expected_query_ids=expected_query_ids,
        expected_gold_sha256=expected_gold_sha256,
    )


def atomic_write_text(path: Path | str, content: str) -> None:
    """Durably and atomically replace one tracked tuning artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
