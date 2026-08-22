"""System-neutral execution, validation, and reporting for retrieval benchmarks."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Final, Protocol

from legal_rag.evaluation.gold import GoldQuery
from legal_rag.evaluation.metrics import (
    RetrievalMetrics,
    evaluate_ranking,
    macro_average,
)


SYSTEMS: Final[tuple[str, ...]] = (
    "bm25",
    "dense",
    "hybrid_rrf",
    "hybrid_reranker",
)
SYSTEM_LABELS: Final[dict[str, str]] = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid_rrf": "BM25 + Dense + RRF",
    "hybrid_reranker": "Hybrid + Reranker",
}
METRIC_FIELDS: Final[tuple[str, ...]] = (
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
)


class EvaluationArtifactError(ValueError):
    """Raised when benchmark execution or generated artifacts are incomplete."""


class RankedResult(Protocol):
    paragraph_uid: str


SearchExecution = Callable[
    [str], tuple[Sequence[RankedResult], Mapping[str, Any]]
]


@dataclass(frozen=True, slots=True)
class EvaluationSystem:
    """Named retrieval execution used by the shared evaluation loop."""

    name: str
    search: SearchExecution

    def __post_init__(self) -> None:
        if self.name not in SYSTEMS:
            raise ValueError(f"unknown retrieval system: {self.name}")


def _gold_map(query: GoldQuery) -> dict[str, int]:
    return {
        label.paragraph_uid: label.relevance
        for label in query.relevant_paragraphs
    }


def _finite_nonnegative(value: float, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EvaluationArtifactError(f"{context} must be finite and non-negative")
    return number


def evaluate_system(
    queries: Sequence[GoldQuery],
    system: EvaluationSystem,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Evaluate one retrieval system over every query at final depth ten."""

    records: list[dict[str, Any]] = []
    system_started = perf_counter()
    for index, query in enumerate(queries, start=1):
        query_started = perf_counter()
        results, diagnostics = system.search(query.query)
        runtime_seconds = perf_counter() - query_started
        if len(results) > 10:
            raise EvaluationArtifactError(
                f"{system.name}/{query.query_id} returned more than 10 results"
            )
        ranked_uids = [result.paragraph_uid for result in results]
        if len(ranked_uids) != len(set(ranked_uids)):
            raise EvaluationArtifactError(
                f"{system.name}/{query.query_id} returned duplicate paragraph_uid values"
            )
        gold = _gold_map(query)
        metrics = evaluate_ranking(ranked_uids, gold)
        first_relevant_rank = next(
            (
                rank
                for rank, paragraph_uid in enumerate(ranked_uids, start=1)
                if paragraph_uid in gold
            ),
            None,
        )
        records.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "system": system.name,
                **metrics.to_dict(),
                "first_relevant_rank": first_relevant_rank,
                "gold_relevant_uid_count": len(gold),
                "gold_relevance": [
                    {"paragraph_uid": uid, "relevance": grade}
                    for uid, grade in sorted(gold.items())
                ],
                "retrieved": [
                    {
                        "paragraph_uid": paragraph_uid,
                        "rank": rank,
                        "relevance": gold.get(paragraph_uid, 0),
                    }
                    for rank, paragraph_uid in enumerate(ranked_uids, start=1)
                ],
                "runtime_seconds": _finite_nonnegative(
                    runtime_seconds,
                    context=f"{system.name}/{query.query_id} runtime_seconds",
                ),
                "diagnostics": dict(diagnostics),
            }
        )
        if progress is not None:
            progress(
                f"{SYSTEM_LABELS[system.name]}: {index}/{len(queries)} "
                f"({query.query_id})"
            )
    return records, perf_counter() - system_started


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Recompute macro metrics from query-system records."""

    aggregate: dict[str, dict[str, float]] = {}
    for system in SYSTEMS:
        system_metrics = [
            RetrievalMetrics(
                recall_at_5=float(record["recall_at_5"]),
                recall_at_10=float(record["recall_at_10"]),
                reciprocal_rank=float(record["reciprocal_rank"]),
                ndcg_at_10=float(record["ndcg_at_10"]),
            )
            for record in records
            if record.get("system") == system
        ]
        if not system_metrics:
            raise EvaluationArtifactError(f"no per-query records for {system}")
        averaged = macro_average(system_metrics)
        aggregate[system] = {
            "recall_at_5": averaged.recall_at_5,
            "recall_at_10": averaged.recall_at_10,
            "mrr": averaged.reciprocal_rank,
            "ndcg_at_10": averaged.ndcg_at_10,
        }
    return aggregate


def build_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize factual hit/rank outcomes and nDCG-only system comparisons."""

    by_pair = {
        (str(record["query_id"]), str(record["system"])): record
        for record in records
    }
    per_system: dict[str, Any] = {}
    for system in SYSTEMS:
        selected = [record for record in records if record.get("system") == system]
        ndcg_values = [float(record["ndcg_at_10"]) for record in selected]
        best = max(ndcg_values)
        worst = min(ndcg_values)
        per_system[system] = {
            "queries_with_relevant_top_5": sum(
                float(record["recall_at_5"]) > 0 for record in selected
            ),
            "queries_with_relevant_top_10": sum(
                float(record["recall_at_10"]) > 0 for record in selected
            ),
            "queries_with_zero_relevant_top_10": sum(
                float(record["recall_at_10"]) == 0 for record in selected
            ),
            "queries_with_first_relevant_rank_1": sum(
                record.get("first_relevant_rank") == 1 for record in selected
            ),
            "best_query_ids_by_ndcg_at_10": sorted(
                str(record["query_id"])
                for record in selected
                if math.isclose(float(record["ndcg_at_10"]), best, abs_tol=1e-15)
            ),
            "worst_query_ids_by_ndcg_at_10": sorted(
                str(record["query_id"])
                for record in selected
                if math.isclose(float(record["ndcg_at_10"]), worst, abs_tol=1e-15)
            ),
        }

    query_ids = sorted({str(record["query_id"]) for record in records})
    hybrid_beats_both = [
        query_id
        for query_id in query_ids
        if float(by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"])
        > float(by_pair[(query_id, "bm25")]["ndcg_at_10"])
        and float(by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"])
        > float(by_pair[(query_id, "dense")]["ndcg_at_10"])
    ]
    reranker_improves = [
        query_id
        for query_id in query_ids
        if float(by_pair[(query_id, "hybrid_reranker")]["ndcg_at_10"])
        > float(by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"])
    ]
    reranker_worsens = [
        query_id
        for query_id in query_ids
        if float(by_pair[(query_id, "hybrid_reranker")]["ndcg_at_10"])
        < float(by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"])
    ]
    return {
        "per_system": per_system,
        "comparison_metric": "ndcg_at_10",
        "hybrid_beats_both_individual_retrievers": {
            "count": len(hybrid_beats_both),
            "query_ids": hybrid_beats_both,
        },
        "reranker_improves_hybrid": {
            "count": len(reranker_improves),
            "query_ids": reranker_improves,
        },
        "reranker_worsens_hybrid": {
            "count": len(reranker_worsens),
            "query_ids": reranker_worsens,
        },
    }


def _record_gold(record: Mapping[str, Any]) -> dict[str, int]:
    raw_labels = record.get("gold_relevance")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise EvaluationArtifactError("gold_relevance must be a non-empty array")
    gold: dict[str, int] = {}
    for label in raw_labels:
        if not isinstance(label, Mapping):
            raise EvaluationArtifactError("gold_relevance entries must be objects")
        uid = label.get("paragraph_uid")
        grade = label.get("relevance")
        if not isinstance(uid, str) or not uid.strip():
            raise EvaluationArtifactError("gold paragraph_uid must be non-empty")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade <= 0:
            raise EvaluationArtifactError("gold relevance must be a positive integer")
        if uid in gold:
            raise EvaluationArtifactError(f"duplicate gold paragraph_uid: {uid}")
        gold[uid] = grade
    return gold


def validate_evaluation_data(
    aggregate_document: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_query_ids: Sequence[str],
) -> None:
    """Fail closed unless aggregate and per-query benchmark data are complete."""

    expected_ids = set(expected_query_ids)
    if len(expected_ids) != len(expected_query_ids):
        raise EvaluationArtifactError("expected query IDs must be unique")
    expected_pairs = {(query_id, system) for query_id in expected_ids for system in SYSTEMS}
    actual_pairs: set[tuple[str, str]] = set()
    query_text: dict[str, str] = {}
    for record in records:
        query_id = record.get("query_id")
        system = record.get("system")
        if not isinstance(query_id, str) or not isinstance(system, str):
            raise EvaluationArtifactError("query_id and system must be strings")
        pair = (query_id, system)
        if pair in actual_pairs:
            raise EvaluationArtifactError(f"duplicate query-system record: {pair}")
        actual_pairs.add(pair)
        text = record.get("query")
        if not isinstance(text, str) or not text.strip():
            raise EvaluationArtifactError(f"{pair} has an empty query")
        if query_id in query_text and query_text[query_id] != text:
            raise EvaluationArtifactError(f"inconsistent query text for {query_id}")
        query_text[query_id] = text

        gold = _record_gold(record)
        if record.get("gold_relevant_uid_count") != len(gold):
            raise EvaluationArtifactError(f"{pair} gold UID count mismatch")
        retrieved = record.get("retrieved")
        if not isinstance(retrieved, list) or len(retrieved) > 10:
            raise EvaluationArtifactError(f"{pair} must contain at most 10 results")
        ranked_uids: list[str] = []
        for expected_rank, hit in enumerate(retrieved, start=1):
            if not isinstance(hit, Mapping):
                raise EvaluationArtifactError(f"{pair} retrieved hit must be an object")
            uid = hit.get("paragraph_uid")
            if not isinstance(uid, str) or not uid.strip():
                raise EvaluationArtifactError(f"{pair} has an empty retrieved UID")
            if hit.get("rank") != expected_rank:
                raise EvaluationArtifactError(f"{pair} has non-sequential ranks")
            if hit.get("relevance") != gold.get(uid, 0):
                raise EvaluationArtifactError(f"{pair} has incorrect retrieved grade")
            ranked_uids.append(uid)
        if len(ranked_uids) != len(set(ranked_uids)):
            raise EvaluationArtifactError(f"{pair} has duplicate retrieved UIDs")

        recomputed = evaluate_ranking(ranked_uids, gold)
        expected_first = next(
            (rank for rank, uid in enumerate(ranked_uids, start=1) if uid in gold),
            None,
        )
        if record.get("first_relevant_rank") != expected_first:
            raise EvaluationArtifactError(f"{pair} first relevant rank mismatch")
        for field, value in recomputed.to_dict().items():
            actual = record.get(field)
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise EvaluationArtifactError(f"{pair} {field} must be numeric")
            if not math.isfinite(float(actual)) or not 0 <= float(actual) <= 1:
                raise EvaluationArtifactError(f"{pair} {field} must be within [0,1]")
            if not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12):
                raise EvaluationArtifactError(f"{pair} {field} does not recompute")
        _finite_nonnegative(float(record.get("runtime_seconds", -1)), context=f"{pair} runtime")

    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        unexpected = sorted(actual_pairs - expected_pairs)
        raise EvaluationArtifactError(
            f"incomplete query-system coverage: missing={missing}, unexpected={unexpected}"
        )
    if len(records) != len(expected_ids) * len(SYSTEMS):
        raise EvaluationArtifactError("incorrect per-query record count")

    stored_metrics = aggregate_document.get("metrics")
    if not isinstance(stored_metrics, Mapping):
        raise EvaluationArtifactError("aggregate document has no metrics object")
    recomputed_aggregate = aggregate_records(records)
    for system in SYSTEMS:
        stored_system = stored_metrics.get(system)
        if not isinstance(stored_system, Mapping):
            raise EvaluationArtifactError(f"aggregate metrics missing {system}")
        for field in METRIC_FIELDS:
            value = stored_system.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvaluationArtifactError(f"aggregate {system}/{field} is not numeric")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise EvaluationArtifactError(f"aggregate {system}/{field} outside [0,1]")
            if not math.isclose(
                float(value),
                recomputed_aggregate[system][field],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise EvaluationArtifactError(
                    f"aggregate {system}/{field} does not match per-query mean"
                )


def validate_evaluation_artifacts(
    metrics_path: Path | str,
    per_query_path: Path | str,
    *,
    expected_query_ids: Sequence[str],
) -> None:
    """Load and validate persisted benchmark artifacts."""

    try:
        aggregate = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in Path(per_query_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationArtifactError(f"could not load evaluation artifacts: {error}") from error
    if not isinstance(aggregate, Mapping) or any(
        not isinstance(record, Mapping) for record in records
    ):
        raise EvaluationArtifactError("evaluation artifacts must contain JSON objects")
    validate_evaluation_data(
        aggregate,
        records,
        expected_query_ids=expected_query_ids,
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def render_report(aggregate_document: Mapping[str, Any]) -> str:
    """Render a compact factual Markdown summary from validated aggregate data."""

    metrics = aggregate_document["metrics"]
    diagnostics = aggregate_document["diagnostics"]
    configuration = aggregate_document["configuration"]
    runtime = aggregate_document["runtime_seconds"]
    lines = [
        "# Retrieval Evaluation Report",
        "",
        f"Generated: `{aggregate_document['generated_at']}`",
        "",
        "This benchmark evaluates all 40 frozen pilot queries over the complete "
        "100-judgment corpus without metadata filters. No retrieval parameter or "
        "gold label was tuned during evaluation.",
        "",
        "## Metric definitions",
        "",
        "- Recall@K is macro-averaged binary paragraph recall over all positive "
        "gold grades, with duplicate UIDs counted once.",
        "- MRR is the macro mean reciprocal rank of the first exact gold UID in "
        "the returned top-10 ranking; misses score zero.",
        "- nDCG@10 is macro-averaged graded nDCG with gain `2^relevance - 1` "
        "and discount `log2(rank + 1)`.",
        "- Relevance requires an exact `paragraph_uid` match.",
        "",
        "## Configuration",
        "",
        f"- Corpus paragraphs: {configuration['corpus_paragraph_count']}",
        f"- Gold queries: {configuration['gold_query_count']}",
        f"- BM25: k1={configuration['bm25']['k1']}, b={configuration['bm25']['b']}",
        f"- Dense model: `{configuration['dense']['model']}`",
        f"- Hybrid: BM25 depth={configuration['hybrid_rrf']['bm25_candidate_depth']}, "
        f"dense depth={configuration['hybrid_rrf']['dense_candidate_depth']}, "
        f"RRF k={configuration['hybrid_rrf']['rrf_k']}",
        f"- Reranker: `{configuration['hybrid_reranker']['model']}`, "
        f"candidate depth={configuration['hybrid_reranker']['candidate_k']}",
        f"- Final depth: {configuration['final_top_k']}",
        "- Metadata filters: none",
        "",
        "## Aggregate comparison",
        "",
        "| Retrieval System | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "| ---------------- | -------: | --------: | --: | ------: |",
    ]
    for system in SYSTEMS:
        values = metrics[system]
        lines.append(
            f"| {SYSTEM_LABELS[system]} | {values['recall_at_5']:.6f} | "
            f"{values['recall_at_10']:.6f} | {values['mrr']:.6f} | "
            f"{values['ndcg_at_10']:.6f} |"
        )
    lines.extend(["", "## Factual diagnostics", ""])
    for system in SYSTEMS:
        values = diagnostics["per_system"][system]
        lines.append(
            f"- {SYSTEM_LABELS[system]}: relevant hit in top 5 for "
            f"{values['queries_with_relevant_top_5']} queries; top 10 for "
            f"{values['queries_with_relevant_top_10']}; zero hits in top 10 for "
            f"{values['queries_with_zero_relevant_top_10']}; rank-1 hit for "
            f"{values['queries_with_first_relevant_rank_1']}. Best nDCG query IDs: "
            f"{', '.join(values['best_query_ids_by_ndcg_at_10'])}. Worst: "
            f"{', '.join(values['worst_query_ids_by_ndcg_at_10'])}."
        )
    lines.extend(
        [
            "",
            "Strict nDCG@10 comparisons:",
            "",
            f"- Hybrid beats both individual retrievers: "
            f"{diagnostics['hybrid_beats_both_individual_retrievers']['count']} queries "
            f"({', '.join(diagnostics['hybrid_beats_both_individual_retrievers']['query_ids']) or 'none'}).",
            f"- Reranker improves hybrid: "
            f"{diagnostics['reranker_improves_hybrid']['count']} queries "
            f"({', '.join(diagnostics['reranker_improves_hybrid']['query_ids']) or 'none'}).",
            f"- Reranker worsens hybrid: "
            f"{diagnostics['reranker_worsens_hybrid']['count']} queries "
            f"({', '.join(diagnostics['reranker_worsens_hybrid']['query_ids']) or 'none'}).",
            "",
            "The reranker can reorder only the 50 hybrid candidates it receives; "
            "it cannot recover a relevant paragraph absent from that pool.",
            "",
            "## Approximate runtimes",
            "",
            f"- BM25 evaluation: {runtime['bm25']:.3f} s",
            f"- Dense evaluation: {runtime['dense']:.3f} s",
            f"- Hybrid evaluation: {runtime['hybrid_rrf']:.3f} s",
            f"- Reranker evaluation: {runtime['hybrid_reranker']:.3f} s",
            f"- Cross-encoder cold load: {runtime['cross_encoder_cold_load']:.3f} s",
            f"- Query execution total: {runtime['evaluation_total']:.3f} s",
            f"- Setup plus execution: {runtime['benchmark_total']:.3f} s",
            "",
            "These are pilot measurements, not tuning conclusions. Per-query "
            "rankings are retained for later error analysis.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_evaluation_artifacts(
    output_directory: Path | str,
    aggregate_document: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_query_ids: Sequence[str],
) -> tuple[Path, Path, Path]:
    """Validate, atomically persist, reload, and revalidate all three outputs."""

    validate_evaluation_data(
        aggregate_document,
        records,
        expected_query_ids=expected_query_ids,
    )
    output = Path(output_directory)
    metrics_path = output / "retrieval_metrics.json"
    per_query_path = output / "retrieval_per_query.jsonl"
    report_path = output / "retrieval_evaluation_report.md"
    ordered_records = sorted(
        records,
        key=lambda record: (
            str(record["query_id"]),
            SYSTEMS.index(str(record["system"])),
        ),
    )
    _atomic_text(
        metrics_path,
        json.dumps(aggregate_document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_text(
        per_query_path,
        "".join(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            for record in ordered_records
        ),
    )
    _atomic_text(report_path, render_report(aggregate_document))
    validate_evaluation_artifacts(
        metrics_path,
        per_query_path,
        expected_query_ids=expected_query_ids,
    )
    return metrics_path, per_query_path, report_path
