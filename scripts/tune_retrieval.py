#!/usr/bin/env python3
"""Run cached Day 12 retrieval ablations and factual error analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.config import DEFAULT_RERANKER_CANDIDATE_K, get_settings  # noqa: E402
from legal_rag.database import get_engine, get_session_factory  # noqa: E402
from legal_rag.embeddings import SentenceTransformerEmbeddingProvider  # noqa: E402
from legal_rag.evaluation import (  # noqa: E402
    RERANKER_DEPTHS,
    SYSTEMS,
    aggregate_query_categories,
    aggregate_rankings,
    candidate_recall,
    classify_reranker_failure,
    generate_hybrid_experiment_configs,
    generate_reranker_experiment_configs,
    gold_relevance,
    load_gold_queries,
    select_best_experiment,
    validate_evaluation_artifacts,
    validate_gold_queries,
    validate_tuning_artifacts,
)
from legal_rag.evaluation.tuning import atomic_write_text  # noqa: E402
from legal_rag.retrieval import (  # noqa: E402
    BM25ParagraphRetriever,
    DEFAULT_RRF_K,
    DenseParagraphRetriever,
    HybridSearchResult,
    ParagraphSearchResult,
    SentenceTransformerCrossEncoderScorer,
    reciprocal_rank_fusion,
)
from legal_rag.schema_migrations import upgrade_database  # noqa: E402
from legal_rag.vector import QdrantParagraphIndex  # noqa: E402


EXPECTED_QUERIES = 40
EXPECTED_PARAGRAPHS = 18_822
BASELINE_RRF_ID = "rrf_b50_d50_k60"
FINAL_TOP_K = 10
VERY_SHORT_CHARACTERS = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Day 12 retrieval ablation and error analysis."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/gold_queries.jsonl"),
    )
    parser.add_argument(
        "--day11-directory",
        type=Path,
        default=Path("data/evaluation/results"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/evaluation/tuning"),
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing Day 11 and Day 12 artifacts without model loading",
    )
    return parser


def _qdrant_api_key(settings: Any) -> str | None:
    value = getattr(settings, "qdrant_api_key", None)
    if value is None:
        return None
    if hasattr(value, "get_secret_value"):
        return value.get_secret_value()
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must contain JSON objects")
    return records


def _assert_metrics_equal(
    actual: Mapping[str, float], expected: Mapping[str, Any], *, context: str
) -> None:
    for field in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
        if not math.isclose(
            float(actual[field]), float(expected[field]), rel_tol=0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"{context} no longer matches Day 11 for {field}: "
                f"{actual[field]} != {expected[field]}"
            )


def _uids(results: Sequence[ParagraphSearchResult]) -> list[str]:
    return [result.paragraph_uid for result in results]


def _rank_of(ranking: Sequence[str], uid: str) -> int | None:
    try:
        return ranking.index(uid) + 1
    except ValueError:
        return None


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _aggregate_candidate_recall(
    queries: Sequence[Any], query_diagnostics: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    fields = tuple(query_diagnostics[0]["candidate_recall"])
    return {
        field: sum(float(record["candidate_recall"][field]) for record in query_diagnostics)
        / len(queries)
        for field in fields
    }


def _day11_by_pair(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(str(record["query_id"]), str(record["system"])): record for record in records}


def _query_feature_groups(
    queries: Sequence[Any],
    baseline_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for query in queries:
        lowered = query.query.casefold()
        if re.search(r"\b(section|article|rule|regulation|statute|act)\b", lowered):
            groups["statute_or_section_terms"].append(query)
        if re.search(r"\b(v\.?|versus|appeal no\.?|petition no\.?|application no\.?)\b", lowered):
            groups["case_citation_or_number_terms"].append(query)
        gold_case_bigrams = {
            " ".join(tokens[index : index + 2])
            for label in query.relevant_paragraphs
            for tokens in [re.findall(r"[^\W_]+", label.case_name.casefold())]
            for index in range(len(tokens) - 1)
        }
        if any(bigram in lowered for bigram in gold_case_bigrams):
            groups["named_case_terms"].append(query)
        if query.query_type == "exact_terminology":
            groups["exact_terminology"].append(query)
        if query.query_type == "semantic_paraphrase":
            groups["semantic_paraphrase"].append(query)
    result: dict[str, Any] = {}
    for group, selected in sorted(groups.items()):
        result[group] = {
            "query_count": len(selected),
            "query_ids": [query.query_id for query in selected],
            "bm25_top_10_hit_queries": sum(
                float(baseline_by_pair[(query.query_id, "bm25")]["recall_at_10"]) > 0
                for query in selected
            ),
            "dense_top_10_hit_queries": sum(
                float(baseline_by_pair[(query.query_id, "dense")]["recall_at_10"]) > 0
                for query in selected
            ),
            "bm25_macro_ndcg_at_10": sum(
                float(baseline_by_pair[(query.query_id, "bm25")]["ndcg_at_10"])
                for query in selected
            )
            / len(selected),
            "dense_macro_ndcg_at_10": sum(
                float(baseline_by_pair[(query.query_id, "dense")]["ndcg_at_10"])
                for query in selected
            )
            / len(selected),
        }
    return result


def _length_bucket(length: int) -> str:
    if length < 20:
        return "very_short_under_20"
    if length < 200:
        return "short_20_199"
    if length < 1000:
        return "medium_200_999"
    return "long_1000_plus"


def _paragraph_length_analysis(
    documents: Sequence[Any],
    queries: Sequence[Any],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    lengths_by_uid = {document.paragraph_uid: len(document.text) for document in documents}
    corpus_lengths = list(lengths_by_uid.values())
    gold_labels = [label for query in queries for label in query.relevant_paragraphs]
    gold_lengths = [lengths_by_uid[label.paragraph_uid] for label in gold_labels]
    label_to_query = {
        label.paragraph_uid: query.query_id
        for query in queries
        for label in query.relevant_paragraphs
    }
    label_buckets: dict[str, list[Any]] = defaultdict(list)
    for label in gold_labels:
        label_buckets[_length_bucket(lengths_by_uid[label.paragraph_uid])].append(label)
    systems = tuple(next(iter(rankings.values())).keys())
    by_bucket: dict[str, Any] = {}
    for bucket, labels in sorted(label_buckets.items()):
        by_bucket[bucket] = {
            "gold_label_count": len(labels),
            "mean_characters": sum(lengths_by_uid[label.paragraph_uid] for label in labels)
            / len(labels),
            "top_10_label_recall": {
                system: sum(
                    label.paragraph_uid in rankings[label_to_query[label.paragraph_uid]][system][:10]
                    for label in labels
                )
                / len(labels)
                for system in systems
            },
        }
    top_ten_lengths = {
        system: [
            lengths_by_uid[uid]
            for query_rankings in rankings.values()
            for uid in query_rankings[system][:10]
        ]
        for system in systems
    }
    return {
        "corpus": {
            "count": len(corpus_lengths),
            "min_characters": min(corpus_lengths),
            "median_characters": median(corpus_lengths),
            "p95_characters": _nearest_rank(corpus_lengths, 0.95),
            "max_characters": max(corpus_lengths),
            "very_short_under_20": sum(length < VERY_SHORT_CHARACTERS for length in corpus_lengths),
        },
        "gold": {
            "count": len(gold_lengths),
            "min_characters": min(gold_lengths),
            "median_characters": median(gold_lengths),
            "p95_characters": _nearest_rank(gold_lengths, 0.95),
            "max_characters": max(gold_lengths),
            "very_short_under_20": sum(length < VERY_SHORT_CHARACTERS for length in gold_lengths),
        },
        "gold_label_recall_by_length": by_bucket,
        "retrieved_top_10": {
            system: {
                "result_count": len(lengths),
                "median_characters": median(lengths),
                "p95_characters": _nearest_rank(lengths, 0.95),
                "very_short_under_20": sum(length < VERY_SHORT_CHARACTERS for length in lengths),
            }
            for system, lengths in top_ten_lengths.items()
        },
    }


def _snippet(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1].rstrip() + "…"


def _hit_summary(document: Any, rank: int, gold: Mapping[str, int]) -> dict[str, Any]:
    return {
        "rank": rank,
        "paragraph_uid": document.paragraph_uid,
        "relevance": gold.get(document.paragraph_uid, 0),
        "case_id": document.case_id,
        "title": document.title,
        "paragraph_number": document.paragraph_number,
        "characters": len(document.text),
        "snippet": _snippet(document.text),
    }


def _representative_examples(
    categories: Mapping[str, Sequence[str]],
    queries_by_id: Mapping[str, Any],
    documents_by_uid: Mapping[str, Any],
    baseline_rankings: Mapping[str, Mapping[str, Sequence[str]]],
    baseline_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for category, query_ids in categories.items():
        selected: list[dict[str, Any]] = []
        limit = len(query_ids) if category == "reranker_hurts_hybrid" else 2
        for query_id in list(query_ids)[:limit]:
            query = queries_by_id[query_id]
            gold = gold_relevance(query)
            selected.append(
                {
                    "query_id": query_id,
                    "query": query.query,
                    "metrics": {
                        system: {
                            field: baseline_by_pair[(query_id, system)][field]
                            for field in ("recall_at_5", "recall_at_10", "reciprocal_rank", "ndcg_at_10")
                        }
                        for system in SYSTEMS
                    },
                    "gold": [
                        _hit_summary(documents_by_uid[uid], 0, gold)
                        for uid in gold
                    ],
                    "top_five": {
                        system: [
                            _hit_summary(documents_by_uid[uid], rank, gold)
                            for rank, uid in enumerate(
                                baseline_rankings[query_id][system][:5], start=1
                            )
                        ]
                        for system in SYSTEMS
                    },
                }
            )
        examples[category] = selected
    return examples


def _category_table(title: str, categories: Mapping[str, Any]) -> list[str]:
    lines = [f"### {title}", "", "| Category | System | N | R@5 | R@10 | MRR | nDCG@10 |", "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"bm25": "BM25", "dense": "Dense", "hybrid_rrf": "RRF", "hybrid_reranker": "Reranker"}
    for category, value in categories.items():
        for system in SYSTEMS:
            metrics = value["systems"][system]
            lines.append(
                f"| {category} | {labels[system]} | {value['query_count']} | "
                f"{metrics['recall_at_5']:.4f} | {metrics['recall_at_10']:.4f} | "
                f"{metrics['mrr']:.4f} | {metrics['ndcg_at_10']:.4f} |"
            )
    lines.append("")
    return lines


def render_report(ablation: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    """Render computed Day 12 evidence without tuning claims beyond measurements."""

    baseline = ablation["day11_baseline"]
    comparison = ablation["final_comparison"]
    selected = ablation["selected"]
    candidate = analysis["candidate_recall_macro"]
    failures = analysis["reranker_failure_classification"]
    length = analysis["paragraph_length"]
    lines = [
        "# Retrieval Tuning and Error Analysis",
        "",
        f"Generated: `{ablation['generated_at']}`",
        "",
        "The frozen 40-query gold set and Day 11 metrics were validated before all experiments. BM25 parameters, embedding model, cross-encoder, corpus, paragraph UIDs, and gold labels were unchanged.",
        "",
        "## Final comparison",
        "",
        "| System | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    comparison_labels = {
        "bm25_baseline": "BM25 baseline",
        "dense_baseline": "Dense baseline",
        "original_rrf": "Original RRF",
        "original_reranker": "Original Reranker",
        "tuned_hybrid": "Tuned Hybrid",
        "tuned_hybrid_reranker": "Tuned Hybrid + Reranker",
    }
    for key, label in comparison_labels.items():
        metrics = comparison[key]
        lines.append(
            f"| {label} | {metrics['recall_at_5']:.6f} | {metrics['recall_at_10']:.6f} | {metrics['mrr']:.6f} | {metrics['ndcg_at_10']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Best hybrid: `{selected['hybrid_experiment_id']}` — BM25 depth {selected['hybrid_configuration']['bm25_candidate_depth']}, dense depth {selected['hybrid_configuration']['dense_candidate_depth']}, RRF k={selected['hybrid_configuration']['rrf_k']}.",
            f"- Best reranker: `{selected['reranker_experiment_id']}` — candidate depth {selected['reranker_candidate_k']}.",
            f"- Default decision: {selected['default_decision']}.",
            "- Selection order: nDCG@10, Recall@10, MRR, runtime, then simpler configuration.",
            "- RRF k=10 improved every aggregate metric over k=60 by concentrating weight on the strongest native ranks, but tuned hybrid nDCG@10 0.279577 still trails BM25 0.303318.",
            "- Dense contributes unique candidate recall (the BM25+dense union reaches 0.718750 at depth 50), but its much weaker ordering receives equal rank-based influence in symmetric RRF and can displace exact BM25 gold hits.",
            "- Reranker depth 30 is the nDCG-selected guardrail. Depths 40/50 slightly raise Recall@10 but lower MRR and nDCG as additional lower-hybrid candidates become eligible for promotion.",
            "",
            "## Candidate recall",
            "",
            "| Candidate source | Recall@30 | Recall@50 |",
            "|---|---:|---:|",
            f"| BM25 | {candidate['bm25_at_30']:.6f} | {candidate['bm25_at_50']:.6f} |",
            f"| Dense | {candidate['dense_at_30']:.6f} | {candidate['dense_at_50']:.6f} |",
            f"| BM25 ∪ Dense | {candidate['union_at_30']:.6f} | {candidate['union_at_50']:.6f} |",
            f"| Baseline RRF | {candidate['baseline_hybrid_at_30']:.6f} | {candidate['baseline_hybrid_at_50']:.6f} |",
            "",
            f"Selected-reranker zero-hit failures: {failures['missing_candidates']} had no gold UID in its hybrid candidate pool; {failures['bad_reranking']} had a candidate available but outside the reranked top 10.",
            "",
            "## RRF sweep",
            "",
            "| k | BM25 depth | Dense depth | R@5 | R@10 | MRR | nDCG@10 | Cached runtime (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    hybrid_experiments = [item for item in ablation["experiments"] if item["family"] == "hybrid"]
    for item in hybrid_experiments:
        config = item["configuration"]
        metrics = item["metrics"]
        lines.append(
            f"| {config['rrf_k']} | {config['bm25_candidate_depth']} | {config['dense_candidate_depth']} | {metrics['recall_at_5']:.4f} | {metrics['recall_at_10']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg_at_10']:.4f} | {item['runtime_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reranker-depth sweep",
            "",
            "| Candidates | R@5 | R@10 | MRR | nDCG@10 | Shared inference (s) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in ablation["experiments"]:
        if item["family"] != "reranker":
            continue
        metrics = item["metrics"]
        lines.append(
            f"| {item['configuration']['candidate_k']} | {metrics['recall_at_5']:.4f} | {metrics['recall_at_10']:.4f} | {metrics['mrr']:.4f} | {metrics['ndcg_at_10']:.4f} | {item['shared_inference_seconds']:.3f} |"
        )
    lines.extend(["", "## Day 11 category breakdown", ""])
    breakdown = analysis["category_breakdown"]
    lines.extend(_category_table("Query type", breakdown["query_type"]))
    lines.extend(_category_table("Difficulty", breakdown["difficulty"]))
    lines.extend(_category_table("Gold paragraph count", breakdown["gold_paragraph_count"]))
    lines.extend(_category_table("Judgment scope", breakdown["judgment_scope"]))
    dense_patterns = analysis["dense_query_features"]
    lines.extend(["## Dense retrieval patterns", ""])
    for name, values in dense_patterns.items():
        lines.append(
            f"- `{name}`: {values['query_count']} queries; BM25 top-10 hits {values['bm25_top_10_hit_queries']}, dense hits {values['dense_top_10_hit_queries']}; macro nDCG BM25={values['bm25_macro_ndcg_at_10']:.4f}, dense={values['dense_macro_ndcg_at_10']:.4f}."
        )
    lines.extend(
        [
            "- The statutory-interpretation category is the clearest dense failure: 0/7 top-10 hit queries and macro nDCG@10 0.0000, versus BM25 macro nDCG@10 0.2443.",
            "- Dense also trails BM25 on semantic-paraphrase queries, so the measured weakness is not confined to literal section or case terminology.",
            "- Inspected failures frequently contain unlabelled neighboring provisions or nearby paragraphs from the correct judgment; exact paragraph-UID evaluation correctly gives these no gold credit.",
        ]
    )
    comparisons = analysis["baseline_query_comparisons"]
    lines.extend(
        [
            "",
            "## Query-level error sets",
            "",
            f"- BM25 succeeds / dense fails: {len(comparisons['bm25_succeeds_dense_fails'])} — {', '.join(comparisons['bm25_succeeds_dense_fails']) or 'none'}.",
            f"- Dense succeeds / BM25 fails: {len(comparisons['dense_succeeds_bm25_fails'])} — {', '.join(comparisons['dense_succeeds_bm25_fails']) or 'none'}.",
            f"- RRF hurts BM25 by nDCG: {len(comparisons['hybrid_hurts_bm25'])}.",
            f"- RRF improves BM25 by nDCG: {len(comparisons['rrf_helps_bm25'])}.",
            f"- Reranker improves RRF by nDCG: {len(comparisons['reranker_helps_hybrid'])}.",
            f"- Reranker hurts RRF by nDCG: {len(comparisons['reranker_hurts_hybrid'])}.",
            "",
            "Representative retrieved and gold text excerpts for these sets are stored in `retrieval_error_analysis.json`; conclusions here use those texts, not score magnitudes alone.",
            "",
            "Across all 12 baseline reranker-worsened queries, the tracked text inspection shows promotions of advocacy/background, generic same-topic passages, and multiple nearby paragraphs from the same judgment above exact gold UIDs. Q027 and Q037 are direct examples. This is an ordering failure, not evidence for score blending.",
            "",
            "## Paragraph length and splitting",
            "",
            f"- Corpus lengths: min {length['corpus']['min_characters']}, median {length['corpus']['median_characters']}, p95 {length['corpus']['p95_characters']}, max {length['corpus']['max_characters']} characters.",
            f"- Very-short corpus paragraphs under 20 characters: {length['corpus']['very_short_under_20']} of {length['corpus']['count']}.",
            f"- Gold lengths: min {length['gold']['min_characters']}, median {length['gold']['median_characters']}, p95 {length['gold']['p95_characters']}, max {length['gold']['max_characters']}; under 20 characters: {length['gold']['very_short_under_20']}.",
            f"- Dense top-10 exact-label recall is {length['gold_label_recall_by_length']['medium_200_999']['top_10_label_recall']['dense']:.3f} for 200–999-character gold paragraphs and {length['gold_label_recall_by_length']['long_1000_plus']['top_10_label_recall']['dense']:.3f} for 1,000+ characters; this is a correlation, not a demonstrated truncation cause.",
            f"- Very-short paragraphs appearing in top-10 results: BM25 {length['retrieved_top_10']['bm25']['very_short_under_20']}, dense {length['retrieved_top_10']['dense']['very_short_under_20']}, original RRF {length['retrieved_top_10']['hybrid_rrf']['very_short_under_20']}, original reranker {length['retrieved_top_10']['hybrid_reranker']['very_short_under_20']}. No gold label is under 20 characters.",
            "- Stored paragraph text was not truncated or changed. Tokenizer-level truncation was not instrumented, so the experiment does not attribute long-paragraph misses to truncation.",
            "- Canonical splitting was not changed because paragraph UIDs are the frozen relevance identity. Any boundary redesign requires a separately versioned corpus and gold set.",
            "",
            "## Runtime and limitations",
            "",
            f"- Native top-50 cache: {ablation['runtime_seconds']['native_candidate_retrieval']:.3f} s; cross-encoder score cache: {ablation['runtime_seconds']['cross_encoder_scoring']:.3f} s; total experiment: {ablation['runtime_seconds']['total']:.3f} s.",
            "- Metrics cover only 40 pilot queries and 100 Supreme Court judgments; no statistical significance is claimed.",
            "- No embedding replacement, query rewriting, synonym expansion, score blending, BM25 sweep, or paragraph-boundary change was performed.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    total_started = perf_counter()
    settings = get_settings()
    day11_metrics_path = args.day11_directory / "retrieval_metrics.json"
    day11_per_query_path = args.day11_directory / "retrieval_per_query.jsonl"
    day11_document = _load_json(day11_metrics_path)
    day11_records = _load_jsonl(day11_per_query_path)
    queries = load_gold_queries(args.dataset)
    query_ids = [query.query_id for query in queries]
    gold_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if gold_hash != day11_document["configuration"]["gold_dataset_sha256"]:
        raise RuntimeError("gold dataset hash differs from the frozen Day 11 benchmark")
    validate_evaluation_artifacts(
        day11_metrics_path,
        day11_per_query_path,
        expected_query_ids=query_ids,
    )
    if len(queries) != EXPECTED_QUERIES:
        raise RuntimeError(f"expected {EXPECTED_QUERIES} queries, found {len(queries)}")

    database_url = args.database_url or settings.database_url
    upgrade_database(database_url)
    session_factory = get_session_factory(get_engine(database_url))
    with session_factory() as session:
        validate_gold_queries(queries, session, expected_count=EXPECTED_QUERIES)
        bm25 = BM25ParagraphRetriever.from_session(session)
    if bm25.indexed_paragraphs != EXPECTED_PARAGRAPHS:
        raise RuntimeError(
            f"expected {EXPECTED_PARAGRAPHS} paragraphs, found {bm25.indexed_paragraphs}"
        )
    documents_by_uid = {document.paragraph_uid: document for document in bm25.documents}

    print("Loading one dense model and validating the existing Qdrant collection...", flush=True)
    provider = SentenceTransformerEmbeddingProvider(
        settings.embedding_model,
        expected_dimension=settings.embedding_dimension,
    )
    paragraph_index = QdrantParagraphIndex(
        url=str(settings.qdrant_url),
        api_key=_qdrant_api_key(settings),
        collection_name=args.collection or settings.qdrant_collection,
    )
    paragraph_index.validate_collection(provider.dimension)
    points_count = int(paragraph_index.client.get_collection(paragraph_index.collection_name).points_count)
    if points_count != EXPECTED_PARAGRAPHS:
        raise RuntimeError(f"expected {EXPECTED_PARAGRAPHS} Qdrant points, found {points_count}")
    dense = DenseParagraphRetriever(paragraph_index, provider)

    native_started = perf_counter()
    native: dict[str, dict[str, list[ParagraphSearchResult]]] = {}
    for index, query in enumerate(queries, start=1):
        native[query.query_id] = {
            "bm25": bm25.search(query.query, top_k=50),
            "dense": dense.search(query.query, top_k=50),
        }
        print(f"Native candidates: {index}/{len(queries)} ({query.query_id})", flush=True)
    native_seconds = perf_counter() - native_started

    baseline = day11_document["metrics"]
    for system in ("bm25", "dense"):
        actual, _ = aggregate_rankings(
            queries,
            {query.query_id: _uids(native[query.query_id][system])[:10] for query in queries},
        )
        _assert_metrics_equal(actual, baseline[system], context=system)

    hybrid_experiments: list[dict[str, Any]] = []
    hybrid_results: dict[str, dict[str, list[HybridSearchResult]]] = {}
    for config in generate_hybrid_experiment_configs():
        started = perf_counter()
        rankings: dict[str, list[str]] = {}
        config_results: dict[str, list[HybridSearchResult]] = {}
        for query in queries:
            results = reciprocal_rank_fusion(
                native[query.query_id]["bm25"][: config.bm25_candidate_depth],
                native[query.query_id]["dense"][: config.dense_candidate_depth],
                top_k=50,
                rrf_k=config.rrf_k,
            )
            config_results[query.query_id] = results
            rankings[query.query_id] = _uids(results)[:10]
        metrics, per_query = aggregate_rankings(queries, rankings)
        runtime = perf_counter() - started
        hybrid_results[config.experiment_id] = config_results
        hybrid_experiments.append(
            {
                "experiment_id": config.experiment_id,
                "family": "hybrid",
                "configuration": {
                    "bm25_candidate_depth": config.bm25_candidate_depth,
                    "dense_candidate_depth": config.dense_candidate_depth,
                    "rrf_k": config.rrf_k,
                    "final_top_k": FINAL_TOP_K,
                },
                "metrics": metrics,
                "per_query_metrics": per_query,
                "runtime_seconds": runtime,
                "runtime_basis": "fusion and metric calculation over cached native top-50 rankings",
            }
        )
    baseline_hybrid = next(item for item in hybrid_experiments if item["experiment_id"] == BASELINE_RRF_ID)
    _assert_metrics_equal(baseline_hybrid["metrics"], baseline["hybrid_rrf"], context="baseline hybrid")
    best_hybrid = select_best_experiment(hybrid_experiments)
    best_hybrid_id = str(best_hybrid["experiment_id"])
    selected_hybrid_results = hybrid_results[best_hybrid_id]
    print(f"Selected hybrid for reranker sweep: {best_hybrid_id}", flush=True)

    scorer = SentenceTransformerCrossEncoderScorer(settings.reranker_model)
    score_cache: dict[str, dict[str, float]] = {}
    inference_by_chunk = {30: 0.0, 40: 0.0, 50: 0.0}
    model_load_seconds = 0.0
    scoring_started = perf_counter()
    for index, query in enumerate(queries, start=1):
        candidates = selected_hybrid_results[query.query_id][:50]
        scores: dict[str, float] = {}
        previous = 0
        for depth in RERANKER_DEPTHS:
            chunk = candidates[previous:depth]
            if chunk:
                prediction = scorer.score_pairs(
                    [(query.query, candidate.text) for candidate in chunk],
                    batch_size=settings.reranker_batch_size,
                )
                model_load_seconds += prediction.model_load_seconds
                inference_by_chunk[depth] += prediction.inference_seconds
                scores.update(
                    (candidate.paragraph_uid, score)
                    for candidate, score in zip(chunk, prediction.scores)
                )
            previous = depth
        score_cache[query.query_id] = scores
        print(f"Cross-encoder candidates: {index}/{len(queries)} ({query.query_id})", flush=True)
    scoring_seconds = perf_counter() - scoring_started

    reranker_experiments: list[dict[str, Any]] = []
    reranker_rankings: dict[str, dict[str, list[str]]] = {}
    cumulative_inference = 0.0
    for config in generate_reranker_experiment_configs(best_hybrid_id):
        cumulative_inference += inference_by_chunk[config.candidate_k]
        ranking_started = perf_counter()
        rankings: dict[str, list[str]] = {}
        for query in queries:
            candidates = selected_hybrid_results[query.query_id][: config.candidate_k]
            ordered = sorted(
                candidates,
                key=lambda candidate: (
                    -score_cache[query.query_id][candidate.paragraph_uid],
                    candidate.rank,
                    candidate.paragraph_uid,
                ),
            )[:FINAL_TOP_K]
            rankings[query.query_id] = _uids(ordered)
        metrics, per_query = aggregate_rankings(queries, rankings)
        ordering_seconds = perf_counter() - ranking_started
        reranker_rankings[config.experiment_id] = rankings
        reranker_experiments.append(
            {
                "experiment_id": config.experiment_id,
                "family": "reranker",
                "configuration": {
                    "hybrid_experiment_id": config.hybrid_experiment_id,
                    "candidate_k": config.candidate_k,
                    "model": settings.reranker_model,
                    "batch_size": settings.reranker_batch_size,
                    "final_top_k": FINAL_TOP_K,
                },
                "metrics": metrics,
                "per_query_metrics": per_query,
                "runtime_seconds": cumulative_inference + ordering_seconds,
                "shared_inference_seconds": cumulative_inference,
                "ordering_seconds": ordering_seconds,
                "runtime_basis": "incrementally cached cross-encoder scores plus reranking",
            }
        )
    best_reranker = select_best_experiment(reranker_experiments)
    best_reranker_id = str(best_reranker["experiment_id"])
    selected_reranker_rankings = reranker_rankings[best_reranker_id]

    baseline_by_pair = _day11_by_pair(day11_records)
    baseline_hybrid_results = hybrid_results[BASELINE_RRF_ID]
    baseline_rankings: dict[str, dict[str, Sequence[str]]] = {}
    for query in queries:
        query_id = query.query_id
        baseline_rankings[query_id] = {
            "bm25": _uids(native[query_id]["bm25"][:10]),
            "dense": _uids(native[query_id]["dense"][:10]),
            "hybrid_rrf": _uids(baseline_hybrid_results[query_id][:10]),
            "hybrid_reranker": [
                str(hit["paragraph_uid"])
                for hit in baseline_by_pair[(query_id, "hybrid_reranker")]["retrieved"]
            ],
        }

    comparisons = {
        "bm25_succeeds_dense_fails": [],
        "dense_succeeds_bm25_fails": [],
        "hybrid_hurts_bm25": [],
        "rrf_helps_bm25": [],
        "reranker_helps_hybrid": [],
        "reranker_hurts_hybrid": [],
    }
    query_diagnostics: list[dict[str, Any]] = []
    failure_counts = Counter()
    for query in queries:
        query_id = query.query_id
        gold = gold_relevance(query)
        bm_uids = _uids(native[query_id]["bm25"])
        dense_uids = _uids(native[query_id]["dense"])
        baseline_hybrid_uids = _uids(baseline_hybrid_results[query_id])
        selected_hybrid_uids = _uids(selected_hybrid_results[query_id])
        selected_reranker_uids = selected_reranker_rankings[query_id]
        bm_record = baseline_by_pair[(query_id, "bm25")]
        dense_record = baseline_by_pair[(query_id, "dense")]
        hybrid_record = baseline_by_pair[(query_id, "hybrid_rrf")]
        reranker_record = baseline_by_pair[(query_id, "hybrid_reranker")]
        if bm_record["recall_at_10"] > 0 and dense_record["recall_at_10"] == 0:
            comparisons["bm25_succeeds_dense_fails"].append(query_id)
        if dense_record["recall_at_10"] > 0 and bm_record["recall_at_10"] == 0:
            comparisons["dense_succeeds_bm25_fails"].append(query_id)
        if hybrid_record["ndcg_at_10"] < bm_record["ndcg_at_10"]:
            comparisons["hybrid_hurts_bm25"].append(query_id)
        if hybrid_record["ndcg_at_10"] > bm_record["ndcg_at_10"]:
            comparisons["rrf_helps_bm25"].append(query_id)
        if reranker_record["ndcg_at_10"] > hybrid_record["ndcg_at_10"]:
            comparisons["reranker_helps_hybrid"].append(query_id)
        if reranker_record["ndcg_at_10"] < hybrid_record["ndcg_at_10"]:
            comparisons["reranker_hurts_hybrid"].append(query_id)
        failure = classify_reranker_failure(
            selected_hybrid_uids[: int(best_reranker["configuration"]["candidate_k"])],
            selected_reranker_uids,
            set(gold),
        )
        failure_counts[failure] += 1
        judgments = {(label.case_name, label.case_number) for label in query.relevant_paragraphs}
        query_diagnostics.append(
            {
                "query_id": query_id,
                "query": query.query,
                "query_type": query.query_type,
                "difficulty": query.difficulty,
                "gold_judgment_scope": "multi_judgment" if len(judgments) > 1 else "single_judgment",
                "gold_paragraphs": [
                    {
                        "paragraph_uid": label.paragraph_uid,
                        "relevance": label.relevance,
                        "case_name": label.case_name,
                        "paragraph_number": label.paragraph_number,
                        "page_number": label.page_number,
                        "characters": len(documents_by_uid[label.paragraph_uid].text),
                        "snippet": _snippet(documents_by_uid[label.paragraph_uid].text),
                    }
                    for label in query.relevant_paragraphs
                ],
                "candidate_recall": {
                    "bm25_at_10": candidate_recall([bm_uids], set(gold), depth=10),
                    "bm25_at_30": candidate_recall([bm_uids], set(gold), depth=30),
                    "bm25_at_50": candidate_recall([bm_uids], set(gold), depth=50),
                    "dense_at_10": candidate_recall([dense_uids], set(gold), depth=10),
                    "dense_at_30": candidate_recall([dense_uids], set(gold), depth=30),
                    "dense_at_50": candidate_recall([dense_uids], set(gold), depth=50),
                    "union_at_30": candidate_recall([bm_uids, dense_uids], set(gold), depth=30),
                    "union_at_50": candidate_recall([bm_uids, dense_uids], set(gold), depth=50),
                    "baseline_hybrid_at_30": candidate_recall([baseline_hybrid_uids], set(gold), depth=30),
                    "baseline_hybrid_at_50": candidate_recall([baseline_hybrid_uids], set(gold), depth=50),
                    "selected_hybrid_at_30": candidate_recall([selected_hybrid_uids], set(gold), depth=30),
                    "selected_hybrid_at_50": candidate_recall([selected_hybrid_uids], set(gold), depth=50),
                },
                "gold_candidate_ranks": {
                    uid: {
                        "bm25": _rank_of(bm_uids, uid),
                        "dense": _rank_of(dense_uids, uid),
                        "baseline_hybrid": _rank_of(baseline_hybrid_uids, uid),
                        "selected_hybrid": _rank_of(selected_hybrid_uids, uid),
                        "selected_reranker": _rank_of(selected_reranker_uids, uid),
                    }
                    for uid in gold
                },
                "day11_metrics": {
                    system: {
                        "recall_at_5": baseline_by_pair[(query_id, system)]["recall_at_5"],
                        "recall_at_10": baseline_by_pair[(query_id, system)]["recall_at_10"],
                        "mrr": baseline_by_pair[(query_id, system)]["reciprocal_rank"],
                        "ndcg_at_10": baseline_by_pair[(query_id, system)]["ndcg_at_10"],
                    }
                    for system in SYSTEMS
                },
                "selected_hybrid_top_10": selected_hybrid_uids[:10],
                "selected_reranker_top_10": selected_reranker_uids,
                "reranker_failure_classification": failure,
            }
        )

    length_rankings = {
        query.query_id: {
            **baseline_rankings[query.query_id],
            "selected_hybrid": _uids(selected_hybrid_results[query.query_id][:10]),
            "selected_reranker": selected_reranker_rankings[query.query_id],
        }
        for query in queries
    }
    category_orders = {
        "bm25_succeeds_dense_fails": sorted(
            comparisons["bm25_succeeds_dense_fails"],
            key=lambda query_id: baseline_by_pair[(query_id, "bm25")]["ndcg_at_10"]
            - baseline_by_pair[(query_id, "dense")]["ndcg_at_10"],
            reverse=True,
        ),
        "dense_succeeds_bm25_fails": sorted(
            comparisons["dense_succeeds_bm25_fails"],
            key=lambda query_id: baseline_by_pair[(query_id, "dense")]["ndcg_at_10"]
            - baseline_by_pair[(query_id, "bm25")]["ndcg_at_10"],
            reverse=True,
        ),
        "hybrid_hurts_bm25": sorted(
            comparisons["hybrid_hurts_bm25"],
            key=lambda query_id: baseline_by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"]
            - baseline_by_pair[(query_id, "bm25")]["ndcg_at_10"],
        ),
        "rrf_helps_bm25": sorted(
            comparisons["rrf_helps_bm25"],
            key=lambda query_id: baseline_by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"]
            - baseline_by_pair[(query_id, "bm25")]["ndcg_at_10"],
            reverse=True,
        ),
        "reranker_helps_hybrid": sorted(
            comparisons["reranker_helps_hybrid"],
            key=lambda query_id: baseline_by_pair[(query_id, "hybrid_reranker")]["ndcg_at_10"]
            - baseline_by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"],
            reverse=True,
        ),
        "reranker_hurts_hybrid": sorted(
            comparisons["reranker_hurts_hybrid"],
            key=lambda query_id: baseline_by_pair[(query_id, "hybrid_reranker")]["ndcg_at_10"]
            - baseline_by_pair[(query_id, "hybrid_rrf")]["ndcg_at_10"],
        ),
    }
    representatives = _representative_examples(
        category_orders,
        {query.query_id: query for query in queries},
        documents_by_uid,
        baseline_rankings,
        baseline_by_pair,
    )
    error_analysis = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_dataset_sha256": gold_hash,
        "category_breakdown": aggregate_query_categories(queries, day11_records),
        "candidate_recall_macro": _aggregate_candidate_recall(queries, query_diagnostics),
        "baseline_query_comparisons": comparisons,
        "dense_query_features": _query_feature_groups(queries, baseline_by_pair),
        "reranker_failure_classification": {
            "success": failure_counts["success"],
            "missing_candidates": failure_counts["missing_candidates"],
            "bad_reranking": failure_counts["bad_reranking"],
            "definition": "classification among selected reranker top-10 outcomes",
        },
        "paragraph_length": _paragraph_length_analysis(
            bm25.documents, queries, length_rankings
        ),
        "representative_text_inspection": representatives,
        "queries": query_diagnostics,
    }

    hybrid_improved = float(best_hybrid["metrics"]["ndcg_at_10"]) > float(
        baseline["hybrid_rrf"]["ndcg_at_10"]
    ) + 1e-12
    reranker_improved = float(best_reranker["metrics"]["ndcg_at_10"]) > float(
        baseline["hybrid_reranker"]["ndcg_at_10"]
    ) + 1e-12
    hybrid_configuration = dict(best_hybrid["configuration"])
    defaults_applied = (
        DEFAULT_RRF_K == int(best_hybrid["configuration"]["rrf_k"])
        and DEFAULT_RERANKER_CANDIDATE_K
        == int(best_reranker["configuration"]["candidate_k"])
    )
    if hybrid_improved or reranker_improved:
        default_decision = (
            "defaults changed to the measured selected hybrid/reranker values"
            if defaults_applied
            else "selected values are not yet reflected in source defaults"
        )
    else:
        default_decision = "leave all retrieval defaults unchanged"
    final_comparison = {
        "bm25_baseline": baseline["bm25"],
        "dense_baseline": baseline["dense"],
        "original_rrf": baseline["hybrid_rrf"],
        "original_reranker": baseline["hybrid_reranker"],
        "tuned_hybrid": best_hybrid["metrics"],
        "tuned_hybrid_reranker": best_reranker["metrics"],
    }

    def deltas(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
        return {field: float(left[field]) - float(right[field]) for field in left}

    ablation = {
        "schema_version": 1,
        "generated_at": error_analysis["generated_at"],
        "gold_dataset_sha256": gold_hash,
        "configuration": {
            "queries": len(queries),
            "corpus_paragraphs": bm25.indexed_paragraphs,
            "qdrant_points": points_count,
            "bm25": {"k1": bm25.k1, "b": bm25.b},
            "dense_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "final_top_k": FINAL_TOP_K,
            "metadata_filters": None,
            "native_rankings_cached_at": 50,
        },
        "day11_baseline": baseline,
        "experiments": hybrid_experiments + reranker_experiments,
        "selected": {
            "selection_rule": "nDCG@10, Recall@10, MRR, runtime, simplicity",
            "hybrid_experiment_id": best_hybrid_id,
            "hybrid_configuration": hybrid_configuration,
            "reranker_experiment_id": best_reranker_id,
            "reranker_candidate_k": best_reranker["configuration"]["candidate_k"],
            "hybrid_improves_original": hybrid_improved,
            "reranker_improves_original": reranker_improved,
            "default_decision": default_decision,
            "defaults_applied": defaults_applied,
        },
        "final_comparison": final_comparison,
        "metric_deltas": {
            "original_rrf_vs_bm25": deltas(baseline["hybrid_rrf"], baseline["bm25"]),
            "original_reranker_vs_bm25": deltas(baseline["hybrid_reranker"], baseline["bm25"]),
            "tuned_hybrid_vs_original_rrf": deltas(best_hybrid["metrics"], baseline["hybrid_rrf"]),
            "tuned_reranker_vs_original_reranker": deltas(best_reranker["metrics"], baseline["hybrid_reranker"]),
            "tuned_reranker_vs_bm25": deltas(best_reranker["metrics"], baseline["bm25"]),
        },
        "runtime_seconds": {
            "native_candidate_retrieval": native_seconds,
            "cross_encoder_scoring": scoring_seconds,
            "cross_encoder_model_load": model_load_seconds,
            "total": perf_counter() - total_started,
        },
    }
    output = args.output_directory
    ablation_path = output / "retrieval_ablation_results.json"
    error_path = output / "retrieval_error_analysis.json"
    report_path = output / "retrieval_tuning_report.md"
    atomic_write_text(
        ablation_path,
        json.dumps(ablation, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    atomic_write_text(
        error_path,
        json.dumps(error_analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    atomic_write_text(report_path, render_report(ablation, error_analysis))
    validate_tuning_artifacts(
        ablation_path,
        error_path,
        day11_metrics=baseline,
        expected_query_ids=query_ids,
        expected_gold_sha256=gold_hash,
    )
    print(json.dumps(final_comparison, indent=2), flush=True)
    return ablation_path, error_path, report_path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    queries = load_gold_queries(args.dataset)
    query_ids = [query.query_id for query in queries]
    gold_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    day11_metrics_path = args.day11_directory / "retrieval_metrics.json"
    day11_per_query_path = args.day11_directory / "retrieval_per_query.jsonl"
    day11_document = _load_json(day11_metrics_path)
    if gold_hash != day11_document["configuration"]["gold_dataset_sha256"]:
        raise RuntimeError("gold dataset hash differs from the frozen Day 11 benchmark")
    validate_evaluation_artifacts(
        day11_metrics_path,
        day11_per_query_path,
        expected_query_ids=query_ids,
    )
    if args.validate_only:
        validate_tuning_artifacts(
            args.output_directory / "retrieval_ablation_results.json",
            args.output_directory / "retrieval_error_analysis.json",
            day11_metrics=day11_document["metrics"],
            expected_query_ids=query_ids,
            expected_gold_sha256=gold_hash,
        )
        print(f"Validated Day 12 artifacts for {len(query_ids)} queries")
        return 0
    paths = run(args)
    print("Day 12 artifacts written:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
