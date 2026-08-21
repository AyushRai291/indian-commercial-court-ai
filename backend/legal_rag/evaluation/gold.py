"""Typed loading, validation, and review rendering for gold queries."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from legal_rag.models import Case, Paragraph


QUERY_TYPES: Final[tuple[str, ...]] = (
    "legal_principle",
    "statutory_interpretation",
    "doctrine_or_test",
    "procedural",
    "fact_pattern",
    "exact_terminology",
    "semantic_paraphrase",
)
DIFFICULTIES: Final[tuple[str, ...]] = ("easy", "medium", "hard")
RELEVANCE_GRADES: Final[tuple[int, ...]] = (1, 2, 3)

_QUERY_FIELDS: Final[frozenset[str]] = frozenset(
    {"query_id", "query", "query_type", "difficulty", "notes", "relevant_paragraphs"}
)
_LABEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "paragraph_uid",
        "relevance",
        "case_name",
        "case_number",
        "paragraph_number",
        "page_number",
        "reason",
    }
)


class GoldValidationError(ValueError):
    """Raised when gold data fail schema or corpus-provenance validation."""


@dataclass(frozen=True, slots=True)
class GoldParagraphLabel:
    """One graded, paragraph-level relevance judgment."""

    paragraph_uid: str
    relevance: int
    case_name: str
    case_number: str | None
    paragraph_number: int
    page_number: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class GoldQuery:
    """One legal-research query and its verified relevant paragraphs."""

    query_id: str
    query: str
    query_type: str
    difficulty: str
    notes: str
    relevant_paragraphs: tuple[GoldParagraphLabel, ...]


@dataclass(frozen=True, slots=True)
class GoldDatasetStatistics:
    """Descriptive statistics for a validated gold dataset."""

    total_queries: int
    queries_by_type: dict[str, int]
    queries_by_difficulty: dict[str, int]
    total_relevance_labels: int
    relevance_by_grade: dict[str, int]
    average_relevant_paragraphs_per_query: float
    minimum_relevant_paragraphs_per_query: int
    maximum_relevant_paragraphs_per_query: int
    distinct_judgments: int
    queries_with_multiple_relevant_paragraphs: int
    queries_with_multiple_relevant_judgments: int

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _AuthoritativeParagraph:
    paragraph: Paragraph
    case: Case


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GoldValidationError(f"{context} must be a JSON object")
    return value


def _check_fields(
    value: Mapping[str, object], expected: frozenset[str], *, context: str
) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    details: list[str] = []
    if missing:
        details.append(f"missing fields: {', '.join(missing)}")
    if unknown:
        details.append(f"unknown fields: {', '.join(unknown)}")
    if details:
        raise GoldValidationError(f"{context}: {'; '.join(details)}")


def _string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoldValidationError(f"{context} must be a non-empty string")
    return value.strip()


def _nullable_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context=context)


def _integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoldValidationError(f"{context} must be an integer")
    return value


def _nullable_integer(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context=context)


def _parse_label(value: object, *, context: str) -> GoldParagraphLabel:
    record = _mapping(value, context=context)
    _check_fields(record, _LABEL_FIELDS, context=context)
    relevance = _integer(record["relevance"], context=f"{context}.relevance")
    if relevance not in RELEVANCE_GRADES:
        raise GoldValidationError(
            f"{context}.relevance must be one of {RELEVANCE_GRADES}, got {relevance}"
        )
    paragraph_number = _integer(
        record["paragraph_number"], context=f"{context}.paragraph_number"
    )
    if paragraph_number <= 0:
        raise GoldValidationError(f"{context}.paragraph_number must be positive")
    page_number = _nullable_integer(
        record["page_number"], context=f"{context}.page_number"
    )
    if page_number is not None and page_number <= 0:
        raise GoldValidationError(f"{context}.page_number must be positive or null")
    return GoldParagraphLabel(
        paragraph_uid=_string(
            record["paragraph_uid"], context=f"{context}.paragraph_uid"
        ),
        relevance=relevance,
        case_name=_string(record["case_name"], context=f"{context}.case_name"),
        case_number=_nullable_string(
            record["case_number"], context=f"{context}.case_number"
        ),
        paragraph_number=paragraph_number,
        page_number=page_number,
        reason=_string(record["reason"], context=f"{context}.reason"),
    )


def parse_gold_query(value: object, *, context: str = "gold query") -> GoldQuery:
    """Parse and structurally validate one decoded JSON gold-query record."""

    record = _mapping(value, context=context)
    _check_fields(record, _QUERY_FIELDS, context=context)
    query_type = _string(record["query_type"], context=f"{context}.query_type")
    if query_type not in QUERY_TYPES:
        raise GoldValidationError(
            f"{context}.query_type must be one of {QUERY_TYPES}, got {query_type!r}"
        )
    difficulty = _string(record["difficulty"], context=f"{context}.difficulty")
    if difficulty not in DIFFICULTIES:
        raise GoldValidationError(
            f"{context}.difficulty must be one of {DIFFICULTIES}, got {difficulty!r}"
        )
    raw_labels = record["relevant_paragraphs"]
    if isinstance(raw_labels, (str, bytes)) or not isinstance(raw_labels, Sequence):
        raise GoldValidationError(f"{context}.relevant_paragraphs must be an array")
    if not raw_labels:
        raise GoldValidationError(
            f"{context}.relevant_paragraphs must contain at least one label"
        )
    labels = tuple(
        _parse_label(item, context=f"{context}.relevant_paragraphs[{index}]")
        for index, item in enumerate(raw_labels)
    )
    return GoldQuery(
        query_id=_string(record["query_id"], context=f"{context}.query_id"),
        query=_string(record["query"], context=f"{context}.query"),
        query_type=query_type,
        difficulty=difficulty,
        notes=_string(record["notes"], context=f"{context}.notes"),
        relevant_paragraphs=labels,
    )


def load_gold_queries(path: Path | str) -> tuple[GoldQuery, ...]:
    """Load and structurally validate a UTF-8 JSONL gold-query dataset."""

    dataset_path = Path(path)
    queries: list[GoldQuery] = []
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GoldValidationError(
            f"could not read gold dataset {dataset_path}: {error}"
        ) from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        context = f"{dataset_path}:{line_number}"
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise GoldValidationError(f"{context}: invalid JSON: {error.msg}") from error
        queries.append(parse_gold_query(value, context=context))
    if not queries:
        raise GoldValidationError(f"gold dataset {dataset_path} contains no records")
    return tuple(queries)


def _authoritative_paragraphs(
    session: Session, paragraph_uids: set[str]
) -> dict[str, _AuthoritativeParagraph]:
    if not paragraph_uids:
        return {}
    rows = session.execute(
        select(Paragraph, Case)
        .join(Case, Paragraph.case_id == Case.id)
        .where(Paragraph.paragraph_uid.in_(sorted(paragraph_uids)))
    ).all()
    return {
        paragraph.paragraph_uid: _AuthoritativeParagraph(paragraph, case)
        for paragraph, case in rows
    }


def _duplicates(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _typed_structure_errors(query: GoldQuery) -> list[str]:
    """Defend the public validator against manually constructed bad dataclasses."""

    errors: list[str] = []
    for field, value in (
        ("query_id", query.query_id),
        ("query", query.query),
        ("query_type", query.query_type),
        ("difficulty", query.difficulty),
        ("notes", query.notes),
    ):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{query.query_id or '<unknown>'}.{field} must be non-empty")
    if query.query_type not in QUERY_TYPES:
        errors.append(
            f"{query.query_id}.query_type must be one of {QUERY_TYPES}, "
            f"got {query.query_type!r}"
        )
    if query.difficulty not in DIFFICULTIES:
        errors.append(
            f"{query.query_id}.difficulty must be one of {DIFFICULTIES}, "
            f"got {query.difficulty!r}"
        )
    if not query.relevant_paragraphs:
        errors.append(f"{query.query_id} has no relevant paragraph labels")
    for index, label in enumerate(query.relevant_paragraphs):
        context = f"{query.query_id}.relevant_paragraphs[{index}]"
        for field, value in (
            ("paragraph_uid", label.paragraph_uid),
            ("case_name", label.case_name),
            ("reason", label.reason),
        ):
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{context}.{field} must be non-empty")
        if label.case_number is not None and (
            not isinstance(label.case_number, str) or not label.case_number.strip()
        ):
            errors.append(f"{context}.case_number must be non-empty or null")
        if isinstance(label.relevance, bool) or label.relevance not in RELEVANCE_GRADES:
            errors.append(
                f"{context}.relevance must be one of {RELEVANCE_GRADES}, "
                f"got {label.relevance!r}"
            )
        if isinstance(label.paragraph_number, bool) or not isinstance(
            label.paragraph_number, int
        ) or label.paragraph_number <= 0:
            errors.append(f"{context}.paragraph_number must be a positive integer")
        if label.page_number is not None and (
            isinstance(label.page_number, bool)
            or not isinstance(label.page_number, int)
            or label.page_number <= 0
        ):
            errors.append(f"{context}.page_number must be positive or null")
    return errors


def _metadata_errors(
    query: GoldQuery,
    label: GoldParagraphLabel,
    authoritative: _AuthoritativeParagraph,
) -> list[str]:
    expected = authoritative
    comparisons = (
        ("case_name", label.case_name, expected.case.title),
        ("case_number", label.case_number, expected.case.case_number),
        (
            "paragraph_number",
            label.paragraph_number,
            expected.paragraph.paragraph_number,
        ),
        ("page_number", label.page_number, expected.paragraph.page_number),
    )
    return [
        f"{query.query_id} label {label.paragraph_uid}: {field} mismatch "
        f"(gold={gold!r}, database={database!r})"
        for field, gold, database in comparisons
        if gold != database
    ]


def _raise_validation_errors(errors: Sequence[str]) -> None:
    if errors:
        raise GoldValidationError(
            f"gold dataset validation failed with {len(errors)} error(s):\n- "
            + "\n- ".join(errors)
        )


def _statistics(
    queries: Sequence[GoldQuery],
    authoritative: Mapping[str, _AuthoritativeParagraph],
) -> GoldDatasetStatistics:
    label_counts = [len(query.relevant_paragraphs) for query in queries]
    relevance_counts = Counter(
        label.relevance for query in queries for label in query.relevant_paragraphs
    )
    case_ids = {
        authoritative[label.paragraph_uid].case.id
        for query in queries
        for label in query.relevant_paragraphs
    }
    multiple_judgment_queries = sum(
        len(
            {
                authoritative[label.paragraph_uid].case.id
                for label in query.relevant_paragraphs
            }
        )
        > 1
        for query in queries
    )
    return GoldDatasetStatistics(
        total_queries=len(queries),
        queries_by_type={
            query_type: sum(query.query_type == query_type for query in queries)
            for query_type in QUERY_TYPES
        },
        queries_by_difficulty={
            difficulty: sum(query.difficulty == difficulty for query in queries)
            for difficulty in DIFFICULTIES
        },
        total_relevance_labels=sum(label_counts),
        relevance_by_grade={
            str(grade): relevance_counts[grade] for grade in RELEVANCE_GRADES
        },
        average_relevant_paragraphs_per_query=sum(label_counts) / len(queries),
        minimum_relevant_paragraphs_per_query=min(label_counts),
        maximum_relevant_paragraphs_per_query=max(label_counts),
        distinct_judgments=len(case_ids),
        queries_with_multiple_relevant_paragraphs=sum(
            count > 1 for count in label_counts
        ),
        queries_with_multiple_relevant_judgments=multiple_judgment_queries,
    )


def validate_gold_queries(
    queries: Sequence[GoldQuery],
    session: Session,
    *,
    expected_count: int = 40,
) -> GoldDatasetStatistics:
    """Validate dataset-level invariants and all labels against PostgreSQL."""

    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise TypeError("expected_count must be an integer")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    errors: list[str] = []
    if len(queries) != expected_count:
        errors.append(f"expected exactly {expected_count} queries, found {len(queries)}")
    structural_errors: list[str] = []
    for query in queries:
        structural_errors.extend(_typed_structure_errors(query))
    if structural_errors:
        errors.extend(structural_errors)
        _raise_validation_errors(errors)
    duplicate_ids = _duplicates([query.query_id for query in queries])
    if duplicate_ids:
        errors.append(f"duplicate query_id values: {', '.join(duplicate_ids)}")
    normalized_queries: defaultdict[str, list[str]] = defaultdict(list)
    for query in queries:
        normalized_queries[query.query.strip().casefold()].append(query.query_id)
    for ids in normalized_queries.values():
        if len(ids) > 1:
            errors.append(f"duplicate query text in: {', '.join(sorted(ids))}")
    all_uids: set[str] = set()
    for query in queries:
        uids = [label.paragraph_uid for label in query.relevant_paragraphs]
        duplicate_labels = _duplicates(uids)
        if duplicate_labels:
            errors.append(
                f"{query.query_id} has duplicate paragraph_uid labels: "
                f"{', '.join(duplicate_labels)}"
            )
        if not any(label.relevance == 3 for label in query.relevant_paragraphs):
            errors.append(f"{query.query_id} has no relevance=3 paragraph")
        all_uids.update(uids)

    authoritative = _authoritative_paragraphs(session, all_uids)
    for query in queries:
        for label in query.relevant_paragraphs:
            stored = authoritative.get(label.paragraph_uid)
            if stored is None:
                errors.append(
                    f"{query.query_id} references nonexistent paragraph_uid "
                    f"{label.paragraph_uid}"
                )
                continue
            errors.extend(_metadata_errors(query, label, stored))
    _raise_validation_errors(errors)
    return _statistics(queries, authoritative)


def _snippet(text: str, *, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _markdown_text(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def render_review_markdown(
    queries: Sequence[GoldQuery],
    session: Session,
    *,
    expected_count: int = 40,
    snippet_characters: int = 240,
) -> str:
    """Validate and render deterministic, compact human-review Markdown."""

    if snippet_characters < 40:
        raise ValueError("snippet_characters must be at least 40")
    statistics = validate_gold_queries(
        queries, session, expected_count=expected_count
    )
    uids = {
        label.paragraph_uid
        for query in queries
        for label in query.relevant_paragraphs
    }
    authoritative = _authoritative_paragraphs(session, uids)
    lines = [
        "# Gold Retrieval Evaluation Set Review",
        "",
        f"Queries: {statistics.total_queries} · "
        f"Relevance labels: {statistics.total_relevance_labels} · "
        f"Distinct judgments: {statistics.distinct_judgments}",
        "",
    ]
    for query in queries:
        lines.extend(
            [
                f"## {query.query_id} — {_markdown_text(query.query)}",
                "",
                f"- Type: `{query.query_type}`",
                f"- Difficulty: `{query.difficulty}`",
                f"- Notes: {_markdown_text(query.notes)}",
                "",
                "### Relevant paragraphs",
                "",
            ]
        )
        for index, label in enumerate(query.relevant_paragraphs, start=1):
            stored = authoritative[label.paragraph_uid]
            case_number = label.case_number or "not reported"
            page_number = (
                str(label.page_number) if label.page_number is not None else "not reported"
            )
            lines.extend(
                [
                    f"{index}. **Relevance {label.relevance}** — "
                    f"{_markdown_text(label.case_name)} ({_markdown_text(case_number)}), "
                    f"paragraph {label.paragraph_number}, page {page_number}",
                    f"   - Paragraph UID: `{label.paragraph_uid}`",
                    f"   - Reason: {_markdown_text(label.reason)}",
                    f"   - Evidence: “{_snippet(stored.paragraph.text, limit=snippet_characters)}”",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_review_markdown(
    path: Path | str,
    queries: Sequence[GoldQuery],
    session: Session,
    *,
    expected_count: int = 40,
    snippet_characters: int = 240,
) -> None:
    """Atomically write a validated human-review Markdown artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_review_markdown(
        queries,
        session,
        expected_count=expected_count,
        snippet_characters=snippet_characters,
    )
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
