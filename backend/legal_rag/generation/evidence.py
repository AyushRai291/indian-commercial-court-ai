"""Deterministic request-local evidence assignment and structural validation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from legal_rag.api.schemas import AnswerEvidence, SearchResult

from legal_rag.generation.errors import CitationIntegrityError
from legal_rag.generation.models import GroundedModelOutput


_CITATION_PATTERN = re.compile(r"\[(E[1-9]\d*)\]")
_CITATION_LIKE_PATTERN = re.compile(r"\[([Ee][^\[\]]*)\]")
_EVIDENCE_ID_PATTERN = re.compile(r"E[1-9]\d*")


def assign_evidence_ids(results: Iterable[SearchResult]) -> list[AnswerEvidence]:
    """Assign E1..En in ascending reranked rank while retaining provenance."""

    ordered = sorted(results, key=lambda result: result.final_rank)
    return [
        AnswerEvidence(
            evidence_id=f"E{index}",
            paragraph_uid=result.paragraph_uid,
            text=result.text,
            case_id=result.case_id,
            case_name=result.title,
            case_number=result.case_number,
            court=result.court,
            judgment_date=result.judgment_date,
            source_url=result.source_url,
            paragraph_number=result.paragraph_number,
            page_number=result.page_number,
            bm25_rank=result.bm25_rank,
            bm25_score=result.bm25_score,
            dense_rank=result.dense_rank,
            dense_score=result.dense_score,
            rrf_score=result.rrf_score,
            hybrid_rank=result.hybrid_rank,
            cross_encoder_score=result.cross_encoder_score,
            reranked_rank=result.final_rank,
        )
        for index, result in enumerate(ordered, start=1)
    ]


def cited_evidence_ids(answer: str) -> list[str]:
    """Return unique bracketed evidence IDs in first-appearance order."""

    return list(dict.fromkeys(_CITATION_PATTERN.findall(answer)))


def validate_citation_integrity(
    output: GroundedModelOutput,
    supplied_ids: Iterable[str],
) -> GroundedModelOutput:
    """Require known citations and an exact declared/cited ID sequence match."""

    allowed = set(supplied_ids)
    cited = cited_evidence_ids(output.answer)
    declared = output.used_evidence_ids

    citation_like = _CITATION_LIKE_PATTERN.findall(output.answer)
    if any(_EVIDENCE_ID_PATTERN.fullmatch(value) is None for value in citation_like):
        raise CitationIntegrityError("answer contained a malformed evidence citation")

    if any(_EVIDENCE_ID_PATTERN.fullmatch(value) is None for value in declared):
        raise CitationIntegrityError("model declared a malformed evidence ID")

    unknown = (set(cited) | set(declared)) - allowed
    if unknown:
        raise CitationIntegrityError("model referenced evidence that was not supplied")

    if declared != cited:
        raise CitationIntegrityError(
            "declared evidence IDs do not exactly match answer citations"
        )

    return output
