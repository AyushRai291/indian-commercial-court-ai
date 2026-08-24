"""Grounded answer generation over request-local reranked evidence."""

from legal_rag.generation.errors import (
    CitationIntegrityError,
    GenerationError,
    MalformedModelResponseError,
    ProviderUnavailableError,
)
from legal_rag.generation.evidence import (
    assign_evidence_ids,
    cited_evidence_ids,
    validate_citation_integrity,
)
from legal_rag.generation.models import GroundedModelOutput, GroundedPrompt
from legal_rag.generation.prompt import (
    EVIDENCE_END_DELIMITER,
    EVIDENCE_START_DELIMITER,
    SYSTEM_PROMPT,
    build_grounded_prompt,
)
from legal_rag.generation.provider import (
    GeminiProvider,
    GroundedAnswerProvider,
    StructuredLLMProvider,
)
from legal_rag.generation.service import (
    NO_EVIDENCE_ANSWER,
    AnswerService,
    build_answer_service,
)

__all__ = [
    "AnswerService",
    "CitationIntegrityError",
    "EVIDENCE_END_DELIMITER",
    "EVIDENCE_START_DELIMITER",
    "GenerationError",
    "GeminiProvider",
    "GroundedAnswerProvider",
    "GroundedModelOutput",
    "GroundedPrompt",
    "MalformedModelResponseError",
    "NO_EVIDENCE_ANSWER",
    "ProviderUnavailableError",
    "StructuredLLMProvider",
    "SYSTEM_PROMPT",
    "assign_evidence_ids",
    "build_answer_service",
    "build_grounded_prompt",
    "cited_evidence_ids",
    "validate_citation_integrity",
]
