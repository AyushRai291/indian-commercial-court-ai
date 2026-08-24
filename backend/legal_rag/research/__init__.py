"""End-to-end legal research orchestration."""

from legal_rag.research.service import (
    RESEARCH_VERIFICATION_UNAVAILABLE_ERROR,
    ResearchService,
    build_research_service,
)

__all__ = [
    "RESEARCH_VERIFICATION_UNAVAILABLE_ERROR",
    "ResearchService",
    "build_research_service",
]
