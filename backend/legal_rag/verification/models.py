"""Internal claim, prompt, and structured verifier-output models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VerificationStatus(str, Enum):
    """Allowed semantic support classifications."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class ExtractedClaim(BaseModel):
    """One material answer proposition and its claim-local citations."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^C[1-9]\d*$")
    text: str = Field(min_length=1)
    citation_ids: list[str]


class VerifierModelResult(BaseModel):
    """One provider classification in a batched structured response."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(pattern=r"^C[1-9]\d*$")
    status: VerificationStatus
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("reason must not be blank")
        return stripped


class VerifierBatchOutput(BaseModel):
    """One bounded provider response containing every cited claim."""

    model_config = ConfigDict(extra="forbid")

    results: list[VerifierModelResult]
