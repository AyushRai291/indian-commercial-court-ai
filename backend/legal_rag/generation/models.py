"""Internal structured-output and prompt contracts for grounded generation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GroundedModelOutput(BaseModel):
    """Strict payload requested from and returned by the model provider."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="Concise answer containing citations such as [E1].",
    )
    used_evidence_ids: list[str] = Field(
        description=(
            "Unique evidence IDs cited in the answer, in first-appearance order."
        ),
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer must not be blank")
        return stripped


@dataclass(frozen=True, slots=True)
class GroundedPrompt:
    """Separated trusted instructions and untrusted request/evidence data."""

    system_prompt: str
    user_prompt: str
