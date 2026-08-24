"""Verifier adapter over the provider-neutral structured-LLM boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from legal_rag.generation.models import GroundedPrompt
from legal_rag.generation.provider import StructuredLLMProvider
from legal_rag.verification.models import VerifierBatchOutput


class VerificationProvider(Protocol):
    """Injectable batched verifier boundary used by service tests."""

    def verify(self, prompt: GroundedPrompt) -> VerifierBatchOutput: ...


@dataclass(slots=True)
class StructuredVerificationProvider:
    """Request verifier structured output through the existing provider."""

    provider: StructuredLLMProvider

    def verify(self, prompt: GroundedPrompt) -> VerifierBatchOutput:
        return self.provider.parse(prompt, VerifierBatchOutput)

    def close(self) -> None:
        self.provider.close()
