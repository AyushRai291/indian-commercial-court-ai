"""Verifier adapter over the existing structured OpenAI Responses provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from legal_rag.generation.models import GroundedPrompt
from legal_rag.generation.provider import OpenAIResponsesProvider
from legal_rag.verification.models import VerifierBatchOutput


class VerificationProvider(Protocol):
    """Injectable batched verifier boundary used by service tests."""

    def verify(self, prompt: GroundedPrompt) -> VerifierBatchOutput: ...


@dataclass(slots=True)
class OpenAIVerificationProvider:
    """Request verifier structured output through the existing provider."""

    provider: OpenAIResponsesProvider

    def verify(self, prompt: GroundedPrompt) -> VerifierBatchOutput:
        return self.provider.parse(prompt, VerifierBatchOutput)

    def close(self) -> None:
        self.provider.close()
