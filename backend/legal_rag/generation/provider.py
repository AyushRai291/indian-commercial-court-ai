"""Small model-provider boundary with one OpenAI Responses implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from legal_rag.generation.errors import (
    MalformedModelResponseError,
    ProviderUnavailableError,
)
from legal_rag.generation.models import GroundedModelOutput, GroundedPrompt


class GroundedAnswerProvider(Protocol):
    """Provider interface used by the generation service and unit tests."""

    def generate(self, prompt: GroundedPrompt) -> GroundedModelOutput: ...


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


@dataclass(slots=True)
class OpenAIResponsesProvider:
    """Generate a Pydantic-validated payload through the Responses API."""

    api_key: str | None
    model: str
    timeout_seconds: float
    _client: Any | None = field(default=None, init=False, repr=False)

    def generate(self, prompt: GroundedPrompt) -> GroundedModelOutput:
        return self.parse(prompt, GroundedModelOutput)

    def parse(
        self,
        prompt: GroundedPrompt,
        output_type: type[StructuredOutput],
    ) -> StructuredOutput:
        """Return one schema-validated Responses API payload."""

        if not self.api_key:
            raise ProviderUnavailableError("OpenAI API key is not configured")

        try:
            client = self._get_client()
            response = client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt},
                ],
                text_format=output_type,
            )
        except ValidationError as exc:
            raise MalformedModelResponseError(
                "provider response did not match the structured contract"
            ) from exc
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError("OpenAI response request failed") from exc

        parsed = response.output_parsed
        if not isinstance(parsed, output_type):
            raise MalformedModelResponseError(
                "provider returned no parsed structured output"
            )
        return parsed

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )
        return self._client
