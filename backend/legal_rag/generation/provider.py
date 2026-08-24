"""Small structured-LLM boundary with one Gemini implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from google import genai
from google.genai import types
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


class StructuredLLMProvider(Protocol):
    """Provider-neutral structured-output interface for service adapters."""

    def parse(
        self,
        prompt: GroundedPrompt,
        output_type: type[StructuredOutput],
    ) -> StructuredOutput: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class GeminiProvider:
    """Generate Pydantic-validated JSON with the official Google Gen AI SDK."""

    api_key: str | None = field(repr=False)
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
        """Return one schema-validated Gemini payload."""

        if not self.api_key or not self.api_key.strip():
            raise ProviderUnavailableError("Gemini API key is not configured")

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt.user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=prompt.system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=output_type.model_json_schema(),
                    automatic_function_calling=(
                        types.AutomaticFunctionCallingConfig(disable=True)
                    ),
                ),
            )
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError("Gemini request failed") from exc

        try:
            parsed = response.parsed
            if parsed is not None:
                return output_type.model_validate(parsed)
            response_text = response.text
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError("provider returned no structured output")
            return output_type.model_validate_json(response_text)
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise MalformedModelResponseError(
                "provider response did not match the structured contract"
            ) from exc

    def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            client.close()

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=max(1, int(self.timeout_seconds * 1000)),
                ),
            )
        return self._client
