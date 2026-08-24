"""Claim extraction, structural checks, and batched semantic verification."""

from __future__ import annotations

import re
from collections import Counter
from time import perf_counter

from pydantic import ValidationError

from legal_rag.api.schemas import (
    AnswerEvidence,
    VerificationClaim,
    VerificationSummary,
    VerifyRequest,
    VerifyResponse,
)
from legal_rag.config import Settings, get_settings
from legal_rag.generation.errors import (
    CitationIntegrityError,
    MalformedModelResponseError,
    ProviderUnavailableError,
)
from legal_rag.generation.evidence import validate_citation_integrity
from legal_rag.generation.models import GroundedModelOutput
from legal_rag.generation.provider import GeminiProvider
from legal_rag.verification.claims import extract_material_claims
from legal_rag.verification.errors import (
    InvalidVerificationRequestError,
    MalformedVerifierResponseError,
    VerificationError,
    VerificationProviderUnavailableError,
)
from legal_rag.verification.models import (
    ExtractedClaim,
    VerificationStatus,
    VerifierBatchOutput,
    VerifierModelResult,
)
from legal_rag.verification.prompt import build_verifier_prompt
from legal_rag.verification.provider import (
    StructuredVerificationProvider,
    VerificationProvider,
)


UNCITED_CLAIM_REASON = (
    "No evidence citation was attached to this material claim."
)
_EVIDENCE_ID_PATTERN = re.compile(r"E[1-9]\d*")
_EVIDENCE_REFERENCE_PATTERN = re.compile(r"\bE[1-9]\d*\b")


class VerificationService:
    """Verify cited material claims without retrieval or answer rewriting."""

    def __init__(self, provider: VerificationProvider) -> None:
        self.provider = provider

    def verify(self, request: VerifyRequest) -> VerifyResponse:
        total_started = perf_counter()
        evidence_by_id = self._validate_input(request)

        extraction_started = perf_counter()
        claims = extract_material_claims(request.answer)
        claim_extraction_latency_ms = (
            perf_counter() - extraction_started
        ) * 1000.0

        cited_claims = [claim for claim in claims if claim.citation_ids]
        provider_results: dict[str, VerifierModelResult] = {}
        verification_latency_ms = 0.0
        if cited_claims:
            prompt = build_verifier_prompt(cited_claims, evidence_by_id)
            verification_started = perf_counter()
            try:
                raw_output = self.provider.verify(prompt)
                output = VerifierBatchOutput.model_validate(raw_output)
                provider_results = self._validate_provider_output(
                    output,
                    cited_claims,
                )
            except VerificationError:
                raise
            except MalformedModelResponseError as exc:
                raise MalformedVerifierResponseError(
                    "provider response did not match the verifier contract"
                ) from exc
            except ProviderUnavailableError as exc:
                raise VerificationProviderUnavailableError(
                    "verification provider failed"
                ) from exc
            except ValidationError as exc:
                raise MalformedVerifierResponseError(
                    "provider response did not match the verifier contract"
                ) from exc
            except Exception as exc:
                raise VerificationProviderUnavailableError(
                    "verification provider failed"
                ) from exc
            finally:
                verification_latency_ms = (
                    perf_counter() - verification_started
                ) * 1000.0

        verified_claims = [
            self._result_for_claim(
                claim,
                evidence_by_id,
                provider_results.get(claim.claim_id),
            )
            for claim in claims
        ]
        counts = Counter(item.status for item in verified_claims)
        return VerifyResponse(
            claims=verified_claims,
            summary=VerificationSummary(
                supported=counts[VerificationStatus.SUPPORTED],
                partial=counts[VerificationStatus.PARTIAL],
                unsupported=counts[VerificationStatus.UNSUPPORTED],
            ),
            claim_extraction_latency_ms=claim_extraction_latency_ms,
            verification_latency_ms=verification_latency_ms,
            total_latency_ms=(perf_counter() - total_started) * 1000.0,
        )

    @staticmethod
    def _validate_input(request: VerifyRequest) -> dict[str, AnswerEvidence]:
        evidence_ids = [item.evidence_id for item in request.evidence]
        evidence_uids = [item.paragraph_uid for item in request.evidence]
        if any(
            _EVIDENCE_ID_PATTERN.fullmatch(value) is None
            for value in evidence_ids
        ):
            raise InvalidVerificationRequestError(
                "supplied evidence contained a malformed evidence ID"
            )
        if len(evidence_ids) != len(set(evidence_ids)):
            raise InvalidVerificationRequestError(
                "supplied evidence IDs must be unique"
            )
        if len(evidence_uids) != len(set(evidence_uids)):
            raise InvalidVerificationRequestError(
                "supplied paragraph UIDs must be unique"
            )
        try:
            output = GroundedModelOutput(
                answer=request.answer,
                used_evidence_ids=request.used_evidence_ids,
            )
            validate_citation_integrity(output, evidence_ids)
        except (CitationIntegrityError, ValidationError) as exc:
            raise InvalidVerificationRequestError(
                "answer citations did not match supplied evidence"
            ) from exc
        return {item.evidence_id: item for item in request.evidence}

    @staticmethod
    def _validate_provider_output(
        output: VerifierBatchOutput,
        claims: list[ExtractedClaim],
    ) -> dict[str, VerifierModelResult]:
        expected_ids = [claim.claim_id for claim in claims]
        actual_ids = [result.claim_id for result in output.results]
        if (
            len(actual_ids) != len(set(actual_ids))
            or len(actual_ids) != len(expected_ids)
            or set(actual_ids) != set(expected_ids)
        ):
            raise MalformedVerifierResponseError(
                "provider results did not exactly cover the supplied claims"
            )
        claims_by_id = {claim.claim_id: claim for claim in claims}
        for result in output.results:
            referenced_ids = set(
                _EVIDENCE_REFERENCE_PATTERN.findall(result.reason)
            )
            if not referenced_ids.issubset(
                set(claims_by_id[result.claim_id].citation_ids)
            ):
                raise MalformedVerifierResponseError(
                    "provider reason referenced evidence outside the claim"
                )
        return {result.claim_id: result for result in output.results}

    @staticmethod
    def _result_for_claim(
        claim: ExtractedClaim,
        evidence_by_id: dict[str, AnswerEvidence],
        provider_result: VerifierModelResult | None,
    ) -> VerificationClaim:
        evidence_uids = [
            evidence_by_id[evidence_id].paragraph_uid
            for evidence_id in claim.citation_ids
        ]
        if provider_result is None:
            return VerificationClaim(
                claim_id=claim.claim_id,
                claim=claim.text,
                citation_ids=[],
                status=VerificationStatus.UNSUPPORTED,
                reason=UNCITED_CLAIM_REASON,
                evidence_uids=[],
            )
        return VerificationClaim(
            claim_id=claim.claim_id,
            claim=claim.text,
            citation_ids=claim.citation_ids,
            status=provider_result.status,
            reason=provider_result.reason,
            evidence_uids=evidence_uids,
        )

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()


def build_verification_service(
    settings: Settings | None = None,
) -> VerificationService:
    """Compose the verifier over the same lazy Gemini provider boundary."""

    runtime_settings = settings or get_settings()
    structured_provider = GeminiProvider(
        api_key=runtime_settings.gemini_api_key,
        model=(
            runtime_settings.gemini_verifier_model
            or runtime_settings.gemini_model
        ),
        timeout_seconds=runtime_settings.gemini_timeout_seconds,
    )
    return VerificationService(
        StructuredVerificationProvider(structured_provider)
    )
