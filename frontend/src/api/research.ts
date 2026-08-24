import type {
  ResearchRequest,
  ResearchResponse,
  VerificationState,
} from '../types'

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'
const verificationStates = new Set<VerificationState>([
  'complete',
  'not_run',
  'unavailable',
])

export type ResearchErrorKind =
  | 'backend'
  | 'generation'
  | 'verification'
  | 'invalid-response'

export class ResearchApiError extends Error {
  readonly kind: ResearchErrorKind
  readonly status: number | null

  constructor(kind: ResearchErrorKind, status: number | null = null) {
    super('The research request could not be completed.')
    this.name = 'ResearchApiError'
    this.kind = kind
    this.status = status
  }
}

function apiBaseUrl() {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim()
  return (configured || DEFAULT_API_BASE_URL).replace(/\/+$/, '')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || isFiniteNumber(value)
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

function isEvidenceItem(value: unknown) {
  return (
    isRecord(value) &&
    typeof value.evidence_id === 'string' &&
    typeof value.paragraph_uid === 'string' &&
    isFiniteNumber(value.case_id) &&
    typeof value.case_name === 'string' &&
    isNullableString(value.case_number) &&
    isNullableString(value.court) &&
    isNullableString(value.judgment_date) &&
    isNullableNumber(value.page_number) &&
    isFiniteNumber(value.paragraph_number) &&
    isNullableString(value.source_url) &&
    typeof value.text === 'string' &&
    isNullableNumber(value.bm25_rank) &&
    isNullableNumber(value.bm25_score) &&
    isNullableNumber(value.dense_rank) &&
    isNullableNumber(value.dense_score) &&
    isNullableNumber(value.rrf_score) &&
    isNullableNumber(value.hybrid_rank) &&
    isNullableNumber(value.cross_encoder_score) &&
    isFiniteNumber(value.reranked_rank)
  )
}

function isVerifiedClaim(value: unknown) {
  return (
    isRecord(value) &&
    typeof value.claim_id === 'string' &&
    typeof value.claim === 'string' &&
    isStringArray(value.citation_ids) &&
    (value.status === 'SUPPORTED' ||
      value.status === 'PARTIAL' ||
      value.status === 'UNSUPPORTED') &&
    typeof value.reason === 'string' &&
    isStringArray(value.evidence_uids)
  )
}

function isResearchResponse(value: unknown): value is ResearchResponse {
  if (!isRecord(value) || !isRecord(value.latency)) {
    return false
  }

  const verificationState = value.verification_state as VerificationState
  const summary = value.verification_summary
  const hasSummaryCounts =
    isRecord(summary) &&
    isFiniteNumber(summary.supported) &&
    isFiniteNumber(summary.partial) &&
    isFiniteNumber(summary.unsupported)
  const hasValidVerificationState =
    (verificationState === 'unavailable' &&
      summary === null &&
      Array.isArray(value.claims) &&
      value.claims.length === 0 &&
      typeof value.verification_error === 'string' &&
      value.verification_error.trim().length > 0) ||
    (verificationState === 'complete' &&
      hasSummaryCounts &&
      value.verification_error === null) ||
    (verificationState === 'not_run' &&
      hasSummaryCounts &&
      summary.supported === 0 &&
      summary.partial === 0 &&
      summary.unsupported === 0 &&
      Array.isArray(value.claims) &&
      value.claims.length === 0 &&
      value.verification_error === null)

  return (
    typeof value.query === 'string' &&
    typeof value.answer === 'string' &&
    isStringArray(value.used_evidence_ids) &&
    Array.isArray(value.evidence) &&
    value.evidence.every(isEvidenceItem) &&
    Array.isArray(value.claims) &&
    value.claims.every(isVerifiedClaim) &&
    verificationStates.has(verificationState) &&
    hasValidVerificationState &&
    isFiniteNumber(value.latency.retrieval_ms) && value.latency.retrieval_ms >= 0 &&
    isFiniteNumber(value.latency.generation_ms) && value.latency.generation_ms >= 0 &&
    isFiniteNumber(value.latency.verification_ms) && value.latency.verification_ms >= 0 &&
    isFiniteNumber(value.latency.total_ms) && value.latency.total_ms >= 0
  )
}

function classifyFailure(status: number, detail: string): ResearchErrorKind {
  const normalized = detail.toLowerCase()
  if (normalized.includes('verif')) return 'verification'
  if (
    normalized.includes('gemini') ||
    normalized.includes('generation') ||
    normalized.includes('provider') ||
    normalized.includes('answer')
  ) {
    return 'generation'
  }
  return status >= 500 || status === 404 ? 'backend' : 'invalid-response'
}

async function responseDetail(response: Response) {
  try {
    const payload: unknown = await response.json()
    return isRecord(payload) && typeof payload.detail === 'string' ? payload.detail : ''
  } catch {
    return ''
  }
}

export async function runResearch(
  request: ResearchRequest,
  signal?: AbortSignal,
): Promise<ResearchResponse> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl()}/research`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ResearchApiError('backend')
  }

  if (!response.ok) {
    const detail = await responseDetail(response)
    throw new ResearchApiError(classifyFailure(response.status, detail), response.status)
  }

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new ResearchApiError('invalid-response', response.status)
  }

  if (!isResearchResponse(payload)) {
    throw new ResearchApiError('invalid-response', response.status)
  }
  return payload
}
