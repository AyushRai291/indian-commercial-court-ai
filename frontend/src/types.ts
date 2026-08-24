export interface EvidenceItem {
  evidence_id: string
  paragraph_uid: string
  case_id: number
  case_name: string
  case_number: string | null
  court: string | null
  judgment_date: string | null
  page_number: number | null
  paragraph_number: number
  source_url: string | null
  text: string
  bm25_rank: number | null
  bm25_score: number | null
  dense_rank: number | null
  dense_score: number | null
  rrf_score: number | null
  hybrid_rank: number | null
  cross_encoder_score: number | null
  reranked_rank: number
}

export interface AnswerResponse {
  query: string
  answer: string
  used_evidence_ids: string[]
  evidence: EvidenceItem[]
  retrieval_latency_ms: number
  generation_latency_ms: number
  total_latency_ms: number
}

export type VerificationStatus = 'SUPPORTED' | 'PARTIAL' | 'UNSUPPORTED'

export interface VerifiedClaim {
  claim_id: string
  claim: string
  citation_ids: string[]
  status: VerificationStatus
  reason: string
  evidence_uids: string[]
}

export interface VerificationSummary {
  supported: number
  partial: number
  unsupported: number
}

// Mirrors the standalone Day 15 POST /verify response contract.
export interface VerificationResponse {
  claims: VerifiedClaim[]
  summary: VerificationSummary
  claim_extraction_latency_ms: number
  verification_latency_ms: number
  total_latency_ms: number
}

// Presentation-only pairing of the separate /answer and /verify responses.
export interface VerifiedResearchFixture {
  answer: AnswerResponse
  verification: VerificationResponse
}

export interface SearchFilters {
  court: string
  year: string
  caseNumber: string
}

export type WorkspaceView =
  | 'result'
  | 'empty'
  | 'loading'
  | 'no-results'
  | 'backend-error'
  | 'generation-error'
