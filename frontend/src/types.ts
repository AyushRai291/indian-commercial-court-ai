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

export interface ResearchRequestFilters {
  court?: string
  year?: number
  case_number?: string
}

export interface ResearchRequest {
  query: string
  top_k: number
  filters: ResearchRequestFilters
}

export type VerificationState = 'complete' | 'not_run' | 'unavailable'

export interface ResearchLatency {
  retrieval_ms: number
  generation_ms: number
  verification_ms: number
  total_ms: number
}

// Live POST /research response contract.
export interface ResearchResponse {
  query: string
  answer: string
  used_evidence_ids: string[]
  evidence: EvidenceItem[]
  claims: VerifiedClaim[]
  verification_summary: VerificationSummary | null
  verification_state: VerificationState
  verification_error?: string | null
  latency: ResearchLatency
}

// Legacy fixture shape retained for isolated component/test data.
export interface VerificationResponse {
  claims: VerifiedClaim[]
  summary: VerificationSummary
  claim_extraction_latency_ms: number
  verification_latency_ms: number
  total_latency_ms: number
}

// Test/development fixture pairing; runtime research does not use this type.
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
  | 'verification-error'
