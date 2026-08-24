import type { AnswerResponse, VerificationResponse } from '../types'

interface ProductionTraceProps {
  answer: AnswerResponse
  verification: VerificationResponse
}

export function ProductionTrace({ answer, verification }: ProductionTraceProps) {
  return (
    <details className="production-trace">
      <summary>How this answer was produced</summary>
      <ol aria-label="Research pipeline">
        <li>100-judgment corpus</li>
        <li>BM25 + Dense</li>
        <li>RRF</li>
        <li>Cross-Encoder</li>
        <li>Grounded Answer</li>
        <li>Citation Verification</li>
      </ol>
      <p>
        Static fixture timing: retrieval {answer.retrieval_latency_ms.toFixed(1)} ms;
        claim extraction {verification.claim_extraction_latency_ms.toFixed(1)} ms;
        verification {verification.verification_latency_ms.toFixed(1)} ms; total verify{' '}
        {verification.total_latency_ms.toFixed(1)} ms.
      </p>
    </details>
  )
}
