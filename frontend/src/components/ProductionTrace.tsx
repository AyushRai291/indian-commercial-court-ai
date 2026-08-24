import type { ResearchResponse } from '../types'

export function ProductionTrace({ response }: { response: ResearchResponse }) {
  const { latency } = response

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
        Live API timing: retrieval {latency.retrieval_ms.toFixed(1)} ms; generation{' '}
        {latency.generation_ms.toFixed(1)} ms; verification{' '}
        {latency.verification_ms.toFixed(1)} ms; total {latency.total_ms.toFixed(1)} ms.
      </p>
    </details>
  )
}
