import type { ResearchResponse } from '../types'

export function ProductionTrace({ response }: { response: ResearchResponse }) {
  const { latency } = response
  const seconds = (milliseconds: number) => `${(milliseconds / 1000).toFixed(1)}s`

  return (
    <details className="production-trace">
      <summary>How this answer was produced</summary>
      <ol aria-label="Research pipeline">
        <li>100-judgment corpus</li>
        <li>BM25 + semantic retrieval</li>
        <li>RRF fusion</li>
        <li>Cross-encoder reranking</li>
        <li>Grounded generation</li>
        <li>Claim-level citation verification</li>
      </ol>
      <p>
        <strong>Completed in {seconds(latency.total_ms)}</strong>
        <span>
          Retrieval {seconds(latency.retrieval_ms)} · generation{' '}
          {seconds(latency.generation_ms)} · verification{' '}
          {seconds(latency.verification_ms)}
        </span>
      </p>
    </details>
  )
}
