import type { EvidenceItem } from '../types'

interface EvidencePanelProps {
  evidence: EvidenceItem | null
}

function formatDate(value: string | null) {
  if (!value) return 'Unavailable'
  return new Intl.DateTimeFormat('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  }).format(new Date(`${value}T00:00:00`))
}

function formatScore(value: number | null) {
  return value === null ? 'Unavailable' : value.toFixed(3)
}

export function EvidencePanel({ evidence }: EvidencePanelProps) {
  if (!evidence) {
    return (
      <aside className="evidence-panel evidence-panel--empty" id="evidence-panel">
        <span className="eyebrow">Evidence workspace</span>
        <h2>Select a cited paragraph</h2>
        <p>Citations and ranked cards open the exact judgment evidence here.</p>
      </aside>
    )
  }

  return (
    <aside
      className="evidence-panel"
      id="evidence-panel"
      aria-label="Selected judgment evidence"
      aria-live="polite"
    >
      <div className="evidence-panel__header">
        <div>
          <span className="eyebrow">Selected evidence</span>
          <h2>{evidence.evidence_id}</h2>
        </div>
        <span className="rank-pill">Rank {evidence.reranked_rank}</span>
      </div>

      <div className="case-heading">
        <p className="case-heading__court">{evidence.court ?? 'Court unavailable'}</p>
        <h3>{evidence.case_name}</h3>
        <p>{evidence.case_number ?? 'Case number unavailable'}</p>
      </div>

      <dl className="evidence-meta">
        <div>
          <dt>Judgment date</dt>
          <dd>{formatDate(evidence.judgment_date)}</dd>
        </div>
        <div>
          <dt>Pinpoint</dt>
          <dd>Page {evidence.page_number ?? '—'} · Para {evidence.paragraph_number}</dd>
        </div>
      </dl>

      <div className="evidence-text">
        <span className="eyebrow">Full paragraph text</span>
        <p>{evidence.text}</p>
      </div>

      <details className="retrieval-details">
        <summary>Retrieval metadata</summary>
        <dl>
          <div><dt>Paragraph UID</dt><dd>{evidence.paragraph_uid}</dd></div>
          <div><dt>BM25 rank / score</dt><dd>{evidence.bm25_rank ?? '—'} / {formatScore(evidence.bm25_score)}</dd></div>
          <div><dt>Dense rank / score</dt><dd>{evidence.dense_rank ?? '—'} / {formatScore(evidence.dense_score)}</dd></div>
          <div><dt>RRF score</dt><dd>{formatScore(evidence.rrf_score)}</dd></div>
          <div><dt>Hybrid position</dt><dd>{evidence.hybrid_rank ?? 'Unavailable'}</dd></div>
          <div><dt>Cross-encoder score</dt><dd>{formatScore(evidence.cross_encoder_score)}</dd></div>
          <div><dt>Reranked position</dt><dd>{evidence.reranked_rank}</dd></div>
        </dl>
      </details>

      {evidence.source_url ? (
        <a
          className="source-link"
          href={evidence.source_url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`Open source judgment for ${evidence.case_name}`}
        >
          View source judgment <span aria-hidden="true">↗</span>
        </a>
      ) : null}
    </aside>
  )
}
