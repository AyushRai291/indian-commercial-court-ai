import type { EvidenceItem, VerifiedClaim } from '../types'
import { StatusBadge } from './StatusBadge'

interface EvidencePanelProps {
  evidence: EvidenceItem | null
  claims: VerifiedClaim[]
  selectedClaim: VerifiedClaim | null
  onSelectClaim: (claimId: string) => void
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

export function EvidencePanel({
  evidence,
  claims,
  selectedClaim,
  onSelectClaim,
}: EvidencePanelProps) {
  if (!evidence) {
    return (
      <aside
        className="evidence-panel evidence-panel--empty"
        id="evidence-panel"
        aria-label="Selected judgment evidence"
      >
        <span className="eyebrow">Evidence workspace</span>
        <h2>{selectedClaim ? `No cited evidence for ${selectedClaim.claim_id}` : 'Select a cited paragraph'}</h2>
        {selectedClaim ? (
          <>
            <StatusBadge status={selectedClaim.status} />
            <p>{selectedClaim.reason}</p>
          </>
        ) : (
          <p>Citations and ranked cards open the exact judgment evidence here.</p>
        )}
      </aside>
    )
  }

  const citingClaims = claims.filter((claim) =>
    claim.citation_ids.includes(evidence.evidence_id),
  )

  return (
    <aside
      className="evidence-panel"
      id="evidence-panel"
      aria-label="Selected judgment evidence"
    >
      <div className="evidence-panel__header">
        <div>
          <span className="eyebrow">Selected evidence</span>
          <h2>{evidence.evidence_id}</h2>
        </div>
        <span className="rank-pill">Rank {evidence.reranked_rank}</span>
      </div>

      <section className="evidence-relationships" aria-labelledby="relationship-title">
        <div className="evidence-relationships__heading">
          <span className="eyebrow">Claim relationship</span>
          <h3 id="relationship-title">
            Cited by {citingClaims.length} {citingClaims.length === 1 ? 'claim' : 'claims'}
          </h3>
        </div>
        <div className="relationship-list">
          {citingClaims.map((claim) => (
            <button
              className="relationship-item"
              type="button"
              key={claim.claim_id}
              aria-pressed={selectedClaim?.claim_id === claim.claim_id}
              onClick={() => onSelectClaim(claim.claim_id)}
            >
              <span className="relationship-item__topline">
                <strong>{claim.claim_id}</strong>
                <StatusBadge status={claim.status} />
              </span>
              <span className="relationship-item__reason">{claim.reason}</span>
            </button>
          ))}
        </div>
      </section>

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
        <summary>Technical provenance and retrieval scores</summary>
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
          Open source judgment <span aria-hidden="true">↗</span>
        </a>
      ) : null}
    </aside>
  )
}
