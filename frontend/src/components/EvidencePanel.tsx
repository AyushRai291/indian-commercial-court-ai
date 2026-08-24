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
      <p className="sr-only" aria-live="polite">
        Selected evidence {evidence.evidence_id}: {evidence.case_name}, paragraph{' '}
        {evidence.paragraph_number}.
      </p>
      <div className="evidence-panel__header">
        <div>
          <span className="eyebrow">Selected evidence</span>
          <h2>{evidence.evidence_id} · Evidence</h2>
        </div>
      </div>

      <section className="case-heading" aria-labelledby="selected-evidence-case">
        <h3 id="selected-evidence-case">{evidence.case_name}</h3>
        <p className="case-heading__court">{evidence.court ?? 'Court unavailable'}</p>
        <p className="case-heading__number">{evidence.case_number ?? 'Case number unavailable'}</p>
      </section>

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

      <section className="evidence-text" aria-labelledby="evidence-paragraph-title">
        <h3 className="eyebrow" id="evidence-paragraph-title">Full paragraph text</h3>
        <p>{evidence.text}</p>
      </section>

      {evidence.source_url ? (
        <a
          className="source-link"
          href={evidence.source_url}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`Open source judgment for ${evidence.case_name} (opens in a new tab)`}
        >
          Open source judgment <span aria-hidden="true">↗</span>
        </a>
      ) : null}

      <section className="evidence-relationships" aria-labelledby="relationship-title">
        <div className="evidence-relationships__heading">
          <span className="eyebrow">Citation relationship</span>
          <h3 id="relationship-title">Why this evidence was cited</h3>
          <p className="evidence-relationships__count">
            Cited by {citingClaims.length} {citingClaims.length === 1 ? 'claim' : 'claims'}
          </p>
        </div>

        {citingClaims.length > 0 ? (
          <div className="relationship-list">
            {citingClaims.map((claim) => {
              const isSelected = selectedClaim?.claim_id === claim.claim_id
              const detailId = `relationship-detail-${claim.claim_id}`

              return (
                <article
                  className={`relationship-item ${isSelected ? 'relationship-item--expanded' : 'relationship-item--compact'}`}
                  key={claim.claim_id}
                >
                  <button
                    className="relationship-item__toggle"
                    type="button"
                    aria-label={`${isSelected ? 'Current' : 'Select'} claim ${claim.claim_id}, ${claim.status}: ${claim.claim}`}
                    aria-expanded={isSelected}
                    aria-controls={detailId}
                    aria-pressed={isSelected}
                    onClick={() => onSelectClaim(claim.claim_id)}
                  >
                    <span className="relationship-item__topline">
                      <strong>{claim.claim_id}</strong>
                      <StatusBadge status={claim.status} />
                    </span>
                    <span className="relationship-item__claim">{claim.claim}</span>
                  </button>
                  <div
                    className="relationship-item__detail"
                    id={detailId}
                    hidden={!isSelected}
                  >
                    <span className="eyebrow">Verifier reason</span>
                    <p className="relationship-item__reason">{claim.reason}</p>
                  </div>
                </article>
              )
            })}
          </div>
        ) : (
          <p className="evidence-relationships__empty">
            No verified claim currently cites this paragraph.
          </p>
        )}
      </section>

      <details className="retrieval-details">
        <summary>Technical provenance</summary>
        <dl>
          <div><dt>Paragraph UID</dt><dd>{evidence.paragraph_uid}</dd></div>
          <div><dt>BM25 rank</dt><dd>{evidence.bm25_rank ?? 'Unavailable'}</dd></div>
          <div><dt>BM25 score</dt><dd>{formatScore(evidence.bm25_score)}</dd></div>
          <div><dt>Dense rank</dt><dd>{evidence.dense_rank ?? 'Unavailable'}</dd></div>
          <div><dt>Dense score</dt><dd>{formatScore(evidence.dense_score)}</dd></div>
          <div><dt>RRF score</dt><dd>{formatScore(evidence.rrf_score)}</dd></div>
          <div><dt>RRF rank</dt><dd>{evidence.hybrid_rank ?? 'Unavailable'}</dd></div>
          <div><dt>Reranker score</dt><dd>{formatScore(evidence.cross_encoder_score)}</dd></div>
          <div><dt>Retrieval rank</dt><dd>{evidence.reranked_rank}</dd></div>
        </dl>
      </details>
    </aside>
  )
}
