import type {
  AnswerResponse,
  EvidenceItem,
  VerificationResponse,
  VerifiedClaim,
} from '../types'
import { CitationToken } from './CitationToken'
import { ProductionTrace } from './ProductionTrace'
import { StatusBadge } from './StatusBadge'
import { VerificationDetails } from './VerificationDetails'
import { VerificationSummary } from './VerificationSummary'

interface GroundedAnswerProps {
  response: AnswerResponse
  verification: VerificationResponse
  selectedClaimId: string
  selectedEvidenceId: string
  onSelectClaim: (claimId: string) => void
  onSelectEvidence: (claimId: string, evidenceId: string) => void
}

function ClaimRow({
  claim,
  evidenceById,
  selected,
  selectedEvidenceId,
  onSelectClaim,
  onSelectEvidence,
}: {
  claim: VerifiedClaim
  evidenceById: Map<string, EvidenceItem>
  selected: boolean
  selectedEvidenceId: string
  onSelectClaim: (claimId: string) => void
  onSelectEvidence: (claimId: string, evidenceId: string) => void
}) {
  return (
    <article
      className={`claim-row claim-row--${claim.status.toLowerCase()} ${selected ? 'claim-row--selected' : ''}`}
      data-status={claim.status}
    >
      <button
        className="claim-row__select"
        type="button"
        aria-pressed={selected}
        aria-label={`Select ${claim.claim_id}, ${claim.status}: ${claim.claim}`}
        onClick={() => onSelectClaim(claim.claim_id)}
      >
        <span className="claim-row__number">{claim.claim_id}</span>
        <span className="claim-row__body">
          <span className="claim-row__status"><StatusBadge status={claim.status} /></span>
          <span className="claim-row__text">{claim.claim}</span>
        </span>
      </button>

      <div className="claim-row__citations" aria-label={`${claim.claim_id} citations`}>
        <span>{claim.citation_ids.length ? 'Verified against' : 'Evidence check'}</span>
        {claim.citation_ids.length ? (
          claim.citation_ids.map((evidenceId) => {
            const evidence = evidenceById.get(evidenceId)
            return evidence ? (
              <CitationToken
                key={evidenceId}
                evidence={evidence}
                linked={selected}
                selected={selected && evidenceId === selectedEvidenceId}
                onSelect={(selectedEvidenceId) =>
                  onSelectEvidence(claim.claim_id, selectedEvidenceId)
                }
              />
            ) : null
          })
        ) : (
          <strong className="no-citation">No citation attached</strong>
        )}
      </div>
    </article>
  )
}

export function GroundedAnswer({
  response,
  verification,
  selectedClaimId,
  selectedEvidenceId,
  onSelectClaim,
  onSelectEvidence,
}: GroundedAnswerProps) {
  const evidenceById = new Map(
    response.evidence.map((item) => [item.evidence_id, item]),
  )
  const selectedClaim =
    verification.claims.find((claim) => claim.claim_id === selectedClaimId) ??
    verification.claims[0] ??
    null

  return (
    <section className="answer-card" aria-labelledby="answer-title">
      <div className="section-heading answer-card__heading">
        <div>
          <span className="eyebrow">Grounded answer + citation verifier</span>
          <h2 id="answer-title">Research position</h2>
        </div>
        <span className="evidence-count">
          {response.evidence.length} evidence paragraphs
        </span>
      </div>

      <VerificationSummary verification={verification} />

      <div className="claim-list" aria-label="Material claims">
        {verification.claims.map((claim) => (
          <ClaimRow
            key={claim.claim_id}
            claim={claim}
            evidenceById={evidenceById}
            selected={claim.claim_id === selectedClaimId}
            selectedEvidenceId={selectedEvidenceId}
            onSelectClaim={onSelectClaim}
            onSelectEvidence={onSelectEvidence}
          />
        ))}
      </div>

      <VerificationDetails
        claim={selectedClaim}
        evidenceById={evidenceById}
        onSelectEvidence={onSelectEvidence}
      />

      <ProductionTrace answer={response} verification={verification} />
      <div className="answer-footnote">
        Day 15 presentation fixture. Answer and verifier output are static, API-shaped mock data.
      </div>
    </section>
  )
}
