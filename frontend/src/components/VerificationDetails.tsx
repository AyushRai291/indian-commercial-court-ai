import type { EvidenceItem, VerifiedClaim } from '../types'
import { StatusBadge } from './StatusBadge'

interface VerificationDetailsProps {
  claim: VerifiedClaim | null
  evidenceById: Map<string, EvidenceItem>
  onSelectEvidence: (claimId: string, evidenceId: string) => void
}

export function VerificationDetails({
  claim,
  evidenceById,
  onSelectEvidence,
}: VerificationDetailsProps) {
  if (!claim) return null

  return (
    <section className="verification-details" aria-labelledby="verification-details-title" aria-live="polite">
      <div className="verification-details__heading">
        <div>
          <span className="eyebrow">Verification details</span>
          <h3 id="verification-details-title">{claim.claim_id} evidence check</h3>
        </div>
        <StatusBadge status={claim.status} />
      </div>

      <blockquote>{claim.claim}</blockquote>
      <div className="verification-reason">
        <strong>Verifier reason</strong>
        <p>{claim.reason}</p>
      </div>

      <div className="verification-citations">
        <strong>Citations used</strong>
        {claim.citation_ids.length ? (
          <div>
            {claim.citation_ids.map((evidenceId) => {
              const evidence = evidenceById.get(evidenceId)
              return (
                <button
                  key={evidenceId}
                  type="button"
                  onClick={() => onSelectEvidence(claim.claim_id, evidenceId)}
                  aria-label={`Open ${evidenceId} for ${claim.claim_id}${evidence ? `: ${evidence.case_name}` : ''}`}
                >
                  [{evidenceId}]
                </button>
              )
            })}
          </div>
        ) : (
          <span className="no-citation">No citation attached</span>
        )}
      </div>
    </section>
  )
}
