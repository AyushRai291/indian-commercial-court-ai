import type { EvidenceItem, ResearchResponse, VerifiedClaim } from '../types'
import { CitationToken } from './CitationToken'
import { ProductionTrace } from './ProductionTrace'
import { StatusBadge } from './StatusBadge'
import { VerificationDetails } from './VerificationDetails'
import { VerificationSummary } from './VerificationSummary'

interface GroundedAnswerProps {
  response: ResearchResponse
  selectedClaimId: string
  selectedEvidenceId: string
  onSelectClaim: (claimId: string) => void
  onSelectClaimEvidence: (claimId: string, evidenceId: string) => void
  onSelectEvidence: (evidenceId: string) => void
}

const citationTokenPattern = /(\[E[1-9]\d*\])/g
const exactCitationPattern = /^\[(E[1-9]\d*)\]$/

function GroundedResponse({
  answer,
  evidenceById,
  selectedEvidenceId,
  onSelectEvidence,
}: {
  answer: string
  evidenceById: Map<string, EvidenceItem>
  selectedEvidenceId: string
  onSelectEvidence: (evidenceId: string) => void
}) {
  return (
    <div className="grounded-response">
      <span className="eyebrow">Generated answer</span>
      <p>
        {answer.split(citationTokenPattern).map((part, index) => {
          const match = part.match(exactCitationPattern)
          const evidence = match ? evidenceById.get(match[1]) : null
          return evidence ? (
            <CitationToken
              key={`${evidence.evidence_id}-${index}`}
              evidence={evidence}
              selected={selectedEvidenceId === evidence.evidence_id}
              onSelect={onSelectEvidence}
            />
          ) : (
            <span key={`${part}-${index}`}>{part}</span>
          )
        })}
      </p>
    </div>
  )
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
  selectedClaimId,
  selectedEvidenceId,
  onSelectClaim,
  onSelectClaimEvidence,
  onSelectEvidence,
}: GroundedAnswerProps) {
  const evidenceById = new Map(
    response.evidence.map((item) => [item.evidence_id, item]),
  )
  const selectedClaim =
    response.claims.find((claim) => claim.claim_id === selectedClaimId) ??
    response.claims[0] ??
    null
  const verificationSummary =
    response.verification_state === 'complete' ? response.verification_summary : null

  return (
    <section className="answer-card" aria-labelledby="answer-title">
      <div className="section-heading answer-card__heading">
        <div>
          <span className="eyebrow">Grounded answer + citation verifier</span>
          <h2 id="answer-title">Research position</h2>
        </div>
        <span className="evidence-count">
          {response.evidence.length} evidence paragraphs · {response.latency.total_ms.toFixed(0)} ms
        </span>
      </div>

      <GroundedResponse
        answer={response.answer}
        evidenceById={evidenceById}
        selectedEvidenceId={selectedEvidenceId}
        onSelectEvidence={onSelectEvidence}
      />

      {verificationSummary ? (
        <>
          <VerificationSummary
            claimCount={response.claims.length}
            summary={verificationSummary}
          />
          <div className="claim-list" aria-label="Material claims">
            {response.claims.map((claim) => (
              <ClaimRow
                key={claim.claim_id}
                claim={claim}
                evidenceById={evidenceById}
                selected={claim.claim_id === selectedClaimId}
                selectedEvidenceId={selectedEvidenceId}
                onSelectClaim={onSelectClaim}
                onSelectEvidence={onSelectClaimEvidence}
              />
            ))}
          </div>
          <VerificationDetails
            claim={selectedClaim}
            evidenceById={evidenceById}
            onSelectEvidence={onSelectClaimEvidence}
          />
        </>
      ) : (
        <div className="verification-notice" role="status">
          <strong>
            {response.verification_state === 'unavailable'
              ? 'Citation verification unavailable'
              : 'Citation verification was not run'}
          </strong>
          <p>
            The grounded answer and retrieved evidence remain available, but no verifier result
            has been invented. Try the research request again when verification is available.
          </p>
        </div>
      )}

      <ProductionTrace response={response} />
      <div className="answer-footnote">
        Live research response. Open citations to inspect the exact retrieved judgment paragraph.
      </div>
    </section>
  )
}
