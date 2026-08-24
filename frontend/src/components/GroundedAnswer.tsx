import type { EvidenceItem, ResearchResponse, VerifiedClaim } from '../types'
import { CitationToken } from './CitationToken'
import { ProductionTrace } from './ProductionTrace'
import { StatusBadge } from './StatusBadge'
import { VerificationSummary } from './VerificationSummary'

interface GroundedAnswerProps {
  response: ResearchResponse
  selectedClaimId: string
  selectedEvidenceId: string
  onSelectClaim: (claimId: string) => void
  onSelectClaimEvidence: (claimId: string, evidenceId: string) => void
  onSelectEvidence: (evidenceId: string) => void
}

const citationClusterPattern = /((?:\[E[1-9]\d*\]\s*)+)/g
const exactCitationClusterPattern = /^(?:\[E[1-9]\d*\]\s*)+$/
const evidenceIdPattern = /\[(E[1-9]\d*)\]/g

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
  const blocks = answer
    .split(/\n+/)
    .map((block) => block.trim())
    .filter(Boolean)

  return (
    <div className="grounded-response" aria-label="Generated grounded answer">
      <span className="eyebrow">Generated answer</span>
      <div className="grounded-response__prose">
        {blocks.map((block, blockIndex) => (
          <p
            className={blockIndex === 0 ? 'grounded-response__lead' : undefined}
            key={`${block.slice(0, 32)}-${blockIndex}`}
          >
            {block.split(citationClusterPattern).map((part, partIndex) => {
              if (!exactCitationClusterPattern.test(part)) {
                return <span key={`${part.slice(0, 24)}-${partIndex}`}>{part}</span>
              }

              const evidenceIds = Array.from(
                part.matchAll(evidenceIdPattern),
                (match) => match[1],
              )
              return (
                <span
                  className="citation-cluster"
                  role="group"
                  aria-label={`Citations ${evidenceIds.join(', ')}`}
                  key={`citation-cluster-${blockIndex}-${partIndex}`}
                >
                  {evidenceIds.map((evidenceId, evidenceIndex) => {
                    const evidence = evidenceById.get(evidenceId)
                    return evidence ? (
                      <CitationToken
                        key={`${evidenceId}-${evidenceIndex}`}
                        evidence={evidence}
                        selected={selectedEvidenceId === evidenceId}
                        onSelect={onSelectEvidence}
                      />
                    ) : (
                      <span
                        className="citation-token citation-token--unavailable"
                        aria-label={`Citation ${evidenceId}; matching evidence unavailable`}
                        title={`Matching evidence for ${evidenceId} is unavailable`}
                        key={`${evidenceId}-${evidenceIndex}`}
                      >
                        [{evidenceId}]
                      </span>
                    )
                  })}
                </span>
              )
            })}
          </p>
        ))}
      </div>
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
          {response.evidence.length} evidence paragraphs
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
        Select any evidence ID to inspect the exact judgment paragraph and source.
      </div>
    </section>
  )
}
