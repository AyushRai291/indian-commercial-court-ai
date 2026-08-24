import type { AnswerResponse, EvidenceItem } from '../types'
import { CitationToken } from './CitationToken'

interface GroundedAnswerProps {
  response: AnswerResponse
  onSelectEvidence: (evidenceId: string) => void
}

function renderAnswer(
  answer: string,
  evidenceById: Map<string, EvidenceItem>,
  onSelectEvidence: (evidenceId: string) => void,
) {
  const parts = answer.split(/(\[E\d+\])/g)

  return parts.map((part, index) => {
    const match = /^\[(E\d+)\]$/.exec(part)
    const evidence = match ? evidenceById.get(match[1]) : undefined
    if (!evidence) {
      return <span key={`${part}-${index}`}>{part}</span>
    }
    return (
      <CitationToken
        key={`${evidence.evidence_id}-${index}`}
        evidence={evidence}
        onSelect={onSelectEvidence}
      />
    )
  })
}

export function GroundedAnswer({ response, onSelectEvidence }: GroundedAnswerProps) {
  const evidenceById = new Map(
    response.evidence.map((item) => [item.evidence_id, item]),
  )

  return (
    <section className="answer-card" aria-labelledby="answer-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Grounded answer</span>
          <h2 id="answer-title">Research position</h2>
        </div>
        <span className="evidence-count">
          {response.evidence.length} evidence paragraphs
        </span>
      </div>
      <p className="answer-copy">
        {renderAnswer(response.answer, evidenceById, onSelectEvidence)}
      </p>
      <div className="answer-footnote">
        This Day 14 preview displays static mock output shaped like the grounded-answer API.
      </div>
    </section>
  )
}
