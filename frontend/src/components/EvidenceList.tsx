import type { EvidenceItem } from '../types'
import { EvidenceCard } from './EvidenceCard'

interface EvidenceListProps {
  evidence: EvidenceItem[]
  selectedEvidenceId: string
  onSelectEvidence: (evidenceId: string) => void
}

export function EvidenceList({
  evidence,
  selectedEvidenceId,
  onSelectEvidence,
}: EvidenceListProps) {
  return (
    <section className="evidence-list" aria-labelledby="evidence-list-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Ranked retrieval</span>
          <h2 id="evidence-list-title">Evidence considered</h2>
        </div>
        <span className="rank-order">Reranked order</span>
      </div>
      <div className="evidence-list__rows">
        {evidence.map((item) => (
          <EvidenceCard
            key={item.evidence_id}
            evidence={item}
            selected={selectedEvidenceId === item.evidence_id}
            onSelect={onSelectEvidence}
          />
        ))}
      </div>
    </section>
  )
}
