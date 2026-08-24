import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ResearchResponse } from '../types'
import { GroundedAnswer } from './GroundedAnswer'

const response: ResearchResponse = {
  query: 'Can an ineligible arbitrator nominate another person as arbitrator?',
  answer:
    'The nomination power fails with the ineligible office [E1] [E2].\n\nThe rule protects the neutrality of the appointment process [E2].',
  used_evidence_ids: ['E1', 'E2'],
  evidence: [
    {
      evidence_id: 'E1',
      paragraph_uid: 'case-a-20',
      case_id: 1,
      case_name: 'Perkins Eastman Architects DPC v. HSCC (India) Ltd.',
      case_number: 'Arbitration Application No. 32 of 2019',
      court: 'Supreme Court of India',
      judgment_date: '2019-11-26',
      page_number: 19,
      paragraph_number: 20,
      source_url: 'https://example.test/perkins.pdf',
      text: 'The interested party cannot appoint the sole arbitrator.',
      bm25_rank: 1,
      bm25_score: 10,
      dense_rank: 2,
      dense_score: 0.8,
      rrf_score: 0.03,
      hybrid_rank: 1,
      cross_encoder_score: 8,
      reranked_rank: 1,
    },
    {
      evidence_id: 'E2',
      paragraph_uid: 'case-b-54',
      case_id: 2,
      case_name: 'TRF Limited v. Energo Engineering Projects Limited',
      case_number: 'Civil Appeal No. 5306 of 2017',
      court: 'Supreme Court of India',
      judgment_date: '2017-07-03',
      page_number: 33,
      paragraph_number: 54,
      source_url: 'https://example.test/trf.pdf',
      text: 'The power to nominate also fails.',
      bm25_rank: 2,
      bm25_score: 9,
      dense_rank: 1,
      dense_score: 0.9,
      rrf_score: 0.029,
      hybrid_rank: 2,
      cross_encoder_score: 7.8,
      reranked_rank: 2,
    },
  ],
  claims: [
    {
      claim_id: 'C1',
      claim: 'The nomination power fails with the ineligible office.',
      citation_ids: ['E1', 'E2'],
      status: 'SUPPORTED',
      reason: 'Both judgments support the proposition.',
      evidence_uids: ['case-a-20', 'case-b-54'],
    },
  ],
  verification_summary: { supported: 1, partial: 0, unsupported: 0 },
  verification_state: 'complete',
  verification_error: null,
  latency: {
    retrieval_ms: 2400,
    generation_ms: 30100,
    verification_ms: 7694,
    total_ms: 40194,
  },
}

describe('GroundedAnswer presentation rendering', () => {
  it('preserves answer paragraphs and groups adjacent citations without merging their actions', async () => {
    const onSelectEvidence = vi.fn()
    const user = userEvent.setup()
    render(
      <GroundedAnswer
        response={response}
        selectedClaimId="C1"
        selectedEvidenceId="E1"
        onSelectClaim={vi.fn()}
        onSelectClaimEvidence={vi.fn()}
        onSelectEvidence={onSelectEvidence}
      />,
    )

    const prose = screen.getByLabelText('Generated grounded answer')
    expect(prose.querySelectorAll('.grounded-response__prose > p')).toHaveLength(2)
    expect(prose.querySelector('.grounded-response__lead')).toHaveTextContent(
      /nomination power fails/i,
    )

    const citationGroup = within(prose).getByRole('group', { name: 'Citations E1, E2' })
    const citationButtons = within(citationGroup).getAllByRole('button')
    expect(citationButtons).toHaveLength(2)
    await user.click(citationButtons[1])
    expect(onSelectEvidence).toHaveBeenCalledWith('E2')
  })

  it('keeps raw milliseconds out of the result header and presents seconds in diagnostics', () => {
    render(
      <GroundedAnswer
        response={response}
        selectedClaimId="C1"
        selectedEvidenceId="E1"
        onSelectClaim={vi.fn()}
        onSelectClaimEvidence={vi.fn()}
        onSelectEvidence={vi.fn()}
      />,
    )

    expect(screen.getByText('2 evidence paragraphs')).not.toHaveTextContent(/ms/i)
    expect(screen.getByText('Completed in 40.2s')).toBeInTheDocument()
    expect(screen.queryByText(/40194 ms/i)).not.toBeInTheDocument()
  })

  it('preserves a citation ID when its matching evidence is unavailable', () => {
    render(
      <GroundedAnswer
        response={{ ...response, answer: 'The proposition is cited as returned [E1] [E99].' }}
        selectedClaimId="C1"
        selectedEvidenceId="E1"
        onSelectClaim={vi.fn()}
        onSelectClaimEvidence={vi.fn()}
        onSelectEvidence={vi.fn()}
      />,
    )

    expect(
      screen.getByLabelText('Citation E99; matching evidence unavailable'),
    ).toHaveTextContent('[E99]')
    expect(screen.queryByRole('button', { name: /show evidence E99/i })).not.toBeInTheDocument()
  })
})
