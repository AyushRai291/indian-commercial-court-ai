import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { EvidenceItem } from '../types'
import { EvidenceCard } from './EvidenceCard'

const evidence: EvidenceItem = {
  evidence_id: 'E1',
  paragraph_uid: 'case-a-20',
  case_id: 1,
  case_name:
    'Perkins Eastman Architects DPC and Another v. HSCC (India) Limited and Another',
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
}

describe('EvidenceCard', () => {
  it('exposes the complete case name as a native tooltip', () => {
    render(
      <EvidenceCard
        evidence={evidence}
        selected={false}
        highlighted={false}
        onSelect={vi.fn()}
      />,
    )

    expect(screen.getByText(evidence.case_name)).toHaveAttribute('title', evidence.case_name)
  })

  it('retains native button selection semantics for keyboard users', async () => {
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(
      <EvidenceCard
        evidence={evidence}
        selected
        highlighted
        onSelect={onSelect}
      />,
    )

    const card = screen.getByRole('button', { name: /show evidence e1/i })
    expect(card).toHaveAttribute('type', 'button')
    expect(card).toHaveAttribute('aria-pressed', 'true')
    expect(card).toHaveAttribute('aria-controls', 'evidence-panel')

    await user.tab()
    expect(card).toHaveFocus()
    await user.keyboard('{Enter}')
    await user.keyboard(' ')

    expect(onSelect).toHaveBeenNthCalledWith(1, 'E1')
    expect(onSelect).toHaveBeenNthCalledWith(2, 'E1')
  })
})
