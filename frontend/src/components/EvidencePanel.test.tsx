import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { EvidenceItem, VerifiedClaim } from '../types'
import { EvidencePanel } from './EvidencePanel'

const evidence: EvidenceItem = {
  evidence_id: 'E4',
  paragraph_uid: 'perkins-para-20',
  case_id: 101,
  case_name: 'Perkins Eastman Architects DPC v. HSCC (India) Ltd.',
  case_number: 'Arbitration Application No. 32 of 2019',
  court: 'Supreme Court of India',
  judgment_date: '2019-11-26',
  page_number: 19,
  paragraph_number: 20,
  source_url: 'https://example.test/perkins.pdf',
  text: 'A person interested in the outcome cannot have the power to appoint a sole arbitrator.',
  bm25_rank: 2,
  bm25_score: 14.2,
  dense_rank: 1,
  dense_score: 0.91,
  rrf_score: 0.03,
  hybrid_rank: 1,
  cross_encoder_score: 8.4,
  reranked_rank: 1,
}

const claims: VerifiedClaim[] = [
  {
    claim_id: 'C1',
    claim: 'An interested party cannot appoint the sole arbitrator.',
    citation_ids: ['E4'],
    status: 'SUPPORTED',
    reason: 'The selected paragraph directly states the appointment rule.',
    evidence_uids: ['perkins-para-20'],
  },
  {
    claim_id: 'C2',
    claim: 'The rule also applies to a nomination made by the ineligible appointee.',
    citation_ids: ['E4'],
    status: 'PARTIAL',
    reason: 'The paragraph supports the core rule but not every part of this broader claim.',
    evidence_uids: ['perkins-para-20'],
  },
  {
    claim_id: 'C3',
    claim: 'An unrelated claim should not appear in this panel.',
    citation_ids: ['E9'],
    status: 'UNSUPPORTED',
    reason: 'It cites different evidence.',
    evidence_uids: ['other-paragraph'],
  },
]

function renderPanel(onSelectClaim = vi.fn()) {
  render(
    <EvidencePanel
      evidence={evidence}
      claims={claims}
      selectedClaim={claims[0]}
      onSelectClaim={onSelectClaim}
    />,
  )
  return onSelectClaim
}

function expectBefore(earlier: Element, later: Element) {
  expect(earlier.compareDocumentPosition(later) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
}

describe('EvidencePanel evidence-first hierarchy', () => {
  it('places judgment identity, pinpoint, paragraph, and source before claim analysis', () => {
    renderPanel()
    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    const selectedEvidence = within(panel).getByRole('heading', { name: 'E4 · Evidence' })
    const caseName = within(panel).getByRole('heading', { name: evidence.case_name })
    const court = within(panel).getByText(evidence.court!)
    const caseNumber = within(panel).getByText(evidence.case_number!)
    const judgmentDate = within(panel).getByText('26 Nov 2019')
    const pinpoint = within(panel).getByText('Page 19 · Para 20')
    const paragraph = within(panel).getByText(evidence.text)
    const source = within(panel).getByRole('link', { name: /open source judgment/i })
    const relationship = within(panel).getByRole('heading', { name: 'Why this evidence was cited' })

    expectBefore(selectedEvidence, caseName)
    expectBefore(caseName, court)
    expectBefore(court, caseNumber)
    expectBefore(caseNumber, judgmentDate)
    expectBefore(judgmentDate, pinpoint)
    expectBefore(pinpoint, paragraph)
    expectBefore(paragraph, source)
    expectBefore(source, relationship)
  })

  it('expands the selected claim and keeps other related claims compact and keyboard-selectable', async () => {
    const onSelectClaim = renderPanel()
    const user = userEvent.setup()
    const selectedToggle = screen.getByRole('button', {
      name: /current claim C1, SUPPORTED: An interested party cannot appoint/i,
    })
    const relatedToggle = screen.getByRole('button', {
      name: /select claim C2, PARTIAL: The rule also applies/i,
    })

    expect(selectedToggle).toHaveAttribute('aria-expanded', 'true')
    expect(selectedToggle).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText(claims[0].reason)).toBeVisible()
    expect(relatedToggle).toHaveAttribute('aria-expanded', 'false')
    expect(relatedToggle).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByText(claims[1].reason)).not.toBeVisible()
    expect(screen.queryByText(claims[2].claim)).not.toBeInTheDocument()

    relatedToggle.focus()
    await user.keyboard('{Enter}')
    expect(onSelectClaim).toHaveBeenCalledWith('C2')
  })

  it('uses the evidence source URL as a real, safely opened action', () => {
    renderPanel()
    const source = screen.getByRole('link', {
      name: `Open source judgment for ${evidence.case_name} (opens in a new tab)`,
    })

    expect(source).toHaveAttribute('href', evidence.source_url)
    expect(source).toHaveAttribute('target', '_blank')
    expect(source).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('keeps complete technical provenance collapsed until explicitly opened', async () => {
    renderPanel()
    const user = userEvent.setup()
    const summary = screen.getByText('Technical provenance')
    const provenance = summary.closest('details')

    expect(provenance).not.toBeNull()
    expect(provenance).not.toHaveAttribute('open')
    expect(within(provenance!).getByText(evidence.paragraph_uid)).not.toBeVisible()

    await user.click(summary)

    expect(provenance).toHaveAttribute('open')
    expect(within(provenance!).getByText('Paragraph UID')).toBeVisible()
    expect(within(provenance!).getByText('BM25 rank')).toBeVisible()
    expect(within(provenance!).getByText('BM25 score')).toBeVisible()
    expect(within(provenance!).getByText('Dense rank')).toBeVisible()
    expect(within(provenance!).getByText('Dense score')).toBeVisible()
    expect(within(provenance!).getByText('RRF score')).toBeVisible()
    expect(within(provenance!).getByText('RRF rank')).toBeVisible()
    expect(within(provenance!).getByText('Reranker score')).toBeVisible()
    expect(within(provenance!).getByText('Retrieval rank')).toBeVisible()
  })
})
