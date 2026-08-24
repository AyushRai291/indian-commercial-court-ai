import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { ResearchWorkspace } from './ResearchWorkspace'
import { WorkspaceState } from './WorkspaceState'

describe('ResearchWorkspace', () => {
  it('renders the mock grounded answer and complete selected evidence metadata', () => {
    render(<ResearchWorkspace />)

    expect(screen.getByRole('heading', { name: 'Research position' })).toBeInTheDocument()
    expect(screen.getAllByText(/power attached to that office/i)).toHaveLength(2)

    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'E1' })).toBeInTheDocument()
    expect(within(panel).getByText(/Page 33 · Para 84/i)).toBeInTheDocument()
    expect(within(panel).getByText(/e20665a8-c3ab/i)).toBeInTheDocument()
    expect(within(panel).getByRole('link', { name: /open source judgment/i })).toHaveAttribute(
      'href',
      expect.stringContaining('2017_7_409_441_EN.pdf'),
    )
  })

  it('renders claim-level badges and the exact verification summary counts', () => {
    render(<ResearchWorkspace />)

    const answer = screen.getByRole('heading', { name: 'Research position' }).closest('section')
    expect(answer).not.toBeNull()
    expect(answer!.querySelectorAll('[data-status="SUPPORTED"]')).toHaveLength(2)
    expect(answer!.querySelectorAll('[data-status="PARTIAL"]')).toHaveLength(1)
    expect(answer!.querySelectorAll('[data-status="UNSUPPORTED"]')).toHaveLength(1)

    const summary = within(answer!).getByLabelText('Citation verification summary')
    expect(summary).toHaveTextContent(
      '4 claims/2 supported/1 partial/1 unsupported',
    )
  })

  it('selects the matching evidence when an inline citation is clicked', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    const answer = screen.getByRole('heading', { name: 'Research position' }).closest('section')
    expect(answer).not.toBeNull()
    await user.click(within(answer!).getByRole('button', { name: /show evidence E2: Perkins/i }))

    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'E2' })).toBeInTheDocument()
    expect(within(panel).getByText(/interest in the outcome/i)).toBeInTheDocument()
  })

  it('updates selected evidence and verifier reason when a claim is clicked', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    await user.click(screen.getByRole('button', { name: /select C3, PARTIAL:/i }))

    const details = screen.getByRole('region', { name: 'C3 evidence check' })
    expect(within(details).getByText(/does not say that one waiver governs all future disputes/i)).toBeInTheDocument()
    expect(within(details).getByText('PARTIAL')).toBeInTheDocument()

    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'E3' })).toBeInTheDocument()
  })

  it('highlights every evidence item cited by a multi-citation claim', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    await user.click(screen.getByRole('button', { name: /select C2, SUPPORTED:/i }))

    const rankedEvidence = screen.getByRole('heading', { name: 'Evidence considered' }).closest('section')
    expect(rankedEvidence).not.toBeNull()
    expect(within(rankedEvidence!).getByRole('button', { name: /show evidence E1:.*cited by selected claim/i })).toHaveClass('evidence-row--linked')
    expect(within(rankedEvidence!).getByRole('button', { name: /show evidence E2:.*cited by selected claim/i })).toHaveClass('evidence-row--linked')
    expect(within(rankedEvidence!).getByRole('button', { name: /show evidence E3:/i })).not.toHaveClass('evidence-row--linked')

    const answer = screen.getByRole('heading', { name: 'Research position' }).closest('section')
    expect(answer).not.toBeNull()
    const selectedClaimRow = answer!.querySelector<HTMLElement>('.claim-row--selected')
    expect(selectedClaimRow).not.toBeNull()
    const e1Citation = within(selectedClaimRow!).getByRole('button', { name: /show evidence E1:/i })
    const e2Citation = within(selectedClaimRow!).getByRole('button', { name: /show evidence E2:/i })
    expect(e1Citation).toHaveAttribute('aria-pressed', 'true')
    expect(e2Citation).toHaveAttribute('aria-pressed', 'false')
    expect(e1Citation).toHaveClass('citation-token--linked')
    expect(e2Citation).toHaveClass('citation-token--linked')
  })

  it('makes an uncited unsupported claim explicit without selecting evidence', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    await user.click(screen.getByRole('button', { name: /select C4, UNSUPPORTED:/i }))

    const details = screen.getByRole('region', { name: 'C4 evidence check' })
    expect(within(details).getByText('UNSUPPORTED')).toBeInTheDocument()
    expect(within(details).getByText(/no evidence citation was attached/i)).toBeInTheDocument()
    expect(within(details).getByText('No citation attached')).toBeInTheDocument()

    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'No cited evidence for C4' })).toBeInTheDocument()
  })

  it('selects evidence from a ranked evidence card', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    const rankedEvidence = screen.getByRole('heading', { name: 'Evidence considered' }).closest('section')
    expect(rankedEvidence).not.toBeNull()
    await user.click(within(rankedEvidence!).getByRole('button', { name: /show evidence E3:/i }))

    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'E3' })).toBeInTheDocument()
    expect(within(panel).getByText(/express agreement in writing/i)).toBeInTheDocument()
  })

  it('preserves entered search and filter values', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    const query = screen.getByLabelText('Legal question')
    await user.clear(query)
    await user.type(query, 'Commercial wisdom of creditors')
    await user.selectOptions(screen.getByLabelText('Court'), '')
    await user.type(screen.getByLabelText('Year'), '2019')
    await user.type(screen.getByLabelText('Case number'), '8766')

    expect(query).toHaveValue('Commercial wisdom of creditors')
    expect(screen.getByLabelText('Court')).toHaveValue('')
    expect(screen.getByLabelText('Year')).toHaveValue('2019')
    expect(screen.getByLabelText('Case number')).toHaveValue('8766')
  })

  it('renders empty, validation, loading, and no-results states', async () => {
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    await user.click(screen.getByRole('button', { name: /new research/i }))
    expect(screen.getByRole('heading', { name: /start with a focused legal question/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /research judgments/i }))
    expect(screen.getByRole('alert')).toHaveTextContent(/enter a legal question/i)

    const query = screen.getByLabelText('Legal question')
    await user.type(query, 'A maritime insurance issue outside this pilot')
    await user.click(screen.getByRole('button', { name: /research judgments/i }))
    expect(screen.getByRole('heading', { name: /working through the judgment corpus/i })).toBeInTheDocument()

    expect(
      await screen.findByRole('heading', { name: /corpus did not return usable evidence/i }, { timeout: 2000 }),
    ).toBeInTheDocument()
  })
})

describe('WorkspaceState', () => {
  it('renders neutral backend and generation failure messages', () => {
    const { rerender } = render(<WorkspaceState view="backend-error" />)
    expect(screen.getByRole('heading', { name: /judgment service could not be reached/i })).toBeInTheDocument()

    rerender(<WorkspaceState view="generation-error" />)
    expect(screen.getByRole('heading', { name: /no grounded answer was prepared/i })).toBeInTheDocument()
    expect(screen.getByText(/no answer has been invented/i)).toBeInTheDocument()
  })
})
