import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ResearchResponse } from '../types'
import { ResearchWorkspace } from './ResearchWorkspace'
import { WorkspaceState } from './WorkspaceState'

const evidence: ResearchResponse['evidence'] = [
  {
    evidence_id: 'E1',
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
  },
  {
    evidence_id: 'E2',
    paragraph_uid: 'trf-para-54',
    case_id: 102,
    case_name: 'TRF Limited v. Energo Engineering Projects Limited',
    case_number: 'Civil Appeal No. 5306 of 2017',
    court: 'Supreme Court of India',
    judgment_date: '2017-07-03',
    page_number: 33,
    paragraph_number: 54,
    source_url: 'https://example.test/trf.pdf',
    text: 'Once the named arbitrator becomes ineligible, the power to nominate also fails.',
    bm25_rank: 1,
    bm25_score: 15.6,
    dense_rank: 3,
    dense_score: 0.84,
    rrf_score: 0.029,
    hybrid_rank: 2,
    cross_encoder_score: 8.1,
    reranked_rank: 2,
  },
]

const successfulResearch: ResearchResponse = {
  query: 'Can an ineligible arbitrator nominate another person as arbitrator?',
  answer:
    'An ineligible arbitrator cannot nominate another arbitrator [E1]. The nomination power also fails with the office [E2].',
  used_evidence_ids: ['E1', 'E2'],
  evidence,
  claims: [
    {
      claim_id: 'C1',
      claim: 'An ineligible arbitrator cannot nominate another arbitrator.',
      citation_ids: ['E1', 'E2'],
      status: 'SUPPORTED',
      reason: 'Both cited paragraphs directly support the appointment rule.',
      evidence_uids: ['perkins-para-20', 'trf-para-54'],
    },
    {
      claim_id: 'C2',
      claim: 'Every prior appointment automatically becomes void.',
      citation_ids: [],
      status: 'UNSUPPORTED',
      reason: 'No evidence citation was attached to this broader proposition.',
      evidence_uids: [],
    },
  ],
  verification_summary: { supported: 1, partial: 0, unsupported: 1 },
  verification_state: 'complete',
  verification_error: null,
  latency: {
    retrieval_ms: 21.5,
    generation_ms: 302.4,
    verification_ms: 118.2,
    total_ms: 442.1,
  },
}

function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response
}

function stubResearchResponse(payload: unknown = successfulResearch, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload, status))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

async function submitQuestion(question = successfulResearch.query) {
  const user = userEvent.setup()
  const query = screen.getByLabelText('Legal question')
  await user.clear(query)
  await user.type(query, question)
  await user.click(screen.getByRole('button', { name: /research judgments/i }))
  return user
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ResearchWorkspace live API boundary', () => {
  it('starts empty and demo questions only populate the live query', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    expect(screen.getByRole('heading', { name: /start with a focused legal question/i })).toBeInTheDocument()
    await user.click(
      screen.getByRole('button', { name: /can an ineligible arbitrator nominate/i }),
    )

    expect(screen.getByLabelText('Legal question')).toHaveValue(successfulResearch.query)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(screen.queryByRole('heading', { name: 'Research position' })).not.toBeInTheDocument()
  })

  it('posts a trimmed query, top_k, and cleaned optional filters', async () => {
    const fetchMock = stubResearchResponse()
    const user = userEvent.setup()
    render(<ResearchWorkspace />)

    await user.type(screen.getByLabelText('Legal question'), `  ${successfulResearch.query}  `)
    await user.selectOptions(screen.getByLabelText('Court'), 'Supreme Court of India')
    await user.type(screen.getByLabelText('Year'), '2019')
    await user.type(screen.getByLabelText('Case number'), '  AA 32/2019  ')
    await user.click(screen.getByRole('button', { name: /research judgments/i }))
    expect(await screen.findByRole('heading', { name: 'Research position' })).toBeInTheDocument()

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, request] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('http://127.0.0.1:8000/research')
    expect(request.method).toBe('POST')
    expect(request.headers).toEqual({
      Accept: 'application/json',
      'Content-Type': 'application/json',
    })
    expect(JSON.parse(request.body as string)).toEqual({
      query: successfulResearch.query,
      top_k: 10,
      filters: {
        court: 'Supreme Court of India',
        year: 2019,
        case_number: 'AA 32/2019',
      },
    })
  })

  it('omits empty optional filters from the request', async () => {
    const fetchMock = stubResearchResponse()
    render(<ResearchWorkspace />)
    await submitQuestion()
    await screen.findByRole('heading', { name: 'Research position' })

    const request = fetchMock.mock.calls[0][1] as RequestInit
    expect(JSON.parse(request.body as string)).toEqual({
      query: successfulResearch.query,
      top_k: 10,
      filters: {},
    })
  })

  it('shows restrained non-streaming stages while the request is pending', async () => {
    let resolveFetch: (response: Response) => void = () => undefined
    const fetchMock = vi.fn(
      () => new Promise<Response>((resolve) => {
        resolveFetch = resolve
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    render(<ResearchWorkspace />)

    await submitQuestion()
    expect(screen.getByRole('heading', { name: /working through the judgment corpus/i })).toBeInTheDocument()
    expect(screen.getByText('Searching judgments')).toBeInTheDocument()
    expect(screen.getByText('Ranking evidence')).toBeInTheDocument()
    expect(screen.getByText('Generating grounded answer')).toBeInTheDocument()
    expect(screen.getByText('Verifying citations')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /researching judgments/i })).toBeDisabled()

    await act(async () => {
      resolveFetch(jsonResponse(successfulResearch))
    })
    expect(await screen.findByRole('heading', { name: 'Research position' })).toBeInTheDocument()
  })

  it('renders the live answer, verification, evidence source, and nested latency', async () => {
    stubResearchResponse()
    render(<ResearchWorkspace />)
    await submitQuestion()

    const answer = (await screen.findByRole('heading', { name: 'Research position' })).closest('section')
    expect(answer).not.toBeNull()
    expect(within(answer!).getAllByText(/an ineligible arbitrator cannot nominate/i)).not.toHaveLength(0)
    expect(within(answer!).getByLabelText('Citation verification summary')).toHaveTextContent(
      '2 claims/1 supported/0 partial/1 unsupported',
    )
    expect(within(answer!).getByText(/retrieval 21.5 ms/i)).toBeInTheDocument()
    expect(within(answer!).getByText(/total 442.1 ms/i)).toBeInTheDocument()

    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'E1' })).toBeInTheDocument()
    expect(within(panel).getByRole('link', { name: /open source judgment/i })).toHaveAttribute(
      'href',
      'https://example.test/perkins.pdf',
    )
  })

  it('preserves claim to citation to evidence interaction on live results', async () => {
    stubResearchResponse()
    render(<ResearchWorkspace />)
    const user = await submitQuestion()
    const answer = (await screen.findByRole('heading', { name: 'Research position' })).closest('section')
    expect(answer).not.toBeNull()

    await user.click(within(answer!).getAllByRole('button', { name: /show evidence E2:/i })[0])
    const panel = screen.getByRole('complementary', { name: 'Selected judgment evidence' })
    expect(within(panel).getByRole('heading', { name: 'E2' })).toBeInTheDocument()
    expect(within(panel).getByText(/power to nominate also fails/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /select C2, UNSUPPORTED:/i }))
    expect(screen.getByRole('region', { name: 'C2 evidence check' })).toHaveTextContent(
      /no evidence citation was attached/i,
    )
    expect(within(panel).getByRole('heading', { name: 'No cited evidence for C2' })).toBeInTheDocument()
  })

  it('renders a zero-evidence response as an honest no-results state', async () => {
    stubResearchResponse({
      ...successfulResearch,
      answer: 'The retrieved corpus does not contain enough evidence to answer this question.',
      used_evidence_ids: [],
      evidence: [],
      claims: [],
      verification_summary: { supported: 0, partial: 0, unsupported: 0 },
      verification_state: 'not_run',
      latency: { retrieval_ms: 9, generation_ms: 0, verification_ms: 0, total_ms: 9 },
    } satisfies ResearchResponse)
    render(<ResearchWorkspace />)
    await submitQuestion('A question outside the pilot corpus')

    expect(
      await screen.findByRole('heading', { name: /corpus did not return usable evidence/i }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Research position' })).not.toBeInTheDocument()
  })

  it('keeps answer and evidence visible when verification is unavailable', async () => {
    stubResearchResponse({
      ...successfulResearch,
      claims: [],
      verification_summary: null,
      verification_state: 'unavailable',
      verification_error: 'Sensitive upstream verifier detail that must not be rendered',
      latency: { ...successfulResearch.latency, verification_ms: 0 },
    } satisfies ResearchResponse)
    render(<ResearchWorkspace />)
    await submitQuestion()

    expect(await screen.findByText('Citation verification unavailable')).toBeInTheDocument()
    expect(screen.getByText(/no verifier result has been invented/i)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Evidence considered' })).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Selected judgment evidence' })).toHaveTextContent('E1')
    expect(screen.queryByText(/sensitive upstream verifier detail/i)).not.toBeInTheDocument()
  })

  it('shows humane backend, Gemini, and verifier failure states without clearing the query', async () => {
    const fetchMock = vi.fn().mockRejectedValueOnce(new TypeError('connection refused'))
    vi.stubGlobal('fetch', fetchMock)
    render(<ResearchWorkspace />)
    await submitQuestion('Preserve this backend query')
    expect(
      await screen.findByRole('heading', { name: /judgment service could not be reached/i }),
    ).toBeInTheDocument()
    expect(screen.getByLabelText('Legal question')).toHaveValue('Preserve this backend query')

    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: 'Gemini generation provider is temporarily unavailable.' }, 503),
    )
    await submitQuestion('Preserve this Gemini query')
    expect(
      await screen.findByRole('heading', { name: /no grounded answer was prepared/i }),
    ).toBeInTheDocument()
    expect(screen.getByText(/no answer has been invented/i)).toBeInTheDocument()
    expect(screen.getByLabelText('Legal question')).toHaveValue('Preserve this Gemini query')

    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: 'Citation verification is temporarily unavailable.' }, 503),
    )
    await submitQuestion('Preserve this verification query')
    expect(
      await screen.findByRole('heading', { name: /citation verification could not be completed/i }),
    ).toBeInTheDocument()
    expect(screen.getByText(/no verification result has been invented/i)).toBeInTheDocument()
    expect(screen.getByLabelText('Legal question')).toHaveValue('Preserve this verification query')
  })

  it('validates year before making a request', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<ResearchWorkspace />)
    await user.type(screen.getByLabelText('Legal question'), successfulResearch.query)
    await user.type(screen.getByLabelText('Year'), '20x9')
    await user.click(screen.getByRole('button', { name: /research judgments/i }))

    expect(screen.getByRole('alert')).toHaveTextContent(/four digits/i)
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe('WorkspaceState', () => {
  it('renders neutral backend, generation, and verification messages', () => {
    const { rerender } = render(<WorkspaceState view="backend-error" />)
    expect(screen.getByRole('heading', { name: /judgment service could not be reached/i })).toBeInTheDocument()

    rerender(<WorkspaceState view="generation-error" />)
    expect(screen.getByRole('heading', { name: /no grounded answer was prepared/i })).toBeInTheDocument()
    expect(screen.getByText(/no answer has been invented/i)).toBeInTheDocument()

    rerender(<WorkspaceState view="verification-error" />)
    expect(screen.getByRole('heading', { name: /citation verification could not be completed/i })).toBeInTheDocument()
  })
})
