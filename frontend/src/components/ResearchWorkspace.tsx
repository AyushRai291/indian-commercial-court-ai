import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { ResearchApiError, runResearch } from '../api/research'
import type {
  ResearchRequestFilters,
  ResearchResponse,
  SearchFilters,
  WorkspaceView,
} from '../types'
import { EvidenceList } from './EvidenceList'
import { EvidencePanel } from './EvidencePanel'
import { GroundedAnswer } from './GroundedAnswer'
import { LoadingState } from './LoadingState'
import { SearchBar } from './SearchBar'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { WorkspaceState } from './WorkspaceState'

const defaultFilters: SearchFilters = {
  court: '',
  year: '',
  caseNumber: '',
}

const demoQuestions = [
  'Can an ineligible arbitrator nominate another person as arbitrator?',
  'What is the scope of judicial interference with the commercial wisdom of the committee of creditors?',
]

function isAbortError(error: unknown) {
  return error instanceof Error && error.name === 'AbortError'
}

export function ResearchWorkspace() {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<SearchFilters>(defaultFilters)
  const [view, setView] = useState<WorkspaceView>('empty')
  const [activeResult, setActiveResult] = useState<ResearchResponse | null>(null)
  const [selectedClaimId, setSelectedClaimId] = useState('')
  const [selectedEvidenceId, setSelectedEvidenceId] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [loadingStage, setLoadingStage] = useState(0)
  const timers = useRef<Array<ReturnType<typeof setTimeout>>>([])
  const activeRequest = useRef<AbortController | null>(null)

  useEffect(
    () => () => {
      timers.current.forEach(clearTimeout)
      activeRequest.current?.abort()
    },
    [],
  )

  const selectedEvidence = useMemo(
    () =>
      view === 'result'
        ? activeResult?.evidence.find((item) => item.evidence_id === selectedEvidenceId) ?? null
        : null,
    [activeResult, selectedEvidenceId, view],
  )

  const selectedClaim = useMemo(
    () =>
      view === 'result'
        ? activeResult?.claims.find((claim) => claim.claim_id === selectedClaimId) ?? null
        : null,
    [activeResult, selectedClaimId, view],
  )

  const highlightedEvidenceIds = selectedClaim?.citation_ids ?? []

  function clearTimers() {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }

  function cancelResearch() {
    clearTimers()
    activeRequest.current?.abort()
    activeRequest.current = null
  }

  function showExample(exampleQuery: string) {
    cancelResearch()
    setQuery(exampleQuery)
    setActiveResult(null)
    setSelectedClaimId('')
    setSelectedEvidenceId('')
    setValidationError(null)
    setView('empty')
  }

  function startNewResearch() {
    cancelResearch()
    setQuery('')
    setFilters(defaultFilters)
    setActiveResult(null)
    setValidationError(null)
    setSelectedClaimId('')
    setSelectedEvidenceId('')
    setView('empty')
  }

  function startLoadingStages() {
    setLoadingStage(0)
    timers.current = [
      setTimeout(() => setLoadingStage(1), 750),
      setTimeout(() => setLoadingStage(2), 1_600),
      setTimeout(() => setLoadingStage(3), 2_600),
    ]
  }

  async function performResearch(normalizedQuery: string, requestFilters: ResearchRequestFilters) {
    cancelResearch()
    const controller = new AbortController()
    activeRequest.current = controller
    setValidationError(null)
    setActiveResult(null)
    setSelectedClaimId('')
    setSelectedEvidenceId('')
    setView('loading')
    startLoadingStages()

    try {
      const response = await runResearch(
        { query: normalizedQuery, top_k: 10, filters: requestFilters },
        controller.signal,
      )
      if (controller.signal.aborted || activeRequest.current !== controller) return

      clearTimers()
      if (response.evidence.length === 0) {
        setActiveResult(response)
        setView('no-results')
        return
      }

      const firstClaim = response.claims[0]
      setActiveResult(response)
      setSelectedClaimId(firstClaim?.claim_id ?? '')
      setSelectedEvidenceId(
        firstClaim?.citation_ids[0] ?? response.evidence[0]?.evidence_id ?? '',
      )
      setView('result')
    } catch (error) {
      if (controller.signal.aborted || activeRequest.current !== controller || isAbortError(error)) {
        return
      }

      clearTimers()
      setView(
        error instanceof ResearchApiError && error.kind === 'generation'
          ? 'generation-error'
          : error instanceof ResearchApiError && error.kind === 'verification'
            ? 'verification-error'
            : 'backend-error',
      )
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null
    }
  }

  function submitResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedQuery = query.trim()
    if (!normalizedQuery) {
      setValidationError('Enter a legal question before searching the corpus.')
      return
    }

    const normalizedYear = filters.year.trim()
    if (normalizedYear && !/^\d{4}$/.test(normalizedYear)) {
      setValidationError('Enter the judgment year as four digits, for example 2019.')
      return
    }

    const requestFilters: ResearchRequestFilters = {}
    if (filters.court.trim()) requestFilters.court = filters.court.trim()
    if (normalizedYear) requestFilters.year = Number(normalizedYear)
    if (filters.caseNumber.trim()) requestFilters.case_number = filters.caseNumber.trim()

    void performResearch(normalizedQuery, requestFilters)
  }

  function selectClaim(claimId: string) {
    const claim = activeResult?.claims.find((item) => item.claim_id === claimId)
    if (!claim) return
    setSelectedClaimId(claimId)
    setSelectedEvidenceId(claim.citation_ids[0] ?? '')
  }

  function selectClaimEvidence(claimId: string, evidenceId: string) {
    setSelectedClaimId(claimId)
    setSelectedEvidenceId(evidenceId)
  }

  function selectEvidence(evidenceId: string) {
    if (!activeResult) return
    const currentClaim = activeResult.claims.find(
      (claim) => claim.claim_id === selectedClaimId,
    )
    const relatedClaim = currentClaim?.citation_ids.includes(evidenceId)
      ? currentClaim
      : activeResult.claims.find((claim) => claim.citation_ids.includes(evidenceId))

    if (relatedClaim) setSelectedClaimId(relatedClaim.claim_id)
    setSelectedEvidenceId(evidenceId)
  }

  return (
    <div className="workspace-shell">
      <TopBar />
      <Sidebar
        examples={demoQuestions}
        onNewResearch={startNewResearch}
        onSelectExample={showExample}
      />

      <main className="research-main" id="research">
        <div className="research-heading">
          <div>
            <span className="eyebrow">Indian commercial law</span>
            <h1>Ask the judgment corpus</h1>
            <p>Trace every material claim to its citation, exact paragraph, and verification status.</p>
          </div>
          <span className="environment-label">Live research API</span>
        </div>

        <SearchBar
          query={query}
          filters={filters}
          validationError={validationError}
          isLoading={view === 'loading'}
          onQueryChange={setQuery}
          onFiltersChange={setFilters}
          onSubmit={submitResearch}
        />

        {view === 'loading' ? <LoadingState activeStage={loadingStage} /> : null}
        {view === 'result' && activeResult ? (
          <>
            <GroundedAnswer
              response={activeResult}
              selectedClaimId={selectedClaimId}
              selectedEvidenceId={selectedEvidenceId}
              onSelectClaim={selectClaim}
              onSelectClaimEvidence={selectClaimEvidence}
              onSelectEvidence={selectEvidence}
            />
            <EvidenceList
              evidence={activeResult.evidence}
              selectedEvidenceId={selectedEvidenceId}
              highlightedEvidenceIds={highlightedEvidenceIds}
              onSelectEvidence={selectEvidence}
            />
          </>
        ) : null}
        {view !== 'loading' && view !== 'result' ? <WorkspaceState view={view} /> : null}
      </main>

      <EvidencePanel
        evidence={selectedEvidence}
        claims={activeResult?.claims ?? []}
        selectedClaim={selectedClaim}
        onSelectClaim={selectClaim}
      />
    </div>
  )
}
