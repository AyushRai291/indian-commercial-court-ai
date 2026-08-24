import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  arbitratorResearch,
  commercialWisdomResearch,
  mockResearchResults,
} from '../mocks/answerResponses'
import type { SearchFilters, VerifiedResearchFixture, WorkspaceView } from '../types'
import { EvidenceList } from './EvidenceList'
import { EvidencePanel } from './EvidencePanel'
import { GroundedAnswer } from './GroundedAnswer'
import { LoadingState } from './LoadingState'
import { SearchBar } from './SearchBar'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { WorkspaceState } from './WorkspaceState'

const defaultFilters: SearchFilters = {
  court: 'Supreme Court of India',
  year: '',
  caseNumber: '',
}

export function ResearchWorkspace() {
  const [query, setQuery] = useState(arbitratorResearch.answer.query)
  const [filters, setFilters] = useState<SearchFilters>(defaultFilters)
  const [view, setView] = useState<WorkspaceView>('result')
  const [activeResult, setActiveResult] = useState<VerifiedResearchFixture>(arbitratorResearch)
  const [selectedClaimId, setSelectedClaimId] = useState('C1')
  const [selectedEvidenceId, setSelectedEvidenceId] = useState('E1')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [loadingStage, setLoadingStage] = useState(0)
  const timers = useRef<Array<ReturnType<typeof setTimeout>>>([])

  useEffect(() => () => timers.current.forEach(clearTimeout), [])

  const selectedEvidence = useMemo(
    () =>
      view === 'result'
        ? activeResult.answer.evidence.find((item) => item.evidence_id === selectedEvidenceId) ??
          null
        : null,
    [activeResult, selectedEvidenceId, view],
  )

  const selectedClaim = useMemo(
    () =>
      view === 'result'
        ? activeResult.verification.claims.find((claim) => claim.claim_id === selectedClaimId) ??
          null
        : null,
    [activeResult, selectedClaimId, view],
  )

  const highlightedEvidenceIds = selectedClaim?.citation_ids ?? []

  function clearTimers() {
    timers.current.forEach(clearTimeout)
    timers.current = []
  }

  function showExample(result: VerifiedResearchFixture) {
    clearTimers()
    const firstClaim = result.verification.claims[0]
    setQuery(result.answer.query)
    setActiveResult(result)
    setSelectedClaimId(firstClaim?.claim_id ?? '')
    setSelectedEvidenceId(firstClaim?.citation_ids[0] ?? result.answer.evidence[0]?.evidence_id ?? '')
    setValidationError(null)
    setView('result')
  }

  function startNewResearch() {
    clearTimers()
    setQuery('')
    setFilters(defaultFilters)
    setValidationError(null)
    setSelectedClaimId('')
    setSelectedEvidenceId('')
    setView('empty')
  }

  function submitResearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const normalizedQuery = query.trim()
    if (!normalizedQuery) {
      setValidationError('Enter a legal question before searching the corpus.')
      return
    }

    clearTimers()
    setValidationError(null)
    setLoadingStage(0)
    setView('loading')

    timers.current = [
      setTimeout(() => setLoadingStage(1), 350),
      setTimeout(() => setLoadingStage(2), 700),
      setTimeout(() => setLoadingStage(3), 950),
      setTimeout(() => {
        const lowered = normalizedQuery.toLowerCase()
        const result = lowered.includes('commercial wisdom') || lowered.includes('creditor')
          ? commercialWisdomResearch
          : lowered.includes('arbitrator') || lowered.includes('appointment')
            ? arbitratorResearch
            : null

        if (!result) {
          setSelectedClaimId('')
          setSelectedEvidenceId('')
          setView('no-results')
          return
        }

        const firstClaim = result.verification.claims[0]
        setActiveResult(result)
        setSelectedClaimId(firstClaim?.claim_id ?? '')
        setSelectedEvidenceId(firstClaim?.citation_ids[0] ?? result.answer.evidence[0]?.evidence_id ?? '')
        setView('result')
      }, 1250),
    ]
  }

  function selectClaim(claimId: string) {
    const claim = activeResult.verification.claims.find((item) => item.claim_id === claimId)
    if (!claim) return
    setSelectedClaimId(claimId)
    setSelectedEvidenceId(claim.citation_ids[0] ?? '')
  }

  function selectClaimEvidence(claimId: string, evidenceId: string) {
    setSelectedClaimId(claimId)
    setSelectedEvidenceId(evidenceId)
  }

  function selectEvidence(evidenceId: string) {
    const currentClaim = activeResult.verification.claims.find(
      (claim) => claim.claim_id === selectedClaimId,
    )
    const relatedClaim = currentClaim?.citation_ids.includes(evidenceId)
      ? currentClaim
      : activeResult.verification.claims.find((claim) => claim.citation_ids.includes(evidenceId))

    if (relatedClaim) setSelectedClaimId(relatedClaim.claim_id)
    setSelectedEvidenceId(evidenceId)
  }

  return (
    <div className="workspace-shell">
      <TopBar />
      <Sidebar
        examples={mockResearchResults}
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
          <span className="mock-label">Static demo data</span>
        </div>

        <SearchBar
          query={query}
          filters={filters}
          validationError={validationError}
          onQueryChange={setQuery}
          onFiltersChange={setFilters}
          onSubmit={submitResearch}
        />

        {view === 'loading' ? <LoadingState activeStage={loadingStage} /> : null}
        {view === 'result' ? (
          <>
            <GroundedAnswer
              response={activeResult.answer}
              verification={activeResult.verification}
              selectedClaimId={selectedClaimId}
              selectedEvidenceId={selectedEvidenceId}
              onSelectClaim={selectClaim}
              onSelectEvidence={selectClaimEvidence}
            />
            <EvidenceList
              evidence={activeResult.answer.evidence}
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
        claims={activeResult.verification.claims}
        selectedClaim={selectedClaim}
        onSelectClaim={selectClaim}
      />
    </div>
  )
}
