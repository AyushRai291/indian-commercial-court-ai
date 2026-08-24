import type { WorkspaceView } from '../types'

const stateCopy: Record<Exclude<WorkspaceView, 'result' | 'loading'>, { label: string; title: string; body: string }> = {
  empty: {
    label: 'Ready for research',
    title: 'Research a focused legal question',
    body: 'Ask about a statutory provision, arbitration issue, insolvency principle, or case proposition in the curated judgment corpus.',
  },
  'no-results': {
    label: 'No matching evidence',
    title: 'No usable judgment passages were found',
    body: 'Try a narrower commercial-law question, check the case details, or remove one of the filters.',
  },
  'backend-error': {
    label: 'Research service unavailable',
    title: 'The judgment corpus could not be reached',
    body: 'Your question is preserved. Please try again when the research service is available.',
  },
  'generation-error': {
    label: 'Answer unavailable',
    title: 'Evidence was found, but no grounded answer was prepared',
    body: 'The answer service may be temporarily unavailable. Your question is preserved, and no substitute answer has been invented.',
  },
  'verification-error': {
    label: 'Verification unavailable',
    title: 'The citations could not be verified',
    body: 'The grounded answer is preserved, but no verification result has been invented. Try again when the verifier is available.',
  },
}

export function WorkspaceState({ view }: { view: Exclude<WorkspaceView, 'result' | 'loading'> }) {
  const copy = stateCopy[view]
  return (
    <section className={`workspace-state workspace-state--${view}`} role="status">
      <span className="state-mark" aria-hidden="true">§</span>
      <span className="eyebrow">{copy.label}</span>
      <h2>{copy.title}</h2>
      <p>{copy.body}</p>
    </section>
  )
}
