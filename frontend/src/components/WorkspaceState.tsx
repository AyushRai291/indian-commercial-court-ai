import type { WorkspaceView } from '../types'

const stateCopy: Record<Exclude<WorkspaceView, 'result' | 'loading'>, { label: string; title: string; body: string }> = {
  empty: {
    label: 'Ready for research',
    title: 'Start with a focused legal question',
    body: 'Ask about a statutory provision, appointment issue, insolvency principle, or case proposition in the pilot corpus.',
  },
  'no-results': {
    label: 'No relevant judgments',
    title: 'The corpus did not return usable evidence',
    body: 'Try a narrower commercial-law question or remove one of the judgment filters.',
  },
  'backend-error': {
    label: 'Search unavailable',
    title: 'The judgment service could not be reached',
    body: 'Your query is preserved. Try again after the research service is available.',
  },
  'generation-error': {
    label: 'Generation unavailable',
    title: 'Evidence was found, but no grounded answer was prepared',
    body: 'The evidence can still be reviewed directly. No answer has been invented or substituted.',
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
