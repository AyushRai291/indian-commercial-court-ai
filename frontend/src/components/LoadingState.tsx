const stages = [
  'Searching judgments',
  'Ranking evidence',
  'Generating grounded answer',
  'Verifying citations',
]

export function LoadingState({ activeStage }: { activeStage: number }) {
  return (
    <section className="loading-state" aria-live="polite" aria-label="Research in progress">
      <div className="loading-state__heading">
        <span className="eyebrow">Research in progress</span>
        <h2>Working through the judgment corpus</h2>
        <p>One non-streaming request runs the complete research pipeline.</p>
      </div>
      <ol>
        {stages.map((stage, index) => (
          <li
            className={index < activeStage ? 'is-complete' : index === activeStage ? 'is-active' : ''}
            key={stage}
          >
            <span aria-hidden="true">{index < activeStage ? '✓' : index + 1}</span>
            <div><strong>{stage}</strong><i /></div>
          </li>
        ))}
      </ol>
    </section>
  )
}
