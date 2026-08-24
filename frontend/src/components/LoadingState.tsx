const stages = [
  'Searching judgments',
  'Ranking evidence',
  'Generating grounded answer',
  'Verifying citations',
]

export function LoadingState() {
  return (
    <section className="loading-state" aria-live="polite" aria-label="Research in progress">
      <div className="loading-state__heading">
        <span className="eyebrow">Research in progress</span>
        <h2>Reviewing the judgment corpus</h2>
        <p>Retrieving the strongest passages and validating every citation.</p>
      </div>
      <ol>
        {stages.map((stage, index) => (
          <li key={stage}>
            <span aria-hidden="true">{index + 1}</span>
            <div><strong>{stage}</strong><i /></div>
          </li>
        ))}
      </ol>
    </section>
  )
}
