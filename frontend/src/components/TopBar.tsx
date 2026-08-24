export function TopBar() {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">CC</span>
        <div>
          <strong>Commercial Court Research</strong>
          <span>Indian judgments · evidence-grounded workspace</span>
        </div>
      </div>
      <div className="topbar__status">
        <span className="corpus-pill">100 judgments</span>
        <span className="preview-status"><i aria-hidden="true" /> Static preview</span>
      </div>
    </header>
  )
}
