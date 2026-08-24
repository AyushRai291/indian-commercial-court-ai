interface SidebarProps {
  examples: string[]
  onNewResearch: () => void
  onSelectExample: (query: string) => void
}

export function Sidebar({ examples, onNewResearch, onSelectExample }: SidebarProps) {
  return (
    <nav className="sidebar" aria-label="Research workspace navigation">
      <button className="new-research" type="button" onClick={onNewResearch}>
        <span aria-hidden="true">＋</span> New research
      </button>

      <div className="nav-group">
        <span className="nav-label">Workspace</span>
        <a className="nav-item nav-item--active" href="#research" aria-current="page">
          <span className="nav-glyph" aria-hidden="true">⌕</span> Legal research
        </a>
      </div>

      <div className="recent-research" aria-label="Curated research questions">
        <span className="nav-label">Try a question</span>
        {examples.slice(0, 4).map((example, index) => (
          <button
            key={example}
            type="button"
            title={example}
            onClick={() => onSelectExample(example)}
          >
            <span className="preset-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
            <span>{example}</span>
          </button>
        ))}
      </div>

      <div className="sidebar-note">
        <span>Live research workflow</span>
        <p>Retrieved judgments, grounded answers, and citations verified claim by claim.</p>
      </div>
    </nav>
  )
}
