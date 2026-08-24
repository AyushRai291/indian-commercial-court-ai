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
          <span className="nav-glyph" aria-hidden="true">⌕</span> Search
        </a>
        <a className="nav-item" href="#recent">
          <span className="nav-glyph" aria-hidden="true">◷</span> Recent queries
        </a>
        <a className="nav-item" href="#corpus">
          <span className="nav-glyph" aria-hidden="true">▤</span> Corpus
        </a>
      </div>

      <div className="recent-research" id="recent">
        <span className="nav-label">Demo questions</span>
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onSelectExample(example)}
          >
            {example}
          </button>
        ))}
      </div>

      <div className="sidebar-note">
        <span>Day 16 live research</span>
        <p>Each query runs retrieval, grounded generation, and citation verification.</p>
      </div>
    </nav>
  )
}
