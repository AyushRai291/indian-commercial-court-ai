import type { AnswerResponse } from '../types'

interface SidebarProps {
  examples: AnswerResponse[]
  onNewResearch: () => void
  onSelectExample: (answer: AnswerResponse) => void
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
            key={example.query}
            type="button"
            onClick={() => onSelectExample(example)}
          >
            {example.query}
          </button>
        ))}
      </div>

      <div className="sidebar-note">
        <span>Day 14 presentation shell</span>
        <p>Answers and evidence on this screen are clearly marked static demo data.</p>
      </div>
    </nav>
  )
}
