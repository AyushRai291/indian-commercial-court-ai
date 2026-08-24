import type { VerifiedResearchFixture } from '../types'

interface SidebarProps {
  examples: VerifiedResearchFixture[]
  onNewResearch: () => void
  onSelectExample: (answer: VerifiedResearchFixture) => void
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
            key={example.answer.query}
            type="button"
            onClick={() => onSelectExample(example)}
          >
            {example.answer.query}
          </button>
        ))}
      </div>

      <div className="sidebar-note">
        <span>Day 15 verifier preview</span>
        <p>Answers, claim checks, and evidence are clearly marked static demo data.</p>
      </div>
    </nav>
  )
}
