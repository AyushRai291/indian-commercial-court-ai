import type { SearchFilters } from '../types'

interface FilterBarProps {
  filters: SearchFilters
  onChange: (filters: SearchFilters) => void
}

export function FilterBar({ filters, onChange }: FilterBarProps) {
  return (
    <div className="filter-bar" aria-label="Judgment filters">
      <label>
        <span>Court</span>
        <select
          value={filters.court}
          onChange={(event) => onChange({ ...filters, court: event.target.value })}
        >
          <option value="">All courts</option>
          <option value="Supreme Court of India">Supreme Court of India</option>
        </select>
      </label>
      <label>
        <span>Year</span>
        <input
          inputMode="numeric"
          placeholder="All years"
          value={filters.year}
          onChange={(event) => onChange({ ...filters, year: event.target.value })}
        />
      </label>
      <label className="filter-bar__case">
        <span>Case number</span>
        <input
          placeholder="Any case number"
          value={filters.caseNumber}
          onChange={(event) => onChange({ ...filters, caseNumber: event.target.value })}
        />
      </label>
    </div>
  )
}
