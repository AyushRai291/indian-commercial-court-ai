import type { FormEvent } from 'react'
import type { SearchFilters } from '../types'
import { FilterBar } from './FilterBar'

interface SearchBarProps {
  query: string
  filters: SearchFilters
  validationError: string | null
  onQueryChange: (query: string) => void
  onFiltersChange: (filters: SearchFilters) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

export function SearchBar({
  query,
  filters,
  validationError,
  onQueryChange,
  onFiltersChange,
  onSubmit,
}: SearchBarProps) {
  return (
    <form className="query-card" aria-label="Legal research query" onSubmit={onSubmit}>
      <label className="query-label" htmlFor="legal-query">Legal question</label>
      <textarea
        id="legal-query"
        rows={3}
        value={query}
        aria-invalid={Boolean(validationError)}
        aria-describedby={validationError ? 'query-error' : 'query-guidance'}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder="Ask a focused question about the commercial court corpus…"
      />
      {validationError ? (
        <p className="query-error" id="query-error" role="alert">{validationError}</p>
      ) : (
        <p className="query-guidance" id="query-guidance">
          Use a specific legal issue, provision, or case proposition.
        </p>
      )}
      <FilterBar filters={filters} onChange={onFiltersChange} />
      <div className="query-card__footer">
        <span>Reranked evidence · grounded answer</span>
        <button type="submit">Research judgments <span aria-hidden="true">→</span></button>
      </div>
    </form>
  )
}
