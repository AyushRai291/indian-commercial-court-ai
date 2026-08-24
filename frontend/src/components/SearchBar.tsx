import type { FormEvent } from 'react'
import type { SearchFilters } from '../types'
import { FilterBar } from './FilterBar'

interface SearchBarProps {
  query: string
  filters: SearchFilters
  validationError: string | null
  isLoading: boolean
  onQueryChange: (query: string) => void
  onFiltersChange: (filters: SearchFilters) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

export function SearchBar({
  query,
  filters,
  validationError,
  isLoading,
  onQueryChange,
  onFiltersChange,
  onSubmit,
}: SearchBarProps) {
  return (
    <form
      className="query-card"
      aria-busy={isLoading}
      aria-label="Legal research query"
      onSubmit={onSubmit}
    >
      <label className="query-label" htmlFor="legal-query">Legal question</label>
      <textarea
        id="legal-query"
        rows={3}
        value={query}
        aria-invalid={Boolean(validationError)}
        aria-describedby={validationError ? 'query-error' : 'query-guidance'}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder="Ask a focused question about an Indian commercial-law issue…"
        spellCheck="true"
      />
      {validationError ? (
        <p className="query-error" id="query-error" role="alert">{validationError}</p>
      ) : (
        <p className="query-guidance" id="query-guidance">
          Frame a specific issue, statutory provision, or case proposition.
        </p>
      )}
      <FilterBar filters={filters} onChange={onFiltersChange} />
      <div className="query-card__footer">
        <span>Hybrid retrieval · grounded answer · verified citations</span>
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Reviewing judgments…' : 'Research judgments'}{' '}
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </form>
  )
}
