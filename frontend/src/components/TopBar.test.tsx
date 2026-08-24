import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { TopBar } from './TopBar'

describe('TopBar research status', () => {
  it.each([
    ['ready', 'Research workspace · Ready'],
    ['working', 'Research workflow · Working'],
    ['live', 'Gemini research · Live'],
    ['degraded', 'Research service · Degraded'],
    ['offline', 'Research service · Offline'],
  ] as const)('renders the %s state without claiming static connectivity', (status, copy) => {
    render(<TopBar status={status} />)
    expect(screen.getByRole('status')).toHaveTextContent(copy)
  })
})
