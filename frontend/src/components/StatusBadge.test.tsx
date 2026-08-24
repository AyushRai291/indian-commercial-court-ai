import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { VerificationStatus } from '../types'
import { StatusBadge } from './StatusBadge'

describe('StatusBadge', () => {
  it.each([
    ['SUPPORTED', 'S'],
    ['PARTIAL', 'P'],
    ['UNSUPPORTED', '!'],
  ] as const)('renders %s with visible text and a non-color icon', (status, icon) => {
    const { container } = render(<StatusBadge status={status as VerificationStatus} />)
    const badge = container.querySelector(`.status-badge--${status.toLowerCase()}`)
    const badgeIcon = badge?.querySelector('.status-badge__icon')

    expect(badge).toHaveTextContent(status)
    expect(badgeIcon).toHaveTextContent(icon)
    expect(badgeIcon).toHaveAttribute('aria-hidden', 'true')
  })
})
