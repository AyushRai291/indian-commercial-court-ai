import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ResearchRequest, ResearchResponse } from '../types'
import { runResearch } from './research'

const request: ResearchRequest = {
  query: 'What rule governs the appointment?',
  top_k: 10,
  filters: {},
}

const baseResponse: ResearchResponse = {
  query: request.query,
  answer: 'The corpus returned no usable evidence.',
  used_evidence_ids: [],
  evidence: [],
  claims: [],
  verification_summary: { supported: 0, partial: 0, unsupported: 0 },
  verification_state: 'not_run',
  verification_error: null,
  latency: {
    retrieval_ms: 3,
    generation_ms: 0,
    verification_ms: 0,
    total_ms: 3,
  },
}

function responseWith(payload: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: vi.fn().mockResolvedValue(payload),
  } as unknown as Response
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('runResearch response validation', () => {
  it.each([
    ['complete', null],
    ['not_run', null],
    ['unavailable', { supported: 0, partial: 0, unsupported: 0 }],
  ] as const)(
    'rejects verification_summary=%j for verification_state=%s',
    async (verificationState, verificationSummary) => {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue(
          responseWith({
            ...baseResponse,
            verification_state: verificationState,
            verification_summary: verificationSummary,
          }),
        ),
      )

      await expect(runResearch(request)).rejects.toMatchObject({
        name: 'ResearchApiError',
        kind: 'invalid-response',
      })
    },
  )

  it('accepts a null summary only for unavailable verification', async () => {
    const unavailableResponse: ResearchResponse = {
      ...baseResponse,
      verification_summary: null,
      verification_state: 'unavailable',
      verification_error: 'Verification service unavailable.',
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(responseWith(unavailableResponse)))

    await expect(runResearch(request)).resolves.toEqual(unavailableResponse)
  })

  it.each([
    ['complete', 'Unexpected error.', [], { supported: 0, partial: 0, unsupported: 0 }],
    ['not_run', null, [], { supported: 1, partial: 0, unsupported: 0 }],
    ['unavailable', null, [], null],
    ['unavailable', 'Verification unavailable.', [{ claim_id: 'C1' }], null],
  ] as const)(
    'rejects an inconsistent %s verification state',
    async (verificationState, verificationError, claims, verificationSummary) => {
      vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue(
          responseWith({
            ...baseResponse,
            verification_state: verificationState,
            verification_error: verificationError,
            verification_summary: verificationSummary,
            claims,
          }),
        ),
      )

      await expect(runResearch(request)).rejects.toMatchObject({
        name: 'ResearchApiError',
        kind: 'invalid-response',
      })
    },
  )
})
