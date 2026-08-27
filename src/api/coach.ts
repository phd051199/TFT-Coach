import type { ApiCoachResult, CoachHistoryHint, CoachInput } from '../types'

export async function requestCoach(input: CoachInput, history?: CoachHistoryHint, signal?: AbortSignal): Promise<ApiCoachResult> {
  const response = await fetch('/api/coach/recommend', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ ...input, ...(history ?? {}) }),
    signal,
  })
  if (!response.ok) throw new Error(`Coach API ${response.status}`)
  return response.json() as Promise<ApiCoachResult>
}
