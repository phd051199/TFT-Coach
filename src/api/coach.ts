import type { ApiCoachResult, CoachHistoryHint, CoachInput } from '../types'

export type ApiHealthResult = {
  ok: boolean
  data: ApiCoachResult['data']
  model: ApiCoachResult['model']
}

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

export async function requestHealth(signal?: AbortSignal): Promise<ApiHealthResult> {
  const response = await fetch('/api/health', { signal })
  if (!response.ok) throw new Error(`Health API ${response.status}`)
  return response.json() as Promise<ApiHealthResult>
}
