import type { LiveQuotes } from '../types'

/**
 * Fetch live quotes via the /api/quotes Vercel edge function.
 * Returns null on failure so the UI can fall back to the static snapshot.
 */
export async function fetchLiveQuotes(): Promise<LiveQuotes | null> {
  try {
    const r = await fetch('/api/quotes', { cache: 'no-store' })
    if (!r.ok) return null
    const data = await r.json()
    return data as LiveQuotes
  } catch {
    return null
  }
}
