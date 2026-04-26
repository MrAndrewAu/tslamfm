import type { LiveQuotes } from '../types'

/**
 * Fetch live quotes from the AWS Lambda Function URL.
 * Configure VITE_QUOTES_URL at build time (Amplify env var). If unset,
 * the live fetch is skipped and the UI falls back to the static snapshot.
 */
const QUOTES_URL = import.meta.env.VITE_QUOTES_URL as string | undefined

export async function fetchLiveQuotes(): Promise<LiveQuotes | null> {
  if (!QUOTES_URL) return null
  try {
    const r = await fetch(QUOTES_URL, { cache: 'no-store' })
    if (!r.ok) return null
    return (await r.json()) as LiveQuotes
  } catch {
    return null
  }
}
