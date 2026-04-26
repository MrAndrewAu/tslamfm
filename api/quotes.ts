// Vercel Edge Function: /api/quotes
// Fetches live quotes for TSLA, QQQ, DX-Y.NYB, ^VIX, NVDA, ARKK from Yahoo Finance.
// No API key required. Free Hobby tier on Vercel covers expected traffic comfortably.

export const config = { runtime: 'edge' }

const SYMBOLS: Record<string, string> = {
  TSLA: 'TSLA',
  QQQ:  'QQQ',
  DXY:  'DX-Y.NYB',
  VIX:  '%5EVIX',     // ^VIX url-encoded
  NVDA: 'NVDA',
  ARKK: 'ARKK',
}

const UA = 'Mozilla/5.0 (compatible; tslamfm/0.1)'

async function fetchOne(yahooSymbol: string): Promise<number | null> {
  const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${yahooSymbol}`
  try {
    const r = await fetch(url, { headers: { 'User-Agent': UA } })
    if (!r.ok) return null
    const j = await r.json()
    const item = j?.quoteResponse?.result?.[0]
    const price = item?.regularMarketPrice ?? item?.postMarketPrice ?? item?.preMarketPrice
    return typeof price === 'number' ? price : null
  } catch { return null }
}

export default async function handler(): Promise<Response> {
  const entries = await Promise.all(
    Object.entries(SYMBOLS).map(async ([k, s]) => [k, await fetchOne(s)] as const)
  )
  const out: Record<string, number | string> = { fetched_at: new Date().toISOString() }
  for (const [k, v] of entries) {
    if (v === null) {
      return new Response(
        JSON.stringify({ error: `failed to fetch ${k}` }),
        { status: 502, headers: { 'content-type': 'application/json' } }
      )
    }
    out[k] = v
  }
  return new Response(JSON.stringify(out), {
    status: 200,
    headers: {
      'content-type': 'application/json',
      'cache-control': 'public, max-age=60, s-maxage=60',
    },
  })
}
