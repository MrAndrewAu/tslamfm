// AWS Lambda handler for the live quote proxy.
// Deploy as a Lambda with a Function URL (auth: NONE, CORS: allow your domain).
// Runtime: Node.js 20.x (or newer). No bundler needed; uses built-in fetch.
//
// After deploy, set the Amplify environment variable:
//   VITE_QUOTES_URL = https://<your-fn-id>.lambda-url.<region>.on.aws/
//
// The frontend will use that URL; if unset (or local dev), it falls back to /api/quotes
// which works behind a local proxy or via `amplify mock` if you wire it up.

const SYMBOLS = {
  TSLA: 'TSLA',
  QQQ:  'QQQ',
  DXY:  'DX-Y.NYB',
  VIX:  '%5EVIX',
  NVDA: 'NVDA',
  ARKK: 'ARKK',
};

const UA = 'Mozilla/5.0 (compatible; tslamfm/0.1)';

async function fetchOne(yahooSymbol) {
  const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${yahooSymbol}`;
  try {
    const r = await fetch(url, { headers: { 'User-Agent': UA } });
    if (!r.ok) return null;
    const j = await r.json();
    const item = j?.quoteResponse?.result?.[0];
    const price = item?.regularMarketPrice ?? item?.postMarketPrice ?? item?.preMarketPrice;
    return typeof price === 'number' ? price : null;
  } catch { return null; }
}

export const handler = async () => {
  const entries = await Promise.all(
    Object.entries(SYMBOLS).map(async ([k, s]) => [k, await fetchOne(s)])
  );

  const headers = {
    'content-type': 'application/json',
    'cache-control': 'public, max-age=300, s-maxage=300',
    'access-control-allow-origin': '*',
    'access-control-allow-methods': 'GET, OPTIONS',
  };

  for (const [k, v] of entries) {
    if (v === null) {
      return {
        statusCode: 502,
        headers,
        body: JSON.stringify({ error: `failed to fetch ${k}` }),
      };
    }
  }

  const out = { fetched_at: new Date().toISOString() };
  for (const [k, v] of entries) out[k] = v;

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify(out),
  };
};
