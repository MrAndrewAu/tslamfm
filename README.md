# tslamfm

**TSLA multi-factor model** — an honest, transparent statistical model for Tesla's stock price.

Live: [tslamfm.com](https://tslamfm.com)

## What it is

`v6-canonical` is a 6-year weekly OLS regression of `log(TSLA)` on five exogenous market factors plus eleven event dummies. It explains ~83% of the in-sample variance and tracks TSLA's *direction* with ~0.78 OOS correlation. It does **not** know anything about Musk's politics, robotaxi delays, or product news — that information lives in the *gap* between actual price and model fair value.

## The model

```
log(TSLA) = α
          + β₁·log(QQQ)         tech-market beta
          + β₂·log(DXY)         dollar regime
          + β₃·log(VIX)         volatility premium
          + β₄·NVDA_excess      NVDA rotation (residualized vs QQQ)
          + β₅·ARKK_excess      speculative-growth rotation
          + Σ event dummies     8-week impulse windows
          + ε
```

## Stack

- **Frontend**: Vite + React 18 + TypeScript + Tailwind + Recharts
- **Hosting**: AWS Amplify (static SPA)
- **Live quotes**: AWS Lambda Function URL proxying Yahoo Finance (no API key required)
- **Model data**: Python script regenerates `public/data/model.json` and `public/data/history.json` from yfinance

## Develop

```bash
npm install
npm run dev
```

Live quotes are skipped in local dev unless you set `VITE_QUOTES_URL` in `.env.local`. The static snapshot still renders.

## Rebuild model data

Requires Python 3.11+ with `numpy pandas yfinance scipy tabulate`.

```bash
npm run rebuild-data
```

## Deploy

### 1. Lambda Function URL (live quotes)

Create a Node.js 20 Lambda. Paste the contents of `lambda/quotes.mjs` as `index.mjs`. Configure:

- **Function URL**: enabled, Auth type **NONE**
- **CORS**: allow origin `https://tslamfm.com` (and `http://localhost:5173` for dev)
- **Timeout**: 10s

Copy the resulting URL (e.g. `https://abc123.lambda-url.us-east-1.on.aws/`).

### 2. Amplify Hosting (frontend)

1. Connect this repo in the AWS Amplify console.
2. Amplify auto-detects `amplify.yml`.
3. In **Environment variables**, set:
   - `VITE_QUOTES_URL = https://abc123.lambda-url.us-east-1.on.aws/`
4. Add the custom domain `tslamfm.com` in **Domain management**.
5. Deploy.

## License & disclaimer

Open methodology, MIT-licensed code. **Not investment advice.** See the in-app methodology panel for what we tested and rejected (lag-cheats, EPS, deliveries, short interest, raw volume, sentiment proxies).
