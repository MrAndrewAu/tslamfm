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
- **Live quotes**: Vercel Edge Function at `/api/quotes` proxies Yahoo Finance (no API key required)
- **Model data**: Python script regenerates `public/data/model.json` and `public/data/history.json` from yfinance

## Develop

```bash
npm install
npm run dev
```

## Rebuild model data

Requires Python 3.11+ with `numpy pandas yfinance scipy tabulate`.

```bash
npm run rebuild-data
```

This refits the model on the latest weekly data and writes fresh JSON.

## Deploy

Push to a Vercel project. The `/api/quotes` Edge Function is deployed automatically.

## License & disclaimer

Open methodology, MIT-licensed code. **Not investment advice.** See the in-app methodology panel for what we tested and rejected (lag-cheats, EPS, deliveries, short interest, raw volume, sentiment proxies).
