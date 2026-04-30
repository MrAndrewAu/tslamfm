import { useEffect, useState } from 'react'
import type { Model, HistoryRow, LiveQuotes } from './types'
import { computeFairValue } from './lib/model'
import { fetchLiveQuotes } from './lib/yahoo'
import Header from './components/Header'
import HistoryChart from './components/HistoryChart'
import FactorBars from './components/FactorBars'
import ModelStats from './components/ModelStats'
import Methodology from './components/Methodology'
import Footer from './components/Footer'

export default function App() {
  const [model, setModel]     = useState<Model | null>(null)
  const [history, setHistory] = useState<HistoryRow[] | null>(null)
  const [live, setLive]       = useState<LiveQuotes | null>(null)
  const [error, setError]     = useState<string | null>(null)

  // Load static model + history on mount
  useEffect(() => {
    Promise.all([
      fetch('/data/model.json').then(r => r.json()),
      fetch('/data/history.json').then(r => r.json()),
    ])
      .then(([m, h]) => { setModel(m); setHistory(h) })
      .catch(e => setError(String(e)))
  }, [])

  // Try to fetch live quotes; auto-refresh every 60s
  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      const q = await fetchLiveQuotes()
      if (!cancelled && q) setLive(q)
    }
    tick()
    const id = setInterval(tick, 300_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center text-bad">
        Failed to load model: {error}
      </div>
    )
  }
  if (!model || !history) {
    return (
      <div className="min-h-screen flex items-center justify-center muted text-sm">
        Loading model…
      </div>
    )
  }

  // Decide which snapshot to display: live (if available) or static current
  const snapshot = (() => {
    // Use the latest adaptive sigma_t for live bands; fall back to constant
    // residual sigma for older model.json files.
    const sigmaForBands = model.stats.sigma_t_latest ?? model.stats.sigma_resid_log
    // Empirical 10/90th-percentile offsets, scaled by sigma_t / sigma_IS so
    // live bands track the current vol regime (matches per-row history scaling).
    const q10 = model.stats.q10_log
    const q90 = model.stats.q90_log
    const sigmaIS = model.stats.sigma_IS_log ?? model.stats.sigma_resid_log
    const qScale = (model.stats.sigma_t_latest ?? sigmaIS) / Math.max(sigmaIS, 1e-8)
    if (live) {
      const r = computeFairValue(model, live)
      return {
        actual: live.TSLA,
        fair:   r.fair,
        low:    q10 != null ? r.fair * Math.exp(q10 * qScale) : r.fair * Math.exp(-sigmaForBands),
        high:   q90 != null ? r.fair * Math.exp(q90 * qScale) : r.fair * Math.exp( sigmaForBands),
        contributions: r.contribution_dollars,
        underlyings: {
          TSLA: live.TSLA, QQQ: live.QQQ, DXY: live.DXY,
          VIX: live.VIX, NVDA: live.NVDA, ARKK: live.ARKK,
          RBOB: model.current.underlyings.RBOB,
          IEF:  model.current.underlyings.IEF,
          SHY:  model.current.underlyings.SHY,
        },
        asOf: new Date(live.fetched_at).toLocaleString('en-US', {
          dateStyle: 'medium', timeStyle: 'short',
        }),
        isLive: true,
      }
    }
    // If the last history row is a partial-week snapshot, use it for the gauge
    // so today's close is reflected even without a live Lambda feed.
    const lastRow = history[history.length - 1]
    if (lastRow?.partial_week) {
      return {
        actual: lastRow.actual,
        fair:   lastRow.fitted,
        low:    lastRow.low_q  ?? lastRow.low,
        high:   lastRow.high_q ?? lastRow.high,
        contributions: model.current.contribution_dollars,
        underlyings: model.current.underlyings,
        asOf: lastRow.date,
        isLive: false,
      }
    }
    return {
      actual: model.current.tsla_actual,
      fair:   model.current.tsla_fair,
      low:    model.current.q_low  ?? model.current.sigma_low,
      high:   model.current.q_high ?? model.current.sigma_high,
      contributions: model.current.contribution_dollars,
      underlyings: model.current.underlyings,
      asOf: model.current.date,
      isLive: false,
    }
  })()

  return (
    <div className="min-h-screen flex flex-col">
      <Header modelVersion={model.version} generatedAt={model.generated_at} />

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-8 space-y-6">
        <HistoryChart
          history={history}
          model={model}
          snapshot={{
            actual: snapshot.actual,
            fair: snapshot.fair,
            low: snapshot.low,
            high: snapshot.high,
            asOf: snapshot.asOf,
            isLive: snapshot.isLive,
          }}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <FactorBars
            contributions={snapshot.contributions}
            fair={snapshot.fair}
            underlyings={snapshot.underlyings}
          />
          <ModelStats stats={model.stats} nWeeks={model.window.n_weeks} />
        </div>

        {model.current.active_events.length > 0 && (
          <div className="panel p-6">
            <div className="text-sm font-semibold mb-3">Currently active event windows</div>
            <ul className="space-y-1 text-sm">
              {model.current.active_events.map(e => (
                <li key={e.name} className="flex items-center justify-between border-b border-line/50 py-1">
                  <span>{e.label}</span>
                  <span className="mono text-xs muted">started {e.start}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <Methodology model={model} />
      </main>

      <Footer />
    </div>
  )
}
