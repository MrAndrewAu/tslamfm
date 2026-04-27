import { useMemo, useState } from 'react'
import {
  Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, ComposedChart, LineChart,
} from 'recharts'
import type { HistoryRow, Model } from '../types'
import { fmtUSD, fmtPct } from '../lib/model'

interface Props {
  history: HistoryRow[]
  model: Model
  // Current snapshot, hoisted from App so the chart panel can headline it.
  snapshot: {
    actual: number
    fair: number
    low: number
    high: number
    asOf: string
    isLive: boolean
  }
}

const RANGES = [
  { key: '1Y', months: 12 },
  { key: '3Y', months: 36 },
  { key: 'ALL', months: 9999 },
] as const

export default function HistoryChart({ history, model, snapshot }: Props) {
  const [range, setRange] = useState<'1Y' | '3Y' | 'ALL'>('ALL')
  const [showResid, setShowResid] = useState(false)

  const data = useMemo(() => {
    const slice = (() => {
      if (range === 'ALL') return history
      const cutoff = new Date()
      cutoff.setMonth(cutoff.getMonth() - RANGES.find(r => r.key === range)!.months)
      const iso = cutoff.toISOString().slice(0, 10)
      return history.filter(r => r.date >= iso)
    })()
    // Prefer asymmetric empirical quantile bands (low_q/high_q) when present;
    // fall back to EWMA ±1σ (low/high) for backward compat with old history.json.
    return slice.map(r => ({ ...r, bandLow: r.low_q ?? r.low, bandHigh: r.high_q ?? r.high }))
  }, [history, range])

  const residData = useMemo(
    () => data.map(r => ({ date: r.date, resid: ((r.actual / r.fitted - 1) * 100) })),
    [data]
  )

  const eventLines = model.events
    .filter(e => e.in_model && e.start >= (data[0]?.date ?? ''))
    .map(e => ({ date: e.start, label: e.label }))

  const { actual, fair, low, high, asOf, isLive } = snapshot
  const gap = (actual / fair - 1) * 100
  const fairColor = '#3b82f6'

  return (
    <div className="panel p-6">
      {/* Headline summary — replaces the standalone Gauge panel */}
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-xs muted uppercase tracking-wider">Fair value vs market</div>
        <div className="text-[11px] muted flex items-center gap-2">
          <span className={`inline-block w-1.5 h-1.5 rounded-full ${isLive ? 'bg-good animate-pulse' : 'bg-dim'}`} />
          {isLive ? 'live' : 'snapshot'} · {asOf}
        </div>
      </div>

      <div className="flex flex-wrap items-baseline gap-x-8 gap-y-2 mb-4">
        <div>
          <div className="text-[10px] muted uppercase tracking-wider">Market</div>
          <div className="text-2xl mono font-semibold leading-none mt-1">{fmtUSD(actual)}</div>
        </div>
        <div>
          <div className="text-[10px] muted uppercase tracking-wider">Fair value</div>
          <div className="text-2xl mono font-semibold leading-none mt-1" style={{ color: fairColor }}>{fmtUSD(fair)}</div>
        </div>
        <div>
          <div className="text-[10px] muted uppercase tracking-wider">Gap</div>
          <div className={`text-2xl mono font-semibold leading-none mt-1 ${gap < 0 ? 'text-good' : gap > 0 ? 'text-bad' : 'text-warn'}`}>
            {fmtPct(gap)}
          </div>
        </div>
        <div>
          <div className="text-[10px] muted uppercase tracking-wider">10–90% range</div>
          <div className="text-sm mono leading-none mt-1.5">{fmtUSD(low, 0)} – {fmtUSD(high, 0)}</div>
        </div>
      </div>

      <div className="flex items-center justify-between mb-3 pt-3 border-t border-line/50">
        <div className="text-[11px] muted">
          Weekly: actual price vs model fair value, 10–90% range
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowResid(s => !s)}
            className="text-[11px] px-2 py-1 rounded border border-line hover:border-dim text-dim"
          >
            {showResid ? 'show fit' : 'show residuals'}
          </button>
          <div className="flex bg-line/40 rounded p-0.5">
            {RANGES.map(r => (
              <button
                key={r.key}
                onClick={() => setRange(r.key)}
                className="text-[11px] px-2.5 py-1 rounded transition"
                style={
                  range === r.key
                    ? { background: 'rgba(59,130,246,0.2)', color: '#3b82f6' }
                    : undefined
                }
              >
                <span className={range === r.key ? '' : 'text-dim hover:text-slate-300'}>{r.key}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ width: '100%', height: 420 }}>
        {showResid ? (
          <ResponsiveContainer>
            <LineChart data={residData} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#1f2632" strokeDasharray="2 2" />
              <XAxis dataKey="date" tick={{ fill: '#7c869a', fontSize: 11 }} stroke="#1f2632" minTickGap={40} />
              <YAxis tick={{ fill: '#7c869a', fontSize: 11 }} stroke="#1f2632" tickFormatter={v => `${v.toFixed(0)}%`} />
              <ReferenceLine y={0} stroke="#7c869a" strokeDasharray="3 3" />
              <Tooltip
                contentStyle={{ background: '#11151c', border: '1px solid #1f2632', borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => `${v.toFixed(1)}%`}
              />
              <Line type="monotone" dataKey="resid" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"  stopColor="#3b82f6" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.12} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1f2632" strokeDasharray="2 2" />
              <XAxis dataKey="date" tick={{ fill: '#7c869a', fontSize: 11 }} stroke="#1f2632" minTickGap={40} />
              <YAxis tick={{ fill: '#7c869a', fontSize: 11 }} stroke="#1f2632" tickFormatter={v => `$${v}`} />
              <Tooltip
                contentStyle={{ background: '#11151c', border: '1px solid #1f2632', borderRadius: 8, fontSize: 12 }}
                formatter={(v: number) => fmtUSD(v, 0)}
                labelStyle={{ color: '#7c869a', marginBottom: 4 }}
              />
              {/* Band: high (top) and low (bottom). Stack-style fill via two areas. */}
              <Area type="monotone" dataKey="bandHigh" stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 3" strokeOpacity={0.5} fill="url(#bandFill)" />
              <Area type="monotone" dataKey="bandLow"  stroke="#3b82f6" strokeWidth={1} strokeDasharray="4 3" strokeOpacity={0.5} fill="#0b0e14" />
              <Line type="monotone" dataKey="fitted" stroke="#3b82f6" strokeWidth={2} dot={false} name="fair value" />
              <Line type="monotone" dataKey="actual" stroke="#e2e8f0" strokeWidth={1.5} dot={false} name="actual" />
              {eventLines.map(e => (
                <ReferenceLine
                  key={e.date} x={e.date} stroke="#7c869a" strokeDasharray="3 3"
                  label={{ value: '', position: 'top', fill: '#7c869a', fontSize: 9 }}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="flex items-center gap-4 mt-3 text-[11px] muted">
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-slate-200" /> actual</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5" style={{ background: '#3b82f6' }} /> model fair</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: 'rgba(59,130,246,0.2)' }} /> 10–90%</span>
        <span className="flex items-center gap-1.5"><span className="w-3 border-t border-dashed border-dim" /> events</span>
      </div>

      <div className="mt-4 pt-3 border-t border-line/50 text-[11px] muted leading-relaxed">
        <span className="text-slate-300 font-semibold">How to read this.</span>{' '}
        The 10–90% range is where, historically, ~80% of weekly closes have landed relative to the model. It is <span className="text-slate-300">not</span> a guarantee — about 1 week in 10 closes below the lower bound and 1 in 10 above. The R² figures below describe how much of TSLA's <span className="text-slate-300">past variance</span> the model explained in-sample (88%) and out-of-sample (79%); they are <span className="text-slate-300">not</span> a probability that next week's price lands in the band. The model sees only QQQ, NVDA, ARKK, DXY, VIX, gasoline, the bond curve, and known catalyst windows — nothing about Musk, FSD, deliveries, or anything else.
      </div>
    </div>
  )
}
