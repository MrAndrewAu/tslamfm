import { useMemo, useState } from 'react'
import {
  Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, ComposedChart, LineChart,
} from 'recharts'
import type { HistoryRow, Model } from '../types'
import { fmtUSD } from '../lib/model'

interface Props {
  history: HistoryRow[]
  model: Model
}

const RANGES = [
  { key: '1Y', months: 12 },
  { key: '3Y', months: 36 },
  { key: 'ALL', months: 9999 },
] as const

export default function HistoryChart({ history, model }: Props) {
  const [range, setRange] = useState<'1Y' | '3Y' | 'ALL'>('ALL')
  const [showResid, setShowResid] = useState(false)

  const data = useMemo(() => {
    if (range === 'ALL') return history
    const cutoff = new Date()
    cutoff.setMonth(cutoff.getMonth() - RANGES.find(r => r.key === range)!.months)
    const iso = cutoff.toISOString().slice(0, 10)
    return history.filter(r => r.date >= iso)
  }, [history, range])

  const residData = useMemo(
    () => data.map(r => ({ date: r.date, resid: ((r.actual / r.fitted - 1) * 100) })),
    [data]
  )

  const eventLines = model.events
    .filter(e => e.in_model && e.start >= (data[0]?.date ?? ''))
    .map(e => ({ date: e.start, label: e.label }))

  return (
    <div className="panel p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-sm font-semibold">Historical fit</div>
          <div className="text-xs muted">
            Weekly: actual price vs model fair value, ±1σ band
          </div>
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
                className={`text-[11px] px-2.5 py-1 rounded transition ${
                  range === r.key ? 'bg-accent/20 text-accent' : 'text-dim hover:text-slate-300'
                }`}
              >
                {r.key}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ width: '100%', height: 380 }}>
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
              <Line type="monotone" dataKey="resid" stroke="#e63946" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="bandFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"  stopColor="#e63946" stopOpacity={0.18} />
                  <stop offset="100%" stopColor="#e63946" stopOpacity={0.04} />
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
              <Area type="monotone" dataKey="high" stroke="none" fill="url(#bandFill)" />
              <Area type="monotone" dataKey="low"  stroke="none" fill="#0b0e14" />
              <Line type="monotone" dataKey="fitted" stroke="#e63946" strokeWidth={2} dot={false} name="fair value" />
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
        <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-accent" /> model fair</span>
        <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-accent/20" /> ±1σ</span>
        <span className="flex items-center gap-1.5"><span className="w-3 border-t border-dashed border-dim" /> events</span>
      </div>
    </div>
  )
}
