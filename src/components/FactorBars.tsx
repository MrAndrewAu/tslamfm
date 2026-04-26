import { fmtUSD } from '../lib/model'
import type { Model } from '../types'

interface Props {
  contributions: Model['current']['contribution_dollars']
  fair: number
  underlyings: Model['current']['underlyings']
}

const ROWS: { key: keyof Props['contributions']; label: string; help: string }[] = [
  { key: 'baseline',      label: 'Baseline (intercept)', help: 'Model constant — what TSLA would be if every factor were zero.' },
  { key: 'QQQ',           label: 'QQQ level',           help: 'Tech-market beta. Positive = tech market lifts TSLA.' },
  { key: 'DXY',           label: 'Dollar (DXY)',        help: 'Dollar strength. Positive coefficient is unusual; reflects regime.' },
  { key: 'VIX',           label: 'VIX',                 help: 'Volatility. TSLA carries a vol premium.' },
  { key: 'NVDA_rotation', label: 'NVDA rotation',       help: 'NVDA performance net of QQQ. Negative β = TSLA suffers when NVDA outperforms.' },
  { key: 'ARKK_rotation', label: 'ARKK rotation',       help: 'ARKK performance net of QQQ. Speculative-growth rotation.' },
  { key: 'events',        label: 'Active events',       help: 'Sum of 8-week event dummies currently active.' },
]

export default function FactorBars({ contributions, fair, underlyings }: Props) {
  const maxAbs = Math.max(...ROWS.map(r => Math.abs(contributions[r.key])))

  return (
    <div className="panel p-6">
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-sm font-semibold">Why this fair value?</div>
        <div className="text-xs muted">contribution to {fmtUSD(fair, 0)}</div>
      </div>
      <div className="text-xs muted mb-5">
        Each factor's marginal contribution to the model's current price prediction.
      </div>

      <div className="space-y-2">
        {ROWS.map(r => {
          const v = contributions[r.key]
          const pct = maxAbs ? (Math.abs(v) / maxAbs) * 100 : 0
          const positive = v >= 0
          return (
            <div key={r.key} className="grid grid-cols-12 gap-3 items-center text-sm">
              <div className="col-span-4 truncate" title={r.help}>{r.label}</div>
              <div className="col-span-6 h-5 relative bg-line/40 rounded">
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-line" />
                <div
                  className={`absolute top-0 bottom-0 ${positive ? 'left-1/2 bg-good/60' : 'right-1/2 bg-bad/60'}`}
                  style={{ width: `${pct / 2}%` }}
                />
              </div>
              <div className={`col-span-2 text-right mono text-[13px] ${positive ? 'text-good' : 'text-bad'}`}>
                {positive ? '+' : ''}{fmtUSD(v, 0)}
              </div>
            </div>
          )
        })}
        <div className="grid grid-cols-12 gap-3 items-center text-sm pt-3 border-t border-line">
          <div className="col-span-4 font-semibold">Model total</div>
          <div className="col-span-6" />
          <div className="col-span-2 text-right mono font-semibold text-accent">{fmtUSD(fair, 0)}</div>
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-line grid grid-cols-3 md:grid-cols-6 gap-3 text-[11px]">
        {(['TSLA', 'QQQ', 'DXY', 'VIX', 'NVDA', 'ARKK'] as const).map(k => (
          <div key={k}>
            <div className="muted uppercase tracking-wider">{k}</div>
            <div className="mono mt-0.5">{underlyings[k].toFixed(2)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
