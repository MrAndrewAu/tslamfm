import { fmtUSD, fmtPct } from '../lib/model'

interface Props {
  actual: number
  fair: number
  low: number
  high: number
  asOf: string
  isLive: boolean
}

/**
 * "Fair Value Gauge" — the hero component.
 *
 *   $actual                        $fair
 *      ●───────────────────────────○
 *           gap: −X% (cheap/rich)
 *
 *  ±1σ band shaded around fair value.
 */
export default function Gauge({ actual, fair, low, high, asOf, isLive }: Props) {
  const gap = (actual / fair - 1) * 100
  const verdict =
    gap < -15 ? { label: 'CHEAP vs model',     color: 'text-good' } :
    gap >  15 ? { label: 'RICH vs model',      color: 'text-bad'  } :
                { label: 'FAIR vs model',      color: 'text-warn' }

  // Place markers on a 0..100 scale spanning low → high (the band)
  const span = Math.max(high - low, 1e-6)
  const pad  = span * 0.4   // visual padding so markers aren't at edges
  const xMin = low  - pad
  const xMax = high + pad
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - xMin) / (xMax - xMin)) * 100))

  // Fair value renders in a neutral blue (the model's "voice").
  // The valuation signal (under/over) is carried by the gap % below.
  const fairColor = '#3b82f6'

  return (
    <div className="panel p-6 md:p-8">
      <div className="flex items-baseline justify-between mb-4">
        <div className="text-xs muted uppercase tracking-wider">Fair Value vs Market</div>
        <div className="text-[11px] muted flex items-center gap-2">
          <span className={`inline-block w-1.5 h-1.5 rounded-full ${isLive ? 'bg-good animate-pulse' : 'bg-dim'}`} />
          {isLive ? 'live' : 'snapshot'} · {asOf}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        <div>
          <div className="text-xs muted uppercase tracking-wider">Market price</div>
          <div className="text-4xl md:text-5xl mono font-semibold mt-1">{fmtUSD(actual)}</div>
        </div>
        <div className="text-right">
          <div className="text-xs muted uppercase tracking-wider">Model fair value</div>
          <div className="text-4xl md:text-5xl mono font-semibold mt-1" style={{ color: fairColor }}>{fmtUSD(fair)}</div>
          <div className="text-[11px] muted mt-1 mono">
            ±1σ band {fmtUSD(low, 0)} – {fmtUSD(high, 0)}
          </div>
        </div>
      </div>

      {/* Gauge bar */}
      <div className="relative h-20">
        {/* Band */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3 rounded-full bg-accent/15 border border-accent/30"
          style={{ left: `${pct(low)}%`, right: `${100 - pct(high)}%` }}
        />
        {/* Centerline */}
        <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-px bg-line" />
        {/* Lower bound tick + label */}
        <div className="absolute top-1/2 -translate-y-1/2" style={{ left: `${pct(low)}%` }}>
          <div className="-translate-x-1/2 w-px h-3 bg-accent/60" />
          <div className="absolute -top-5 -translate-x-1/2 text-[10px] muted whitespace-nowrap mono">{fmtUSD(low, 0)}</div>
          <div className="absolute top-4 -translate-x-1/2 text-[10px] muted whitespace-nowrap">−1σ</div>
        </div>
        {/* Upper bound tick + label */}
        <div className="absolute top-1/2 -translate-y-1/2" style={{ left: `${pct(high)}%` }}>
          <div className="-translate-x-1/2 w-px h-3 bg-accent/60" />
          <div className="absolute -top-5 -translate-x-1/2 text-[10px] muted whitespace-nowrap mono">{fmtUSD(high, 0)}</div>
          <div className="absolute top-4 -translate-x-1/2 text-[10px] muted whitespace-nowrap">+1σ</div>
        </div>
        {/* Fair marker */}
        <div className="absolute top-1/2 -translate-y-1/2" style={{ left: `${pct(fair)}%` }}>
          <div className="-translate-x-1/2 w-4 h-4 rounded-full border-2 border-ink" style={{ background: fairColor }} />
          <div className="absolute top-6 -translate-x-1/2 text-[11px] muted whitespace-nowrap">fair</div>
        </div>
        {/* Actual marker */}
        <div className="absolute top-1/2 -translate-y-1/2" style={{ left: `${pct(actual)}%` }}>
          <div className="-translate-x-1/2 w-5 h-5 rounded-full bg-slate-100 border-2 border-ink" />
          <div className="absolute top-6 -translate-x-1/2 text-[11px] mono whitespace-nowrap">actual</div>
        </div>
      </div>

      <div className="flex items-baseline justify-between mt-8 pt-4 border-t border-line">
        <div className="text-xs muted uppercase tracking-wider">Gap</div>
        <div className="flex items-baseline gap-3">
          <span className={`text-2xl mono font-semibold ${gap < 0 ? 'text-good' : gap > 0 ? 'text-bad' : 'text-warn'}`}>
            {fmtPct(gap)}
          </span>
          <span className={`text-xs font-semibold ${verdict.color}`}>{verdict.label}</span>
        </div>
      </div>
    </div>
  )
}
