import { fmtUSD } from '../lib/model'

interface Props {
  actual: number
  fair: number
}

interface TierDef {
  loExcl: number
  hiIncl: number
  label: string
  dir: 'top' | 'neutral' | 'bottom'
  colorClass: string
  hitRate: { w4: number; w8: number; w13: number; n: number } | null
  tagline: string
}

// ── Tier definitions ──────────────────────────────────────────────────────────
// Hit-rate figures from analyze_gap_signal.py  (192 weekly obs, Sep 2022–May 2026).
// OOS walk-forward r(gap, 13w fwd return) = −0.47 (p < 0.001).
// Signal is reliable at |gap| ≥ 15 %; the ±5–15 % zone is statistically weak.

const TIERS: TierDef[] = [
  {
    loExcl: 0.25, hiIncl: 99,
    label: 'Very Stretched Above Fair Value',
    dir: 'top', colorClass: 'text-bad',
    hitRate: { w4: 57, w8: 100, w13: 71, n: 7 },
    tagline: 'Price far above the fundamental anchor — all 7 historical cases declined within 8 weeks.',
  },
  {
    loExcl: 0.15, hiIncl: 0.25,
    label: 'Stretched Above Fair Value',
    dir: 'top', colorClass: 'text-bad',
    hitRate: { w4: 80, w8: 80, w13: 80, n: 10 },
    tagline: 'Above the fundamental anchor — price declined 8 of 10 times over the following 8–13 weeks.',
  },
  {
    loExcl: 0.05, hiIncl: 0.15,
    label: 'Elevated',
    dir: 'top', colorClass: 'text-warn',
    hitRate: { w4: 59, w8: 53, w13: 39, n: 39 },
    tagline: 'Slightly above fair value. Signal is weak at this level — no reliable mean reversion until gap widens.',
  },
  {
    loExcl: -0.05, hiIncl: 0.05,
    label: 'Near Fair Value',
    dir: 'neutral', colorClass: 'text-[#3b82f6]',
    hitRate: null,
    tagline: 'Price is consistent with the current macro backdrop. No directional signal.',
  },
  {
    loExcl: -0.15, hiIncl: -0.05,
    label: 'Depressed',
    dir: 'bottom', colorClass: 'text-good',
    hitRate: { w4: 66, w8: 74, w13: 80, n: 35 },
    tagline: 'Below the fundamental anchor — price recovered 74–80 % of the time over 8–13 weeks.',
  },
  {
    loExcl: -0.25, hiIncl: -0.15,
    label: 'Stretched Below Fair Value',
    dir: 'bottom', colorClass: 'text-good',
    hitRate: { w4: 60, w8: 73, w13: 73, n: 15 },
    tagline: 'Well below fair value — 73 % rally rate observed at both 8 and 13 weeks.',
  },
  {
    loExcl: -99, hiIncl: -0.25,
    label: 'Very Stretched Below Fair Value',
    dir: 'bottom', colorClass: 'text-good',
    hitRate: { w4: 100, w8: 100, w13: 100, n: 2 },
    tagline: 'Rare extreme — both historical cases were followed by a strong rally.',
  },
]

// ── Gauge bar ─────────────────────────────────────────────────────────────────
// Range −30 % → +30 %, seven color bands.

const GAUGE_MIN = -0.30
const GAUGE_MAX =  0.30

const ZONES = [
  { lo: -0.30, hi: -0.25, color: '#15803d' },  // deep green
  { lo: -0.25, hi: -0.15, color: '#22c55e' },  // green
  { lo: -0.15, hi: -0.05, color: '#86efac' },  // light green
  { lo: -0.05, hi:  0.05, color: '#3b82f6' },  // blue  (neutral)
  { lo:  0.05, hi:  0.15, color: '#fbbf24' },  // amber
  { lo:  0.15, hi:  0.25, color: '#f87171' },  // light red
  { lo:  0.25, hi:  0.30, color: '#ef4444' },  // red
]

function getTier(gap: number): TierDef {
  return TIERS.find(t => gap > t.loExcl && gap <= t.hiIncl) ?? TIERS[3]
}

function markerPct(gap: number): number {
  const clamped = Math.max(GAUGE_MIN, Math.min(GAUGE_MAX, gap))
  return ((clamped - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN)) * 100
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function SignalGauge({ actual, fair }: Props) {
  const gap = (actual - fair) / fair
  const tier = getTier(gap)
  const pct  = markerPct(gap)
  const gapPct = gap * 100

  return (
    <div className="panel p-6">
      {/* Header row */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <div className="text-sm font-semibold">Valuation Signal</div>
          <div className="text-xs muted mt-1">
            Gap between current price and fundamental fair value.
            At ±15 %+, mean reversion has occurred 73–100 % of the time over 8–13 weeks.
          </div>
        </div>
        <div className="text-right shrink-0 ml-6">
          <div className={`text-3xl font-bold mono ${tier.colorClass}`}>
            {gapPct >= 0 ? '+' : ''}{gapPct.toFixed(1)}%
          </div>
          <div className="text-[11px] muted mt-1">actual vs fair value</div>
        </div>
      </div>

      {/* Gauge bar + marker */}
      <div className="mb-1">
        <div className="relative">
          {/* Color bands */}
          <div className="flex h-5 rounded-lg overflow-hidden">
            {ZONES.map((z, i) => (
              <div
                key={i}
                style={{
                  width: `${((z.hi - z.lo) / (GAUGE_MAX - GAUGE_MIN)) * 100}%`,
                  backgroundColor: z.color,
                  opacity: 0.75,
                }}
              />
            ))}
          </div>
          {/* White needle */}
          <div
            className="absolute inset-y-0 w-0.5 bg-white"
            style={{ left: `${pct}%`, transform: 'translateX(-50%)' }}
          />
          {/* Downward triangle below the bar */}
          <div
            className="absolute"
            style={{ left: `${pct}%`, top: '100%', transform: 'translateX(-50%)' }}
          >
            <div
              className="w-0 h-0"
              style={{
                borderLeft: '5px solid transparent',
                borderRight: '5px solid transparent',
                borderTop: '6px solid white',
                marginTop: '2px',
              }}
            />
          </div>
        </div>

        {/* Axis labels */}
        <div className="flex justify-between mt-4 text-[11px] muted select-none">
          <span>← Undervalued (−30 %)</span>
          <span>Fair value (0 %)</span>
          <span>Overvalued (+30 %) →</span>
        </div>
      </div>

      {/* Tier + tagline */}
      <div className="mt-5 pt-4 border-t border-line">
        <div className={`text-sm font-semibold ${tier.colorClass}`}>{tier.label}</div>
        <div className="text-xs muted mt-1">{tier.tagline}</div>
      </div>

      {/* Hit-rate cards */}
      {tier.hitRate !== null && (
        <div className="mt-4">
          <div className="text-[11px] muted mb-2 uppercase tracking-wider">
            Historical accuracy at this tier — {tier.hitRate.n} obs (Sep 2022–May 2026)
          </div>
          <div className="grid grid-cols-3 gap-3">
            {(
              [
                { label: '4-week',  val: tier.hitRate.w4  },
                { label: '8-week',  val: tier.hitRate.w8  },
                { label: '13-week', val: tier.hitRate.w13 },
              ] as const
            ).map(({ label, val }) => (
              <div key={label} className="bg-ink rounded-lg p-3 text-center">
                <div className={`text-xl font-bold mono ${tier.colorClass}`}>{val}%</div>
                <div className="text-[11px] muted mt-0.5">{label}</div>
              </div>
            ))}
          </div>
          <div className="text-[11px] muted mt-2">
            {tier.dir === 'top'
              ? 'Hit = price was lower N weeks later. The 8–13 week horizon is historically most reliable.'
              : 'Hit = price was higher N weeks later. The 8–13 week horizon is historically most reliable.'}
          </div>
        </div>
      )}

      {/* Neutral state */}
      {tier.hitRate === null && (
        <div className="mt-3 text-xs muted">
          No signal at the current gap. The signal activates when the gap widens to ±10–15 %.
          At ±15–25 %, the historical mean-reversion rate is 73–100 % over 8–13 weeks.
        </div>
      )}

      {/* Price pair */}
      <div className="mt-4 pt-4 border-t border-line flex justify-between text-sm">
        <div>
          <span className="muted">Actual </span>
          <span className="mono">{fmtUSD(actual, 0)}</span>
        </div>
        <div>
          <span className="muted">Fair value </span>
          <span className="mono" style={{ color: '#3b82f6' }}>{fmtUSD(fair, 0)}</span>
        </div>
      </div>
    </div>
  )
}
