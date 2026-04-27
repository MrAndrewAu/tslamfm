import { useState } from 'react'
import type { Model } from '../types'

interface Props { model: Model }

function pStars(p: number | null | undefined): string {
  if (p == null) return ''
  if (p < 0.001) return '***'
  if (p < 0.01)  return '**'
  if (p < 0.05)  return '*'
  if (p < 0.1)   return '.'
  return ''
}

export default function Methodology({ model }: Props) {
  const [open, setOpen] = useState(false)
  const bandCoverageBacktest = model.stats.band_coverage_backtest != null
    ? `${(model.stats.band_coverage_backtest * 100).toFixed(1)}%`
    : null
  const bandCoverageOos = model.stats.band_coverage_oos != null
    ? `${(model.stats.band_coverage_oos * 100).toFixed(1)}%`
    : null
  return (
    <div className="panel p-6">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between text-left"
      >
        <div>
          <div className="text-sm font-semibold">Methodology</div>
          <div className="text-xs muted">
            Open the details: full coefficient table, factor definitions, event list, what we rejected and why.
          </div>
        </div>
        <span className="text-dim text-xl">{open ? '−' : '+'}</span>
      </button>

      {open && (
        <div className="mt-6 space-y-6 text-sm">
          <section>
            <h3 className="font-semibold mb-2">The equation</h3>
            <pre className="mono text-xs bg-ink p-4 rounded border border-line overflow-x-auto">
{`log(TSLA) = α
          + β₁·log(QQQ)
          + β₂·log(DXY)
          + β₃·log(VIX)
          + β₄·NVDA_excess
          + β₅·ARKK_excess
          + β₆·RBOB_zscore_52w
          + β₇·curve_IEF_SHY_zscore_52w
          + Σ event_dummies
          + ε`}
            </pre>
            <div className="text-xs muted mt-2 space-y-2">
              <p>
                <code className="mono">NVDA_excess = log(NVDA) − (a + b·log(QQQ))</code> — the residual of NVDA after regressing on QQQ. Same structure for ARKK_excess. This isolates the rotation component from the market beta.
              </p>
              <p>
                <code className="mono">RBOB_zscore_52w = (log(RBOB) − μ₅₂) / σ₅₂</code> — gasoline futures z-scored against the trailing 52 weeks (backward-only, no lookahead). Added in v6.1 as a gas-affordability proxy: high gas hurts EV demand. Cleared a 5-test gauntlet — variant robustness, VIX-complement check, lookahead-clean transform, sub-period split, and a 500-shuffle permutation null (0/500 beat the observed +5.62pp OOS R² lift).
              </p>
              <p>
                <code className="mono">curve_IEF_SHY_zscore_52w = (log(IEF/SHY) − μ₅₂) / σ₅₂</code> — bond-curve shape (7–10y vs 1–3y Treasuries), z-scored against the trailing 52 weeks. Added in v6.2 as a recession/rate-cut pricing proxy: bull-flattening (curve up = long bonds outperform) discounts TSLA. Cleared the same 5-test gauntlet on top of v6.1 with a +4.88pp OOS R² lift, lift positive in all three OOS sub-periods, and 0/500 permutations beat observed. Note: CURVE is a strict complement to RBOB — alone it hurts the model; only additive on top of v6.1.
              </p>
              <p>
                <code className="mono">E_Robotaxi_Austin</code> — 8-week dummy starting 2025-06-22 (Austin commercial robotaxi service launch). Added in v6.3: TSLA traded −7% below model average for all 8 consecutive weeks of the launch window — a classic "sell-the-news" effect. The signal is macro-orthogonal (VIX/RBOB/ARKK were near OOS means throughout). β=−0.119 (p=0.030), walk-forward OOS R² lift +2.30pp, permutation null p=0.020 (permutation 95th percentile was +0.23pp vs observed +0.32pp).
              </p>
              <p>
                Coefficients fit by OLS on weekly Friday closes from {model.window.start} to {model.window.end} ({model.window.n_weeks} weeks).
              </p>
              <p>
                <span className="text-slate-300">Predictive range.</span> The shaded band is calibrated from expanding <span className="text-slate-300">one-step-ahead forecast errors</span>, not full-sample fit residuals. At each row, the raw asymmetry comes from the earlier 10th / 90th percentile forecast errors, while width is scaled by an EWMA(λ=0.94) forecast-error sigma so the range tightens in calm regimes and widens after large misses. This keeps the range lookahead-free and predictive rather than merely descriptive of the fitted history.
                {bandCoverageBacktest && bandCoverageOos && model.stats.band_backtest_start ? ` Realized coverage in backtest was ${bandCoverageBacktest} since ${model.stats.band_backtest_start}, and ${bandCoverageOos} in the official OOS window since ${model.stats.oos.start}.` : ''}
              </p>
            </div>
          </section>

          <section>
            <h3 className="font-semibold mb-2">Coefficients</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs mono">
                <thead className="text-dim border-b border-line">
                  <tr>
                    <th className="text-left py-2 pr-4">Factor</th>
                    <th className="text-right py-2 px-4">β</th>
                    <th className="text-right py-2 px-4">p</th>
                    <th className="text-left py-2 pl-4">sig</th>
                  </tr>
                </thead>
                <tbody>
                  {['Intercept', ...model.factors].map(name => {
                    const b = model.coefficients[name]
                    const p = model.p_values[name]
                    return (
                      <tr key={name} className="border-b border-line/50">
                        <td className="py-1.5 pr-4">{name}</td>
                        <td className="text-right py-1.5 px-4">{b?.toFixed(3) ?? '—'}</td>
                        <td className="text-right py-1.5 px-4">{p != null ? p.toExponential(1) : '—'}</td>
                        <td className="py-1.5 pl-4 text-warn">{pStars(p)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div className="text-[11px] muted mt-2">
              Sig: <span className="text-warn">***</span> p&lt;0.001, <span className="text-warn">**</span> p&lt;0.01, <span className="text-warn">*</span> p&lt;0.05, <span className="text-warn">.</span> p&lt;0.10
            </div>
          </section>

          <section>
            <h3 className="font-semibold mb-2">Events (8-week dummies)</h3>
            <ul className="space-y-1 text-xs">
              {model.events.map(e => (
                <li key={e.name} className="flex items-center justify-between border-b border-line/50 py-1">
                  <span><span className="mono text-dim">{e.start}</span> · {e.label}</span>
                  <span className="mono text-[11px]">
                    {e.in_model
                      ? <>β={e.beta?.toFixed(2)} <span className="text-warn">{pStars(e.p_value)}</span></>
                      : <span className="muted">excluded</span>}
                  </span>
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h3 className="font-semibold mb-2">What we rejected (and why)</h3>
            <ul className="space-y-2 text-xs muted list-disc list-inside">
              <li><span className="text-slate-300">v5 lag-cheat:</span> adding the prior week's TSLA price drove R² to 0.99 — but the model just predicted persistence. A "TSLA next week = TSLA this week" baseline beat it.</li>
              <li><span className="text-slate-300">EPS:</span> R² = 0.009. Trailing earnings have no measurable weekly relationship with TSLA price.</li>
              <li><span className="text-slate-300">Vehicle deliveries:</span> p = 0.68 once macro factors are controlled.</li>
              <li><span className="text-slate-300">Short interest:</span> too smooth a series; no usable signal.</li>
              <li><span className="text-slate-300">VVIX, SKEW, raw volume:</span> tested in v7/v8; none added meaningful out-of-sample lift.</li>
              <li><span className="text-slate-300">12-week realized vol (v9):</span> small in-sample lift, but it's computed from TSLA's own returns — flagged as a soft persistence backdoor.</li>
            </ul>
          </section>

          <section className="text-xs muted">
            <h3 className="font-semibold text-slate-300 mb-2">Honest limits</h3>
            <p>This is a statistical model, not advice. It does not know about: Musk's tweets, robotaxi rollout, Optimus, FSD progress, regulatory action, fraud, brand sentiment, or anything not encoded in QQQ/DXY/VIX/NVDA/ARKK/RBOB/CURVE plus the selected event windows. The current shipped walk-forward stats are OOS R² {model.stats.oos.r2.toFixed(3)} and OOS correlation {model.stats.oos.corr.toFixed(3)}. If actual TSLA sits meaningfully away from model fair value, the market is pricing something this factor set cannot see. Use accordingly.</p>
          </section>
        </div>
      )}
    </div>
  )
}
