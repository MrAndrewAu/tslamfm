import type { ModelStats } from '../types'

interface Props { stats: ModelStats; nWeeks: number }

function Stat({ label, value, help }: { label: string; value: string; help?: string }) {
  return (
    <div title={help}>
      <div className="text-[11px] muted uppercase tracking-wider">{label}</div>
      <div className="mono text-lg mt-0.5">{value}</div>
    </div>
  )
}

export default function ModelStats({ stats, nWeeks }: Props) {
  return (
    <div className="panel p-6">
      <div className="text-sm font-semibold mb-1">Honesty stats</div>
      <div className="text-xs muted mb-5">
        In-sample tells you how well the model fits known data. Out-of-sample
        (walk-forward, refit weekly from 2025-01-03) tells you how well it would
        have predicted the unknown future. <span className="text-slate-300">These are goodness-of-fit numbers, not probabilities</span> — they describe explained variance, not the chance the price lands in the band.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
        <Stat
          label="In-sample R²"
          value={stats.r2_in.toFixed(3)}
          help="Variance explained on the model fit window. Don't be impressed by this alone."
        />
        <Stat
          label="In-sample MAE"
          value={`${stats.mae_in_pct.toFixed(1)}%`}
          help="Mean absolute residual in price terms."
        />
        <Stat
          label="OOS R²"
          value={stats.oos.r2.toFixed(3)}
          help="Walk-forward out-of-sample R² since 2025-01-03. Negative means worse than predicting the mean (typically because of a level dislocation)."
        />
        <Stat
          label="OOS MAE"
          value={`${stats.oos.mae_pct.toFixed(1)}%`}
          help="Walk-forward out-of-sample mean absolute error."
        />
        <Stat
          label="OOS correlation"
          value={stats.oos.corr.toFixed(3)}
          help="OOS directional fit. High values mean the model gets the *shape* of moves right even if the *level* is off."
        />
        <Stat label="Sample" value={`${nWeeks}w`} help="Weekly observations used to fit the model." />
        <Stat label="OOS sample" value={`${stats.oos.n}w`} help="Weeks evaluated out-of-sample." />
        <Stat
          label="σₜ (latest, EWMA)"
          value={(stats.sigma_t_latest ?? stats.sigma_resid_log).toFixed(3)}
          help="Adaptive forecast-error sigma (EWMA, λ=0.94). Used to scale the predictive range so it tightens in calm regimes and widens after large surprises. Constant full-sample fit sigma is shown only as a reference."
        />
      </div>
    </div>
  )
}
