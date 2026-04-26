import type { Model, LiveQuotes } from '../types'

/**
 * Compute current fair value from live quotes using v6-canonical coefficients.
 * Uses the residualization equations to compute NVDA_excess, ARKK_excess on the fly.
 * Active events are taken from the static model snapshot (events evolve slowly,
 * not via live quotes).
 */
export function computeFairValue(model: Model, q: LiveQuotes): {
  fair: number
  log_fair: number
  factors: Record<string, number>
  contribution_dollars: Model['current']['contribution_dollars']
} {
  const log_QQQ = Math.log(q.QQQ)
  const log_DXY = Math.log(q.DXY)
  const log_VIX = Math.log(q.VIX)

  // Residualize NVDA, ARKK against QQQ using stored coefficients
  const r_nvda = model.residualizations.NVDA_excess
  const r_arkk = model.residualizations.ARKK_excess
  const NVDA_excess = Math.log(q.NVDA) - (r_nvda.intercept + r_nvda.beta_log_QQQ * log_QQQ)
  const ARKK_excess = Math.log(q.ARKK) - (r_arkk.intercept + r_arkk.beta_log_QQQ * log_QQQ)

  const fac: Record<string, number> = {
    log_QQQ, log_DXY, log_VIX, NVDA_excess, ARKK_excess,
  }

  // Carry over event dummies from the static current snapshot — they're date-driven.
  for (const k of Object.keys(model.current.factors_now)) {
    if (k.startsWith('E_')) fac[k] = model.current.factors_now[k]
  }

  const c = model.coefficients
  let log_fair = c.Intercept
  for (const f of model.factors) log_fair += (c[f] ?? 0) * (fac[f] ?? 0)
  const fair = Math.exp(log_fair)

  // Contribution decomposition (in $ terms relative to "factor=0")
  const partial = (component_log: number) =>
    Math.exp(log_fair) - Math.exp(log_fair - component_log)

  const cd = (k: string) => partial((c[k] ?? 0) * (fac[k] ?? 0))
  const eventLog = model.factors
    .filter(f => f.startsWith('E_'))
    .reduce((s, f) => s + (c[f] ?? 0) * (fac[f] ?? 0), 0)

  const contributions = {
    QQQ:           cd('log_QQQ'),
    DXY:           cd('log_DXY'),
    VIX:           cd('log_VIX'),
    NVDA_rotation: cd('NVDA_excess'),
    ARKK_rotation: cd('ARKK_excess'),
    events:        partial(eventLog),
    baseline: 0, // filled below
  }
  contributions.baseline = fair - (
    contributions.QQQ + contributions.DXY + contributions.VIX +
    contributions.NVDA_rotation + contributions.ARKK_rotation + contributions.events
  )

  return { fair, log_fair, factors: fac, contribution_dollars: contributions }
}

export function fmtUSD(n: number, decimals = 2): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD',
    minimumFractionDigits: decimals, maximumFractionDigits: decimals,
  }).format(n)
}

export function fmtPct(n: number, decimals = 1): string {
  const sign = n > 0 ? '+' : ''
  return `${sign}${n.toFixed(decimals)}%`
}
