import type { Model, LiveQuotes } from '../types'

/**
 * Compute current fair value from live quotes using the shipped v6.4 coefficients.
 * Uses the residualization equations to compute NVDA_excess, ARKK_excess on the fly.
 * RBOB_zscore_52w, curve_IEF_SHY_zscore_52w, and active event dummies are taken
 * from the static model snapshot (slow-moving regime/calendar factors -- not
 * driven by intraday quotes).
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

  // Carry over RBOB_zscore_52w, curve_IEF_SHY_zscore_52w + event dummies from
  // the static snapshot. These are slow-moving regime / calendar factors;
  // updating from intraday quotes would be misleading.
  const snapFactors = model.current.factors_now
  for (const k of ['RBOB_zscore_52w', 'curve_IEF_SHY_zscore_52w']) {
    if (k in snapFactors) fac[k] = snapFactors[k]
  }
  for (const k of Object.keys(snapFactors)) {
    if (k.startsWith('E_')) fac[k] = snapFactors[k]
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
    QQQ:               cd('log_QQQ'),
    DXY:               cd('log_DXY'),
    VIX:               cd('log_VIX'),
    NVDA_rotation:     cd('NVDA_excess'),
    ARKK_rotation:     cd('ARKK_excess'),
    gas_affordability: cd('RBOB_zscore_52w'),
    recession_pricing: cd('curve_IEF_SHY_zscore_52w'),
    events:            partial(eventLog),
    baseline: 0, // filled below
  }
  contributions.baseline = fair - (
    contributions.QQQ + contributions.DXY + contributions.VIX +
    contributions.NVDA_rotation + contributions.ARKK_rotation +
    contributions.gas_affordability + contributions.recession_pricing +
    contributions.events
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
