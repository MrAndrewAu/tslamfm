// Shared TypeScript types matching scripts/build_model_data.py output

export interface Residualization {
  intercept: number
  beta_log_QQQ: number
  source: string
}

export interface ModelEvent {
  name: string
  start: string
  label: string
  weeks: number
  in_model: boolean
  beta: number | null
  p_value: number | null
}

export interface ModelStats {
  r2_in: number
  mae_in_pct: number
  sigma_resid_log: number
  sigma_t_method?: string
  sigma_t_latest?: number
  q10_log?: number
  q90_log?: number
  sigma_IS_log?: number
  oos: {
    r2: number
    mae_pct: number
    corr: number
    start: string
    n: number
  }
}

export interface CurrentSnapshot {
  date: string
  tsla_actual: number
  tsla_fair: number
  gap_pct: number
  sigma_low: number
  sigma_high: number
  sigma_t?: number
  q_low?: number
  q_high?: number
  factors_now: Record<string, number>
  contribution_dollars: {
    QQQ: number
    DXY: number
    VIX: number
    NVDA_rotation: number
    ARKK_rotation: number
    gas_affordability: number
    recession_pricing: number
    events: number
    baseline: number
  }
  active_events: { name: string; start: string; label: string }[]
  underlyings: {
    TSLA: number; QQQ: number; DXY: number; VIX: number; NVDA: number; ARKK: number
    RBOB: number; IEF: number; SHY: number
  }
}

export interface Model {
  version: string
  generated_at: string
  window: { start: string; end: string; n_weeks: number }
  factors: string[]
  coefficients: Record<string, number>
  p_values: Record<string, number>
  residualizations: Record<string, Residualization>
  rolling_stats?: Record<string, {
    window: number
    min_periods: number
    log_mean_latest: number
    log_std_latest: number
    source: string
  }>
  events: ModelEvent[]
  stats: ModelStats
  current: CurrentSnapshot
}

export interface HistoryRow {
  date: string
  actual: number
  fitted: number
  low: number
  high: number
  sigma_t?: number
  low_q?: number
  high_q?: number
}

export interface LiveQuotes {
  TSLA: number
  QQQ: number
  DXY: number
  VIX: number
  NVDA: number
  ARKK: number
  fetched_at: string
}
