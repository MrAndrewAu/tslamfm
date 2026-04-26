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
  factors_now: Record<string, number>
  contribution_dollars: {
    QQQ: number
    DXY: number
    VIX: number
    NVDA_rotation: number
    ARKK_rotation: number
    events: number
    baseline: number
  }
  active_events: { name: string; start: string; label: string }[]
  underlyings: {
    TSLA: number; QQQ: number; DXY: number; VIX: number; NVDA: number; ARKK: number
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
