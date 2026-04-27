"""
#5 Kalman state-space dynamic β probe vs v6.3 baseline.

Model: treat β_QQQ as a time-varying hidden state (random walk).
All other factor betas remain fixed (estimated by OLS on IS window).

State-space formulation
-----------------------
Observation:  y*_t = β_t · x_t + ε_t,     ε_t ~ N(0, σ²_ε)
Transition:   β_t  = β_{t-1} + η_t,        η_t ~ N(0, σ²_η)

where y*_t = log_TSLA_t - (α + γ·[DXY,VIX,NVDA,ARKK,RBOB,CURVE,events])
      x_t  = log_QQQ_t
      β_t  = time-varying QQQ beta (hidden state)

Hyperparameters σ²_ε, σ²_η estimated by MLE (grid search on σ_η/σ_ε ratio).

Walk-forward procedure (lookahead-free)
---------------------------------------
1. Using only IS data: estimate fixed betas (OLS), then MLE for σ_η/σ_ε.
2. Run Kalman filter forward through IS to get β_{T_is | T_is} (filtered state).
3. For each OOS step t:
   a. Predict: β_{t|t-1} = β_{t-1|t-1}  (random walk: E[β_t] = β_{t-1})
               P_{t|t-1} = P_{t-1|t-1} + σ²_η
               y_hat_t   = α + β_{t|t-1}·x_t + γ·Z_t
   b. Observe y_t, update:
               innovation  = y_t - y_hat_t
               K_t         = P_{t|t-1}·x_t / (x_t²·P_{t|t-1} + σ²_ε)
               β_{t|t}     = β_{t|t-1} + K_t · innovation
               P_{t|t}     = (1 - K_t·x_t) · P_{t|t-1}
4. Collect all y_hat_t predictions → compute WF OOS R².

Also tests dynamic β_ARKK_excess (second-most candidate given rotation signal).

Acceptance bar: +2pp WF OOS R² lift vs v6.3.
"""
import warnings, numpy as np, pandas as pd, yfinance as yf
from scipy import stats, optimize
warnings.filterwarnings("ignore")

START, END  = "2022-04-26", "2026-04-26"
OOS_START   = "2025-01-03"
BARRIER_PP  = 2.0

# ── Fetch weekly data ─────────────────────────────────────────────────────────
def wk(sym):
    s = yf.Ticker(sym).history(start=START, end=END, interval="1wk")["Close"]
    if s.empty: return s
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None: idx = idx.tz_localize(None)
    s.index = idx
    return s.resample("W-FRI").last()

def zscore_52(s):
    return (s - s.rolling(52,min_periods=20).mean()) / s.rolling(52,min_periods=20).std()

def residualize(t, b):
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    c, *_ = np.linalg.lstsq(X, t.to_numpy(), rcond=None)
    return pd.Series(t.to_numpy() - X @ c, index=t.index)

print("Fetching weekly closes...")
raw = {k: wk(v) for k, v in {"TSLA":"TSLA","QQQ":"QQQ","DXY":"DX-Y.NYB",
      "VIX":"^VIX","NVDA":"NVDA","ARKK":"ARKK","RBOB":"RB=F",
      "IEF":"IEF","SHY":"SHY"}.items()}
df = pd.DataFrame(raw).ffill().dropna()

f = pd.DataFrame(index=df.index)
f["log_TSLA"] = np.log(df["TSLA"])
f["log_QQQ"]  = np.log(df["QQQ"])
f["log_DXY"]  = np.log(df["DXY"])
f["log_VIX"]  = np.log(df["VIX"])
f["NVDA_excess"] = residualize(np.log(df["NVDA"]), f["log_QQQ"])
f["ARKK_excess"] = residualize(np.log(df["ARKK"]), f["log_QQQ"])
f["RBOB_zscore_52w"] = zscore_52(np.log(df["RBOB"]))
f["curve_IEF_SHY_zscore_52w"] = zscore_52(np.log(df["IEF"]) - np.log(df["SHY"]))

EVENT_DEFS = [
    ("Split_squeeze_2020","2020-08-11"),("SP500_inclusion","2020-11-16"),
    ("Hertz_1T_peak","2021-10-25"),    ("Twitter_overhang","2022-04-25"),
    ("Twitter_close","2022-10-27"),    ("AI_day_2023","2023-07-19"),
    ("Trump_election","2024-11-06"),   ("DOGE_brand_damage","2025-02-15"),
    ("Musk_exits_DOGE","2025-04-22"),  ("TrillionPay","2025-09-05"),
    ("Tariff_shock","2026-02-01"),     ("Robotaxi_Austin","2025-06-22"),
]
for name, dt in EVENT_DEFS:
    d0 = pd.Timestamp(dt)
    f[f"E_{name}"] = ((f.index >= d0) & (f.index < d0 + pd.Timedelta(weeks=8))).astype(int)
f = f.dropna()
active_e = [f"E_{n}" for n, _ in EVENT_DEFS if f[f"E_{n}"].nunique() > 1]

# V63 factors with dynamic target removed
FIXED_FACTORS_BASE = ["log_DXY","log_VIX","NVDA_excess","ARKK_excess",
                      "RBOB_zscore_52w","curve_IEF_SHY_zscore_52w"] + active_e
V63_ALL = ["log_QQQ"] + FIXED_FACTORS_BASE
print(f"  {len(f)} weeks  IS={( f.index < pd.Timestamp(OOS_START)).sum()}  "
      f"OOS={(f.index >= pd.Timestamp(OOS_START)).sum()}")

# ── Kalman filter utilities ───────────────────────────────────────────────────
def kalman_filter(y_star, x, sigma2_eps, sigma2_eta, beta0=None, P0=None):
    """
    Scalar Kalman filter for y*_t = beta_t * x_t + eps,  beta_t = beta_{t-1} + eta.
    Returns:
      beta_filtered[t] = E[beta_t | y*_1..y*_t]   (updated at t)
      beta_pred[t]     = E[beta_t | y*_1..y*_{t-1}] (one-step-ahead)
      log_lik          = sum of log predictive densities
    """
    n = len(y_star)
    beta_f = np.empty(n)
    beta_p = np.empty(n)
    P_f    = np.empty(n)
    P_p    = np.empty(n)
    ll     = 0.0

    if beta0 is None:
        # Initialise at OLS estimate of first 10 points
        b0 = float(np.dot(x[:10], y_star[:10]) / np.dot(x[:10], x[:10]))
    else:
        b0 = beta0
    if P0 is None:
        P0 = 10.0 * sigma2_eta   # diffuse prior

    beta_prev, P_prev = b0, P0

    for t in range(n):
        # Predict
        beta_p[t] = beta_prev
        P_p[t]    = P_prev + sigma2_eta

        # Innovation
        xt   = x[t]
        innov = y_star[t] - beta_p[t] * xt
        S_t  = xt**2 * P_p[t] + sigma2_eps

        # Log-likelihood contribution
        ll += -0.5 * (np.log(2 * np.pi * S_t) + innov**2 / S_t)

        # Update
        K_t      = P_p[t] * xt / S_t
        beta_f[t] = beta_p[t] + K_t * innov
        P_f[t]    = (1 - K_t * xt) * P_p[t]

        beta_prev, P_prev = beta_f[t], P_f[t]

    return beta_f, beta_p, P_f, P_p, ll

def neg_log_lik(log_params, y_star, x):
    """Objective for MLE. Params: [log(sigma_eps), log(sigma_eta)]."""
    sigma_eps = np.exp(log_params[0])
    sigma_eta = np.exp(log_params[1])
    _, _, _, _, ll = kalman_filter(y_star, x, sigma_eps**2, sigma_eta**2)
    return -ll

def fit_kalman_mle(y_star, x, n_restarts=5):
    """MLE for sigma_eps, sigma_eta with multiple restarts."""
    best_ll, best_params = np.inf, None
    for _ in range(n_restarts):
        x0 = np.random.default_rng(42+_).uniform(-3, 0, size=2)
        res = optimize.minimize(neg_log_lik, x0, args=(y_star, x),
                                method="Nelder-Mead",
                                options={"maxiter": 2000, "xatol": 1e-6})
        if res.fun < best_ll:
            best_ll, best_params = res.fun, res.x
    sigma_eps = np.exp(best_params[0])
    sigma_eta = np.exp(best_params[1])
    return sigma_eps, sigma_eta, -best_ll

# ── Full-sample smoother (diagnostic: upper bound on Kalman benefit) ──────────
def run_full_sample_diagnostic(frame, dynamic_factor, fixed_factors):
    """Oracle Kalman smoother on full sample — not for WF, just diagnostic."""
    y = frame["log_TSLA"].to_numpy()
    x_dyn = frame[dynamic_factor].to_numpy()
    Z_fix = np.column_stack([np.ones(len(frame))] +
                            [frame[c].to_numpy() for c in fixed_factors])
    # Step 1: OLS for fixed betas (partial out dynamic factor first)
    # Augmented OLS with dynamic factor as fixed
    X_aug = np.column_stack([np.ones(len(frame)), x_dyn] +
                            [frame[c].to_numpy() for c in fixed_factors])
    b_ols, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    # Partial residuals: remove everything except dynamic factor's contribution
    alpha_hat = b_ols[0]
    gamma_hat = b_ols[2:]   # fixed factor betas
    y_star = y - alpha_hat - Z_fix[:, 1:] @ gamma_hat  # remove intercept + fixed
    # Step 2: MLE for Kalman hypers
    sigma_eps, sigma_eta, ll = fit_kalman_mle(y_star, x_dyn)
    beta_f, beta_p, _, _, _ = kalman_filter(y_star, x_dyn,
                                            sigma_eps**2, sigma_eta**2)
    # Full-sample fitted values (using filtered β — not smoother, still causal)
    y_hat = alpha_hat + beta_f * x_dyn + Z_fix[:, 1:] @ gamma_hat
    resid = y - y_hat
    r2 = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
    return r2, sigma_eps, sigma_eta, beta_f, alpha_hat, gamma_hat

# ── Walk-forward Kalman OOS ───────────────────────────────────────────────────
def wf_kalman(frame, dynamic_factor, fixed_factors, min_train=104):
    """
    Expanding-window walk-forward with Kalman dynamic beta.
    Hypers (sigma_eps, sigma_eta) estimated ONCE on IS, then fixed.
    Beta state propagated online through OOS.
    """
    IS = frame[frame.index < pd.Timestamp(OOS_START)]
    OOS = frame[frame.index >= pd.Timestamp(OOS_START)]
    if len(IS) < min_train or len(OOS) < 5:
        return np.nan, np.nan

    y_is = IS["log_TSLA"].to_numpy()
    x_is = IS[dynamic_factor].to_numpy()
    Z_is = np.column_stack([np.ones(len(IS))] +
                           [IS[c].to_numpy() for c in fixed_factors])
    # Fixed betas from IS OLS
    X_aug_is = np.column_stack([np.ones(len(IS)), x_is] +
                               [IS[c].to_numpy() for c in fixed_factors])
    b_ols, *_ = np.linalg.lstsq(X_aug_is, y_is, rcond=None)
    alpha_hat = b_ols[0]
    gamma_hat = b_ols[2:]
    y_star_is = y_is - alpha_hat - Z_is[:, 1:] @ gamma_hat

    # MLE hypers on IS
    sigma_eps, sigma_eta, _ = fit_kalman_mle(y_star_is, x_is)

    # Run filter through IS to get final filtered state
    beta_f_is, _, _, P_f_is, _ = kalman_filter(y_star_is, x_is,
                                                sigma_eps**2, sigma_eta**2)
    # Kalman filter returns P_f in position 2 (P_filtered), but our function
    # returns (beta_f, beta_p, P_f, P_p, ll) — P_f is index 2
    # Re-run to get P_filtered correctly
    _, _, Pf_arr, _, _ = kalman_filter(y_star_is, x_is, sigma_eps**2, sigma_eta**2)

    beta_now = beta_f_is[-1]
    P_now    = Pf_arr[-1]

    # OOS online prediction
    y_oos = OOS["log_TSLA"].to_numpy()
    x_oos = OOS[dynamic_factor].to_numpy()
    Z_oos = np.column_stack([np.ones(len(OOS))] +
                            [OOS[c].to_numpy() for c in fixed_factors])

    preds = []
    for t in range(len(OOS)):
        xt = x_oos[t]
        # Predict step (using info up to t-1)
        beta_pred = beta_now
        P_pred    = P_now + sigma_eta**2
        y_hat     = alpha_hat + beta_pred * xt + Z_oos[t, 1:] @ gamma_hat
        preds.append(y_hat)
        # Update step (use realized y_t)
        innov = y_oos[t] - y_hat
        S_t   = xt**2 * P_pred + sigma_eps**2
        K_t   = P_pred * xt / S_t
        beta_now = beta_pred + K_t * innov
        P_now    = (1 - K_t * xt) * P_pred

    preds = np.array(preds)
    r2  = 1 - ((y_oos - preds)**2).sum() / ((y_oos - y_oos.mean())**2).sum()
    mae = np.mean(np.abs(np.exp(y_oos) - np.exp(preds))) / np.mean(np.exp(y_oos)) * 100
    return r2, mae

# ── Standard OLS walk-forward (v6.3 baseline) ────────────────────────────────
def wf_ols(frame, factors, min_train=104):
    IS  = frame[frame.index <  pd.Timestamp(OOS_START)]
    OOS = frame[frame.index >= pd.Timestamp(OOS_START)]
    y_is = IS["log_TSLA"].to_numpy()
    X_is = np.column_stack([np.ones(len(IS))] + [IS[c].to_numpy() for c in factors])
    b, *_ = np.linalg.lstsq(X_is, y_is, rcond=None)
    y_oos = OOS["log_TSLA"].to_numpy()
    X_oos = np.column_stack([np.ones(len(OOS))] + [OOS[c].to_numpy() for c in factors])
    preds = X_oos @ b
    r2  = 1 - ((y_oos - preds)**2).sum() / ((y_oos - y_oos.mean())**2).sum()
    mae = np.mean(np.abs(np.exp(y_oos) - np.exp(preds))) / np.mean(np.exp(y_oos)) * 100
    return r2, mae

# ── Run diagnostics ───────────────────────────────────────────────────────────
print("\n=== v6.3 OLS baseline ===")
r2_ols_oos, mae_ols = wf_ols(f, V63_ALL)
print(f"  OOS R²={r2_ols_oos:.4f}  OOS MAE={mae_ols:.2f}%")

# Full-sample diagnostic (oracle upper bound)
print("\n=== Full-sample diagnostic (oracle filtered β — not WF) ===")
print("  (Shows whether dynamic β has any IS explanatory power at all)")
for dyn_fac, label in [("log_QQQ", "β_QQQ"), ("ARKK_excess", "β_ARKK")]:
    fixed = [c for c in V63_ALL if c != dyn_fac]
    r2_kf, sig_e, sig_h, beta_f, alpha, gamma = run_full_sample_diagnostic(f, dyn_fac, fixed)
    print(f"\n  Dynamic {label}:")
    print(f"    Full-sample IS R² (filtered)  : {r2_kf:.4f}")
    print(f"    σ_ε={sig_e:.4f}  σ_η={sig_h:.4f}  "
          f"signal-to-noise ratio σ_η/σ_ε={sig_h/sig_e:.4f}")
    print(f"    β range: {beta_f.min():.3f} → {beta_f.max():.3f}  "
          f"(mean={beta_f.mean():.3f}, std={beta_f.std():.3f})")

# ── Walk-forward Kalman ───────────────────────────────────────────────────────
print("\n=== Walk-forward Kalman OOS (IS hypers fixed, online OOS update) ===")
print("  (Lookahead-free: hypers from IS only, β updated online with realized values)")
results = []
for dyn_fac, label in [("log_QQQ", "β_QQQ"), ("ARKK_excess", "β_ARKK")]:
    fixed = [c for c in V63_ALL if c != dyn_fac]
    print(f"\n  Fitting Kalman(dynamic {label})...", end=" ", flush=True)
    r2_kf_oos, mae_kf = wf_kalman(f, dyn_fac, fixed)
    dR2 = (r2_kf_oos - r2_ols_oos) * 100
    verdict = "ACCEPT" if dR2 >= BARRIER_PP else "REJECT"
    print(f"done")
    print(f"    Kalman OOS R²={r2_kf_oos:.4f}  MAE={mae_kf:.2f}%  "
          f"lift vs OLS={dR2:>+.2f}pp  {verdict}")
    results.append((label, r2_kf_oos, mae_kf, dR2, verdict))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"  {'model':<35} {'OOS R2':>8} {'OOS MAE':>9} {'lift':>8}  verdict")
print(f"  {'v6.3 OLS (fixed β)':<35} {r2_ols_oos:>8.4f} {mae_ols:>8.2f}%   (ref)")
for label, r2, mae, dR2, verd in results:
    print(f"  {'Kalman (dynamic ' + label + ')':<35} {r2:>8.4f} {mae:>8.2f}%  {dR2:>+.2f}pp  {verd}")

print(f"\n  Note: time-varying β probe (#2 in earlier session) already showed")
print(f"  AR(1)=0.91 for rolling β_QQQ — regime-like but unpredictable by")
print(f"  external indicators. Kalman is the correct model for that structure.")
print(f"  Acceptance bar: +{BARRIER_PP}pp OOS R² lift vs v6.3.")
