"""
validate_temporal_estimator.py

External review point #1: estimate_temporal_dependence.py's RHO_TEMPORAL=0.1683 (adopted into
pool_persistence.py's GLS pooling covariance, Note 10) was estimated ONCE on real data. Real data
can't tell us whether the ESTIMATOR ITSELF is trustworthy -- whether it would report a spurious
nonzero rho even when the true cross-month correlation is exactly zero, especially once realistic
heteroskedastic, per-venue-persistent variance is in the mix (a venue's device count, and so its
variance, doesn't reshuffle every month -- it's the same size business all year).

This script reimplements estimate_temporal_dependence.py's EXACT procedure (demeaning, the
-1/(T-1) null benchmark, raw + variance-standardized lag estimates, the rho_lag1_point = mean of
raw/standardized lag-1 excess) and runs it many times against SYNTHETIC panels where the true
answer is known by construction, at the real panel's actual scale (N=7838 venues, T=8 months --
same as the real balanced panel) and compares the simulated null's sampling distribution against
the real-data result.

Six scenarios:
  A. NULL, homoskedastic: eps_vm iid N(0,1) for every venue-month. The easiest possible case --
     every venue has textbook-equal, period-to-period-independent noise.
  B. NULL, heteroskedastic + PERSISTENT variance: each venue gets its own sigma_v^2 (lognormal,
     mimicking device-count-driven variance spread), THE SAME sigma_v every month (persistent,
     like a real venue's device count doesn't reshuffle monthly) -- but still genuinely
     independent draws across months given sigma_v. This is the realistic case the review's
     concern is really about: does persistent-but-independent variance masquerade as temporal
     correlation, especially in the variance-STANDARDIZED check (d_vm / sqrt(var_combined))?
  C. NULL, heteroskedastic + NOISY variance ESTIMATE: same generative process as B, but the
     var_combined fed to the standardization step is a noisy ESTIMATE of the true sigma_v^2
     (chi-square-noised, mimicking real-world resid_vs_var/model_pred_var estimation error)
     rather than the true value -- checks whether estimation error in the variance itself (not
     just its heteroskedasticity) can fool the standardized check.
  D. ALTERNATIVE, known lag-1-only signal: inject a genuine MA(1)-style lag-1 correlation of
     rho_true=0.15 (chosen close to the real adopted 0.1683) via eps_vm = sqrt(1-rho^2)*z_vm +
     rho*z_{v,m-1}, homoskedastic otherwise -- checks the estimator has the POWER to recover a
     real signal of about the observed magnitude, and that it correctly reports ~0 excess at
     lag 2+ (this MA(1) design has NO lag-2+ correlation by construction, so any lag-2+ excess
     the estimator reports here would itself be an estimator artifact).

A and B/C confirm rho_lag1_point is not a null artifact (see printed SD-away numbers below). But
D's own lag2..7 excess SHRINKS toward zero with increasing lag -- the OPPOSITE of what the real
data shows (lag2..7 excess GROWS more negative with lag; see the real-data reference printed at
the end). A pure lag-1-only true process cannot be the whole story. Two more scenarios chase that
down:
  E. NULL, per-venue random LINEAR TREND + white noise, NO true month-to-month dependence at all:
     eps_vm = beta_v*(m - 3.5) + z_vm, beta_v ~ N(0, sigma_b^2) (a venue's OWN trajectory drifts
     up or down across the 8-month window, heterogeneously across venues, with no city-wide
     direction), z_vm iid N(0,1). Demeaning does NOT remove a per-venue trend (it only removes a
     constant level), so this alone creates a predictable signature: pairs of nearby months stay
     on the same side of the venue's own linear fit (positive excess at low lag), while pairs
     spanning the full 8-month width straddle opposite sides of it (increasingly negative excess
     at high lag) -- captured exactly by avg(c_i * c_{i+lag}) for centered month index c_m=m-3.5,
     which is +3.75, +1.92, -0.25, -2.75, -5.58, -8.75, -12.25 for lag=1..7: monotonically
     DECREASING, i.e. the same shape the real data shows, with NO genuine serial dependence
     required at all.
  F. COMBINED: scenario E's trend term PLUS a smaller genuine MA(1) lag-1 correlation on top,
     grid-searched (by sum of squared error against the real 7-point lag1..7 vector) over
     sigma_b in [0.09..0.12] and rho_ma in [0.06..0.12]. The best fit (sigma_b=0.11, rho_ma=0.12)
     reproduces the ENTIRE real 7-point pattern closely (see printed comparison) -- substantially
     better than either mechanism alone. This suggests the real "temporal dependence" signal is a
     MIX of genuine short-range (lag-1) serial correlation and per-venue secular drift that the
     current GLS model (Note 10's lag-1-adjacency-only covariance bump) does not represent at
     all. If real, this has two implications worth following up on real data (not done here,
     since that requires fitting an actual per-venue trend term against real r_vm, not a
     synthetic panel): the genuine lag-1 dependence correction may be somewhat smaller than the
     single-point rho_lag1_point=0.1683 currently adopted (part of that number may be trend
     bleeding into the lag-1 pair specifically), and there may be real, currently unmodeled
     negative covariance between FAR-APART months that Note 10's lag-1-only GLS bump ignores.

Each scenario runs N_REPLICATES independent synthetic panels; we report the Monte Carlo mean and
SD of rho_lag1_point (and the lag-2..7 excess vector) across replicates, which is the sampling
distribution SIZE of this exact statistic at this exact panel size under each scenario -- directly
comparable to the single real-data point estimate (+0.1683, lag2+ excess near/below zero).
"""
import numpy as np

T = 8
N = 7838
N_REPLICATES = 300
rng = np.random.RandomState(20261022)


def lag_analysis(r, v):
    """Exact reimplementation of estimate_temporal_dependence.py's demeaning + pooled-correlation
    procedure. r, v: dict of month-index -> (N,) arrays (observed residual, var_combined)."""
    rbar = sum(r[m] for m in range(T)) / T
    d_raw = {m: r[m] - rbar for m in range(T)}
    with np.errstate(invalid="ignore", divide="ignore"):
        d_std = {m: d_raw[m] / np.sqrt(v[m]) for m in range(T)}

    def pooled_corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return np.corrcoef(a[mask], b[mask])[0, 1]

    null_benchmark = -1.0 / (T - 1)
    lag_excess_raw, lag_excess_std = [], []
    for lag in range(1, T):
        rs_raw = [pooled_corr(d_raw[i], d_raw[i + lag]) for i in range(T - lag)]
        rs_std = [pooled_corr(d_std[i], d_std[i + lag]) for i in range(T - lag)]
        lag_excess_raw.append(np.mean(rs_raw) - null_benchmark)
        lag_excess_std.append(np.mean(rs_std) - null_benchmark)
    rho_lag1_point = (lag_excess_raw[0] + lag_excess_std[0]) / 2.0
    return rho_lag1_point, np.array(lag_excess_raw), np.array(lag_excess_std)


def run_scenario(name, gen_eps, gen_var_true, gen_var_fed):
    """gen_eps(rng) -> dict m->eps_vm (N,); gen_var_true/gen_var_fed(rng, eps) -> dict m->var (N,)
    (var_fed is what's handed to the standardization step; var_true is the real generative
    variance, only used to sanity-check the simulation itself, not by the estimator)."""
    rho1_list, lag_raw_list, lag_std_list = [], [], []
    for _ in range(N_REPLICATES):
        eps = gen_eps(rng)
        var_fed = gen_var_fed(rng, eps)
        rho1, lag_raw, lag_std = lag_analysis(eps, var_fed)
        rho1_list.append(rho1)
        lag_raw_list.append(lag_raw)
        lag_std_list.append(lag_std)
    rho1_arr = np.array(rho1_list)
    lag_raw_arr = np.array(lag_raw_list)  # (N_REPLICATES, T-1)
    lag_std_arr = np.array(lag_std_list)
    print(f"\n=== {name} ===")
    print(f"rho_lag1_point across {N_REPLICATES} synthetic panels: "
          f"mean={rho1_arr.mean():+.4f}  sd={rho1_arr.std(ddof=1):.4f}  "
          f"[2.5%ile={np.percentile(rho1_arr, 2.5):+.4f}, 97.5%ile={np.percentile(rho1_arr, 97.5):+.4f}]")
    print(f"  (real adopted value: +0.1683 -- "
          f"{(0.1683 - rho1_arr.mean()) / rho1_arr.std(ddof=1):.1f} simulated-null SDs away)" if "NULL" in name else "")
    for lag in range(1, T):
        rr = lag_raw_arr[:, lag - 1]
        rs = lag_std_arr[:, lag - 1]
        print(f"  lag {lag}: raw excess mean={rr.mean():+.4f} sd={rr.std(ddof=1):.4f}   "
              f"std excess mean={rs.mean():+.4f} sd={rs.std(ddof=1):.4f}")
    return rho1_arr, lag_raw_arr, lag_std_arr


# ---- Scenario A: NULL, homoskedastic ----
def A_eps(rng):
    return {m: rng.normal(0, 1, N) for m in range(T)}


def A_var(rng, eps):
    return {m: np.ones(N) for m in range(T)}


run_scenario("A: NULL, homoskedastic (eps iid N(0,1), var=1 for everyone)", A_eps, A_var, A_var)

# ---- Scenario B: NULL, heteroskedastic + PERSISTENT true variance ----
#one sigma_v^2 per venue (lognormal spread, mimicking device-count-driven variance range seen in
#real resid_vs_var/model_pred_var -- roughly 2 orders of magnitude spread), held FIXED across all
#8 of that venue's months (persistent, like a real venue's device count doesn't reshuffle), but
#draws across months are still genuinely independent given sigma_v.
_sigma2_v = None


def B_eps(rng):
    global _sigma2_v
    _sigma2_v = rng.lognormal(mean=0.0, sigma=1.0, size=N)  # persistent per-venue variance
    return {m: rng.normal(0, np.sqrt(_sigma2_v), N) for m in range(T)}


def B_var_true(rng, eps):
    return {m: _sigma2_v.copy() for m in range(T)}


run_scenario("B: NULL, heteroskedastic + PERSISTENT true variance "
             "(var fed = TRUE sigma_v^2, known exactly)", B_eps, B_var_true, B_var_true)

# ---- Scenario C: NULL, heteroskedastic + NOISY variance ESTIMATE ----
#same generative eps as B, but the variance fed to standardization is a noisy ESTIMATE of
#sigma_v^2 (chi-square-noised with 10 "effective df", a stand-in for real-world
#resid_vs_var/model_pred_var estimation error), independently noised EACH MONTH (an estimation
#error that itself doesn't carry any true cross-month signal).
def C_var_noisy(rng, eps):
    df_eff = 10
    out = {}
    for m in range(T):
        noise = rng.chisquare(df_eff, N) / df_eff
        out[m] = _sigma2_v * noise
    return out


run_scenario("C: NULL, heteroskedastic + NOISY variance ESTIMATE "
             "(same eps as B, var fed = noisy per-month estimate of sigma_v^2)",
             B_eps, B_var_true, C_var_noisy)

# ---- Scenario D: ALTERNATIVE, known lag-1-only signal, rho_true=0.15 ----
RHO_TRUE = 0.15


def D_eps(rng):
    z = {m: rng.normal(0, 1, N) for m in range(T)}
    eps = {0: z[0].copy()}
    for m in range(1, T):
        eps[m] = np.sqrt(1 - RHO_TRUE ** 2) * z[m] + RHO_TRUE * z[m - 1]
    return eps


def D_var(rng, eps):
    return {m: np.ones(N) for m in range(T)}


run_scenario(f"D: ALTERNATIVE, known MA(1) lag-1-only signal, rho_true={RHO_TRUE:+.2f} "
             "(homoskedastic)", D_eps, D_var, D_var)

c = np.arange(T) - (T - 1) / 2.0  # centered month index, used by scenarios E/F

# ---- Scenario E: NULL, per-venue random LINEAR TREND + white noise, no true dependence ----
SIGMA_B_E = 0.11


def E_eps(rng):
    beta_v = rng.normal(0, SIGMA_B_E, N)
    z = {m: rng.normal(0, 1, N) for m in range(T)}
    return {m: beta_v * c[m] + z[m] for m in range(T)}


def E_var(rng, eps):
    return {m: np.ones(N) for m in range(T)}


run_scenario(f"E: NULL, per-venue random LINEAR TREND (sigma_b={SIGMA_B_E}) + white noise, "
             "NO true month-to-month dependence", E_eps, E_var, E_var)

# ---- Scenario F: COMBINED trend + smaller genuine lag-1 correlation ----
#best fit against the real 7-point lag1..7 vector, grid-searched separately (see module
#docstring) over sigma_b in [0.09..0.12] and rho_ma in [0.06..0.12].
SIGMA_B_F = 0.11
RHO_MA_F = 0.12


def F_eps(rng):
    beta_v = rng.normal(0, SIGMA_B_F, N)
    z = {m: rng.normal(0, 1, N) for m in range(T)}
    eps_ma = {0: z[0].copy()}
    for m in range(1, T):
        eps_ma[m] = np.sqrt(1 - RHO_MA_F ** 2) * z[m] + RHO_MA_F * z[m - 1]
    return {m: beta_v * c[m] + eps_ma[m] for m in range(T)}


def F_var(rng, eps):
    return {m: np.ones(N) for m in range(T)}


run_scenario(f"F: COMBINED trend (sigma_b={SIGMA_B_F}) + genuine MA(1) lag-1 correlation "
             f"(rho_ma={RHO_MA_F}) -- best fit to the real 7-point pattern", F_eps, F_var, F_var)

print("\n\n=== Real-data reference (from estimate_temporal_dependence_run.log) ===")
print("rho_lag1_point = +0.1683; lag1..7 raw excess = "
      "[+0.1563, -0.0120, -0.0162, -0.0537, -0.0701, -0.1071, -0.1430]; "
      "lag1..7 std excess = [+0.1804, +0.0210, -0.0159, -0.0419, -0.0385, -0.0785, -0.1475]")
print("Scenario F (combined trend + smaller genuine lag-1 correlation) is the best match to this")
print("full 7-point shape found in this script -- see module docstring for what that implies.")
