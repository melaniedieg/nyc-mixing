"""
estimate_visitor_sampling_variance_dm.py

Note 5 fix: replaces fit_mixing_model_2026_0X.py's BOOTSTRAP 2/2 (plain Multinomial(n, p)
visitor-sampling draws) with a Dirichlet-multinomial draw, using the S_hat this project's
overdispersion check now finds real evidence FOR (see estimate_overdispersion.py's revised
docstring -- corrected VIF ~1.48 on the full 8-period balanced panel, vs. the old 3-month
check's 0.91). Standalone script, same pattern as estimate_model_prediction_variance.py: reuses
each period's already-computed expected_mixing_cv (the fixed, out-of-fold CV prediction) rather
than touching fit_mixing_model_2026_0X.py's own CV/model-refit-bootstrap code, which this change
doesn't affect.

Method: standard two-stage representation of a Dirichlet-multinomial draw -- for each bootstrap
iteration and each venue, first draw a randomized probability vector p_tilde ~ Dirichlet(S_hat *
p_v), THEN draw resampled_counts ~ Multinomial(n_v, p_tilde), instead of drawing directly from
Multinomial(n_v, p_v). This reproduces exactly the documented VIF(n) = (S+n)/(S+1) variance-
inflation relationship: at the eligibility floor (n~50-80) it barely moves anything (~3-7%
variance increase at the adopted S_hat), but for a high-traffic venue (n~2,000-10,000) it
inflates variance several-fold -- see estimate_overdispersion.py's docstring for the real numbers.
That means this correction hits big, frequently-visited venues much harder than small ones, not a
flat percentage bump everywhere.

S_hat is read from OUTPUT_DIR/overdispersion_estimate.pkl (produced by estimate_overdispersion.py,
which must be run first), or can be overridden via the S_HAT environment variable for sensitivity
runs against the documented range (S_hat_low=503.9 ... S_hat_high=1451.9).

Tracks EVERY eligible venue, same reasoning as estimate_model_prediction_variance.py: this is a
required input to pool_persistence.py/consolidate_fragments.py for every venue-period, not an
optional check on a flagged subset.

Input:  OUTPUT_DIR/model_ready_<MONTH>.pkl (same file fit_mixing_model_2026_0X.py reads), and
        OUTPUT_DIR/results_mixing_<MONTH>.pkl (for expected_mixing_cv, the fixed out-of-fold
        prediction fit_mixing_model_2026_0X.py already computed).
        OUTPUT_DIR/overdispersion_estimate.pkl (for the adopted S_hat).
Output: OUTPUT_DIR/visitor_sampling_variance_dm_<MONTH>.csv -- one row per eligible venue that
        month: placekey, month, resid_vs_mean_dm, resid_vs_var_dm. pool_persistence.py and
        consolidate_fragments.py now read resid_vs_mean/resid_vs_var from here instead of from
        results_mixing_<MONTH>.pkl's plain-multinomial columns.
"""
import os
import time
import numpy as np
import pandas as pd

MONTH = os.environ["MONTH"]
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
N_BOOT_VS = int(os.environ.get("N_BOOT_VS", "500"))  #same as fit_mixing_model_2026_0X.py's own

od = pd.read_pickle(os.path.join(OUTPUT_DIR, "overdispersion_estimate.pkl"))
S_hat = float(os.environ.get("S_HAT", od["S_hat"]))
print(f"{MONTH}: using S_hat={S_hat:.2f} (adopted estimate from overdispersion_estimate.pkl "
      f"unless overridden by $S_HAT)")

t0 = time.time()
quintile_cols = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{MONTH}.pkl"))
rfull = pd.read_pickle(os.path.join(OUTPUT_DIR, f"results_mixing_{MONTH}.pkl"))["results_full"]
df = df.merge(rfull[["placekey", "expected_mixing_cv"]], on="placekey", how="inner")
print(f"[{time.time()-t0:.0f}s] {MONTH}: {len(df)} venues (after matching to results_full)")

q_counts = df[quintile_cols].to_numpy(dtype=float)
q_totals = q_counts.sum(axis=1)
with np.errstate(invalid="ignore", divide="ignore"):
    q_pvals = np.where(q_totals[:, None] > 0, q_counts / np.where(q_totals[:, None] == 0, 1, q_totals[:, None]), 0.2)
expected_cv_arr = df["expected_mixing_cv"].to_numpy()
placekeys = df["placekey"].to_numpy()
LOG5 = np.log(5)

resid_vs_draws = np.full((len(df), N_BOOT_VS), np.nan)
rng = np.random.RandomState(1)  #same seed convention as fit_mixing_model_2026_0X.py's rng2

#alpha = S_hat * p, per venue -- precompute once, reused every draw (p itself doesn't change,
#only the Dirichlet draw of p_tilde around it does). Floored at a tiny epsilon so a quintile with
#exactly zero observed devices that period (alpha=0, which Dirichlet rejects) still gets a
#well-defined, near-zero draw probability instead of crashing -- negligible effect on any venue
#that actually has devices in that quintile.
alpha = np.clip(S_hat * q_pvals, 1e-6, None)  #[venue, 5]

for b in range(N_BOOT_VS):
    #two-stage Dirichlet-multinomial draw, per venue (can't batch across venues in one
    #np.random.dirichlet call since each venue has its own alpha vector).
    resampled_counts = np.vstack([
        rng.multinomial(int(round(n_b)), rng.dirichlet(a_b)) if n_b > 0 else np.zeros(5)
        for n_b, a_b in zip(q_totals, alpha)
    ]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled_props = np.where(
            q_totals[:, None] > 0,
            resampled_counts / np.where(q_totals[:, None] == 0, 1, q_totals[:, None]),
            0.2,
        )
    resampled_props_clipped = np.clip(resampled_props, 1e-12, None)
    entropy = -(resampled_props_clipped * np.log(resampled_props_clipped)).sum(axis=1)
    y_vs = entropy / LOG5
    resid_vs_draws[:, b] = y_vs - expected_cv_arr
    if (b + 1) % 100 == 0 or b == N_BOOT_VS - 1:
        print(f"[{time.time()-t0:.0f}s] {MONTH}: DM visitor-sampling draw {b+1}/{N_BOOT_VS} done")

out = pd.DataFrame({
    "placekey": placekeys, "month": MONTH,
    "resid_vs_mean_dm": resid_vs_draws.mean(axis=1),
    "resid_vs_var_dm": resid_vs_draws.var(axis=1, ddof=1),
})
out.to_csv(os.path.join(OUTPUT_DIR, f"visitor_sampling_variance_dm_{MONTH}.csv"), index=False)
print(f"\n[{time.time()-t0:.0f}s] {MONTH}: saved {len(out)} venues, "
      f"median resid_vs_var_dm={out['resid_vs_var_dm'].median():.6f}")
print(f"saved {OUTPUT_DIR}/visitor_sampling_variance_dm_{MONTH}.csv")
