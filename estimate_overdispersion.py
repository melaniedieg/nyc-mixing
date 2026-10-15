"""
estimate_overdispersion.py

Point 3 follow-up (REVISED after a third round of external review -- Note 5): checks the
independence/clustering caveat on fit_mixing_model_2026_0X.py's visitor-sampling bootstrap, using
the one piece of real evidence available for it -- the same venues appear across months, so a
venue's own month-to-month movement in quintile shares can be compared against what a plain
multinomial (independent-draws) model would predict for its device count. Any excess beyond that
prediction is real evidence of clustering (devices from the same household/workplace/tour group
visiting together).

REVISION HISTORY, read honestly: the original version of this script used only the first three
real months available (May/June/July 2026) and found VIF_obs=0.91 -- read at the time as real
evidence AGAINST clustering, since the comparison conflates true behavioral change with sampling
noise and can only inflate the ratio, never shrink it below the naive multinomial prediction.
That conclusion rested on an assumption this script never actually tested: that a venue's
different months are independent of each other once you've accounted for its own fitted mean.
With only 3 months (2 lag-pairs at most, and only 1 for most venues), there wasn't enough data to
test that assumption directly, so it was left as a documented-but-unverified caveat.

Now that all 8 periods (Dec 2025-Jul 2026) exist, this script tests it directly: for the 7,838
venues present in EVERY period (the same balanced panel sensitivity_spatial_block_bootstrap.py's
sibling independence test uses), it measures the actual lag-1 correlation between adjacent
months' quintile-share deviations from each venue's own period mean, benchmarked against the
EXACT theoretical correlation that demeaning a balanced T-period panel mechanically forces under
pure independent noise (-1/(T-1), the same fact used for the serial-correlation check in Note 4) --
so the comparison isolates genuine excess correlation from that artifact, not just raw correlation.

FINDING: real, consistently positive excess serial correlation across all 7 available lag-pairs
(+0.029 to +0.291, averaging +0.186 -- see the per-pair printout below), and a raw VIF_obs on this
fuller panel of 1.21 BEFORE even correcting for it (vs. the old 3-month panel's 0.91) -- i.e. the
original "no detectable clustering" conclusion doesn't survive once 8 months are actually used
instead of 3. The corrected VIF (accounting for the fact that E[X^2_v] under a null WITH real
serial correlation rho is (K-1)(M_v-1)(1-rho), not (K-1)(M_v-1)) comes out at roughly 1.24-1.70
across the range of per-pair rho estimates, averaging ~1.48 at the mean estimate -- real evidence
FOR extra-multinomial clustering, reversing the old read. ADOPTED: S_hat and rho_hat below are
fit from the mean-estimate corrected VIF (1.48), the same kind of single-point-estimate-from-an-
honestly-reported-range choice as var_combined's adoption in Note 4 -- see the sensitivity
printout for how much this would move under the low/high ends of the 7-pair range instead.
estimate_visitor_sampling_variance_dm.py uses this script's saved S_hat to apply a proper
Dirichlet-multinomial correction to the visitor-sampling bootstrap -- see that script and
pool_persistence.py/consolidate_fragments.py for how it now feeds the ADOPTED pooling weight.

Caveat on the correction itself, stated plainly: rho is estimated from only 7 lag-pairs (each
itself pooled across 7,838 venues x 5 quintiles, so NOT 7 independent data points about the same
underlying rho -- still a short panel in the dimension that matters for estimating a SERIAL
correlation, which is "how many lag-pairs," not "how many venues"). The 7 per-pair estimates are
reasonably consistent (all positive, no sign flips) but span a real range, so the adopted S_hat
should be read as "best available estimate," not "precisely known."

Method -- the standard multi-sample chi-square test of homogeneity (not a simple per-month
goodness-of-fit test), which matters because each venue's "true" rate has to be ESTIMATED from
its own months of data:
  1. For a venue with M_v months of data, pbar_v,k = the device-count-weighted average quintile
     share across those months.
  2. X^2_v = sum over that venue's months of n_m * sum_k (p_m,k - pbar_v,k)^2 / pbar_v,k.
  3. Because pbar_v was fit FROM those same M_v months, the classical reference under a null of
     plain multinomial sampling AND independent months is E[X^2_v] = (K-1)*(M_v-1), summed per
     venue -- undercorrects if months are positively serially correlated (see above), which this
     revision corrects for using the directly-measured rho.
  4. VIF_obs = sum(X^2_v) / sum((K-1)*(M_v-1)) [raw]; VIF_corrected = VIF_obs / (1 - rho_hat).
  5. Dirichlet-multinomial's variance-inflation factor VIF(n) = (S+n)/(S+1) is linear in n, so
     S_hat = (n_bar - VIF_corrected) / (VIF_corrected - 1), from the CORRECTED ratio.

Input: OUTPUT_DIR/model_ready_<PERIOD>.pkl for every PERIOD below (from each period's own
       build_features_2026_0X.py -- must include q1_devices..q5_devices).
Output: prints the full breakdown; saves {"S_hat", "rho_hat", "vif_obs_raw", "vif_corrected",
        "vif_corrected_low", "vif_corrected_high", "mean_excess_rho", "lag1_excess_by_pair",
        "n_bar", "n_balanced_panel_venues"} to OUTPUT_DIR/overdispersion_estimate.pkl.
"""
import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
QCOLS = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
K = 5

frames = {}
for mm in PERIODS:
    df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{mm}.pkl"))[["placekey"] + QCOLS].copy()
    df["n"] = df[QCOLS].sum(axis=1)
    df = df[df["n"] > 0].copy()
    for k in QCOLS:
        df[k.replace("_devices", "_p")] = df[k] / df["n"]
    frames[mm] = df.set_index("placekey")
    print(f"{mm}: {len(df)} venues with n>0")

pcols = [k.replace("_devices", "_p") for k in QCOLS]

common = frames[PERIODS[0]].index
for mm in PERIODS[1:]:
    common = common.intersection(frames[mm].index)
T = len(PERIODS)
print(f"\nbalanced T={T} panel: {len(common)} venues present in EVERY period "
      f"(same panel used for the independence check elsewhere in this project)")

p = {mm: frames[mm].loc[common, pcols].to_numpy() for mm in PERIODS}
n = {mm: frames[mm].loc[common, "n"].to_numpy() for mm in PERIODS}
num = sum(p[mm] * n[mm][:, None] for mm in PERIODS)
den = sum(n[mm][:, None] for mm in PERIODS)
pbar = num / den

d = {mm: p[mm] - pbar for mm in PERIODS}
with np.errstate(invalid="ignore", divide="ignore"):
    se = {mm: np.sqrt(np.where(pbar > 1e-12, pbar * (1 - pbar), np.nan) / n[mm][:, None]) for mm in PERIODS}
z = {mm: d[mm] / se[mm] for mm in PERIODS}


def pooled_corr(a, b):
    a = a.ravel(); b = b.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[mask], b[mask])[0, 1]


null_benchmark = -1.0 / (T - 1)
print(f"\ndemeaning-artifact null benchmark for T={T}: -1/(T-1) = {null_benchmark:.4f}")
print("lag-1 correlations (standardized z), all 7 adjacent period-pairs:")
excess_by_pair = []
for i in range(T - 1):
    m1, m2 = PERIODS[i], PERIODS[i + 1]
    r = pooled_corr(z[m1], z[m2])
    excess = r - null_benchmark
    excess_by_pair.append(excess)
    print(f"  {m1} -> {m2}: r={r:.4f}  excess over benchmark={excess:+.4f}")

excess_by_pair = np.array(excess_by_pair)
mean_excess = excess_by_pair.mean()
print(f"\nmean excess over benchmark across all 7 pairs: {mean_excess:+.4f} "
      f"(range [{excess_by_pair.min():+.4f}, {excess_by_pair.max():+.4f}])")

#VIF_obs on this balanced panel, raw (uncorrected) and corrected for the measured excess rho.
terms = sum(
    np.where(pbar > 1e-12, (p[mm] - pbar) ** 2 / np.where(pbar == 0, 1, pbar), 0.0) * n[mm][:, None]
    for mm in PERIODS
)
total_x2 = terms.sum()
total_df_classical = (K - 1) * (T - 1) * len(common)
vif_obs_raw = total_x2 / total_df_classical
n_bar = np.mean([n[mm].mean() for mm in PERIODS])

print(f"\nVIF_obs, RAW (uncorrected, same classical reference the old 3-month version used): "
      f"{vif_obs_raw:.4f}  (total X^2={total_x2:.1f}, {len(common)} venues)")
print(f"mean device count per venue-period in this panel: {n_bar:.1f}")

vif_corrected = vif_obs_raw / (1 - mean_excess)
vif_corrected_low = vif_obs_raw / (1 - excess_by_pair.min())
vif_corrected_high = vif_obs_raw / (1 - excess_by_pair.max())
print(f"\nVIF_obs, CORRECTED for measured serial correlation (ADOPTED, mean-estimate rho): "
      f"{vif_corrected:.4f}")
print(f"  sensitivity range across the 7 pairs' own excess estimates: "
      f"[{vif_corrected_low:.4f}, {vif_corrected_high:.4f}]")

if vif_corrected <= 1.0:
    print("\nVIF_corrected <= 1: no evidence of extra-multinomial clustering even after correction.")
    S_hat = 1e6
else:
    S_hat = (n_bar - vif_corrected) / (vif_corrected - 1)
    print(f"\nS_hat (ADOPTED, fitted Dirichlet-multinomial concentration, from the CORRECTED VIF): "
          f"{S_hat:.2f}")
S_hat_low = (n_bar - vif_corrected_low) / (vif_corrected_low - 1) if vif_corrected_low > 1 else 1e6
S_hat_high = (n_bar - vif_corrected_high) / (vif_corrected_high - 1) if vif_corrected_high > 1 else 1e6
rho_hat = 1.0 / (1.0 + S_hat)
print(f"implied intra-venue correlation rho_dm = 1/(1+S): {rho_hat:.5f}")
print(f"sensitivity range for S_hat across the 7-pair range: [{S_hat_high:.1f}, {S_hat_low:.1f}] "
      f"(note: higher corrected VIF -> lower S_hat -> MORE correction)")

pd.to_pickle(
    {"S_hat": S_hat, "S_hat_low": S_hat_low, "S_hat_high": S_hat_high, "rho_hat": rho_hat,
     "vif_obs_raw": vif_obs_raw, "vif_corrected": vif_corrected,
     "vif_corrected_low": vif_corrected_low, "vif_corrected_high": vif_corrected_high,
     "mean_excess_rho": mean_excess, "lag1_excess_by_pair": excess_by_pair.tolist(),
     "n_bar": n_bar, "n_balanced_panel_venues": len(common)},
    os.path.join(OUTPUT_DIR, "overdispersion_estimate.pkl"),
)
print(f"\nsaved {os.path.join(OUTPUT_DIR, 'overdispersion_estimate.pkl')}")
