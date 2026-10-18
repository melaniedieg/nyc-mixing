"""
estimate_temporal_dependence.py (scoping/diagnostic, Note 10)

Direct, reproducible test of the independence assumption pool_persistence.py's docstring flags as
"CAVEAT STILL OPEN": the one-way random-effects pooling treats a venue's eligible months as
independent replicates of r_vm = mu_v + u_v + eps_vm. The README's "independence assumption --
quantified, NOT yet fixed" section cites a "+0.057 lag-1, fading by lag 2+" finding, but no script
reproducing that number on r_vm itself (as opposed to the related-but-different quintile-share
check in estimate_overdispersion.py/Note 5) is committed anywhere in this repo -- it reads like an
earlier, uncommitted scratch check. Given this session's own pool_persistence.py staging-bug
lesson (never trust a remembered number without rerunning it against current, corrected data),
this script measures it fresh, directly on r_vm = improbable_copresence_cv, across ALL available
lags (not just lag-1), on the current Note-7-buffered/Note-5-DM-corrected pipeline.

Method (mirrors estimate_overdispersion.py/Note 5's design so the same demeaning-artifact
benchmark applies): for the T=8 balanced panel (venues eligible in every one of Dec 2025-Jul
2026 -- same 7,838-venue panel used throughout this project's independence checks), demean each
venue's own r_vm series by its own T-month mean. Under a TRUE independence null, that demeaning
mechanically forces correlation of exactly -1/(T-1) between any two months, REGARDLESS OF LAG (not
just adjacent ones) -- so lag-2, lag-3, etc. all share the same -1/(T-1) benchmark, and any
deviation from it at a given lag is real structure, not a demeaning artifact. Pooled correlation is
computed both on raw demeaned residuals and on residuals standardized by sqrt(var_combined) (the
SAME visitor-sampling + model-refit variance that feeds the ADOPTED pooling weight), so a handful
of very-low-variance venues can't dominate the pooled estimate.

Input:  OUTPUT_DIR/results_mixing_<PERIOD>.pkl, OUTPUT_DIR/visitor_sampling_variance_dm_<PERIOD>.csv,
        OUTPUT_DIR/model_prediction_variance_<PERIOD>.csv -- the same three inputs
        pool_persistence.py reads, for all 8 periods.
Output: prints the full per-lag breakdown; saves {"rho_lag1_point", "rho_lag1_raw",
        "rho_lag1_std", "lag_excess_raw", "lag_excess_std", "null_benchmark",
        "n_balanced_panel_venues"} to OUTPUT_DIR/temporal_dependence_estimate.pkl. rho_lag1_point
        (mean of the raw and variance-standardized lag-1 excess estimates) is the ADOPTED
        single-point rho used by pool_persistence.py/consolidate_fragments.py's GLS pooling
        (Note 10) -- same single-point-from-an-honestly-reported-range convention as S_hat in
        overdispersion_estimate.pkl.
"""
import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
T = len(PERIODS)

frames = {}
for mm in PERIODS:
    d = pd.read_pickle(os.path.join(OUTPUT_DIR, f"results_mixing_{mm}.pkl"))
    rf = d["results_full"][["placekey", "improbable_copresence_cv"]].copy()
    rf = rf.drop_duplicates(subset="placekey")
    vs = pd.read_csv(os.path.join(OUTPUT_DIR, f"visitor_sampling_variance_dm_{mm}.csv"))[
        ["placekey", "resid_vs_var_dm"]
    ].rename(columns={"resid_vs_var_dm": "resid_vs_var"})
    rf = rf.merge(vs, on="placekey", how="left")
    mv = pd.read_csv(os.path.join(OUTPUT_DIR, f"model_prediction_variance_{mm}.csv"))[
        ["placekey", "model_pred_var"]
    ]
    rf = rf.merge(mv, on="placekey", how="left")
    rf["var_combined"] = rf["resid_vs_var"] + rf["model_pred_var"].fillna(0.0)
    frames[mm] = rf.set_index("placekey")
    print(f"{mm}: {len(rf)} eligible venues")

common = frames[PERIODS[0]].index
for mm in PERIODS[1:]:
    common = common.intersection(frames[mm].index)
print(f"\nbalanced T={T} panel: {len(common)} venues present in EVERY period "
      f"(same panel used for the independence/overdispersion checks elsewhere in this project)")

r = {mm: frames[mm].loc[common, "improbable_copresence_cv"].to_numpy() for mm in PERIODS}
v = {mm: frames[mm].loc[common, "var_combined"].to_numpy() for mm in PERIODS}

rbar = sum(r[mm] for mm in PERIODS) / T
d_raw = {mm: r[mm] - rbar for mm in PERIODS}
with np.errstate(invalid="ignore", divide="ignore"):
    d_std = {mm: d_raw[mm] / np.sqrt(v[mm]) for mm in PERIODS}


def pooled_corr(a, b):
    a = a.ravel(); b = b.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[mask], b[mask])[0, 1], mask.sum()


null_benchmark = -1.0 / (T - 1)
print(f"\ndemeaning-artifact null benchmark for T={T}: -1/(T-1) = {null_benchmark:.4f} "
      f"(applies at EVERY lag under true independence, not just lag 1)")

lag_excess_raw, lag_excess_std = [], []
print("\n--- raw demeaned r_vm, by lag ---")
for lag in range(1, T):
    rs = [pooled_corr(d_raw[PERIODS[i]], d_raw[PERIODS[i + lag]])[0] for i in range(T - lag)]
    mean_r = np.mean(rs)
    excess = mean_r - null_benchmark
    lag_excess_raw.append(excess)
    print(f"lag {lag} ({T - lag} pairs): mean r={mean_r:+.4f}  excess over benchmark={excess:+.4f}  "
          f"(per-pair: {['%+.4f' % x for x in rs]})")

print("\n--- standardized (d_vm / sqrt(var_combined)), by lag ---")
for lag in range(1, T):
    rs = [pooled_corr(d_std[PERIODS[i]], d_std[PERIODS[i + lag]])[0] for i in range(T - lag)]
    mean_r = np.mean(rs)
    excess = mean_r - null_benchmark
    lag_excess_std.append(excess)
    print(f"lag {lag} ({T - lag} pairs): mean r={mean_r:+.4f}  excess over benchmark={excess:+.4f}  "
          f"(per-pair: {['%+.4f' % x for x in rs]})")

#ADOPTED single-point rho (Note 10): mean of the raw and variance-standardized lag-1 excess
#estimates -- the only lag with a real, consistent signal (lag 2+ sits at or below the
#demeaning-artifact benchmark, i.e. no detectable excess). Same single-point-from-an-honestly-
#reported-range convention as S_hat's adoption in overdispersion_estimate.pkl.
rho_lag1_point = (lag_excess_raw[0] + lag_excess_std[0]) / 2.0
print(f"\nADOPTED rho_lag1_point (mean of raw/standardized lag-1 excess): {rho_lag1_point:+.4f}")

pd.to_pickle(
    {"rho_lag1_point": rho_lag1_point, "rho_lag1_raw": lag_excess_raw[0],
     "rho_lag1_std": lag_excess_std[0], "lag_excess_raw": lag_excess_raw,
     "lag_excess_std": lag_excess_std, "null_benchmark": null_benchmark,
     "n_balanced_panel_venues": len(common)},
    os.path.join(OUTPUT_DIR, "temporal_dependence_estimate.pkl"),
)
print(f"\nsaved {OUTPUT_DIR}/temporal_dependence_estimate.pkl")
