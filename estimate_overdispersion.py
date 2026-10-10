"""
estimate_overdispersion.py

Point 3 follow-up: checks the independence/clustering caveat on fit_mixing_model_2026_0X.py's
visitor-sampling bootstrap, using the one piece of real evidence available for it -- the same
venues appear in all three real months (May/June/July 2026), so a venue's own month-to-month
movement in quintile shares can be compared against what a plain multinomial (independent-draws)
model would predict for its device count. Any excess beyond that prediction is real evidence of
clustering (devices from the same household/workplace/tour group visiting together).

Caveat on this check itself, stated up front: comparing REAL months conflates true behavioral
change (a venue's actual visitor mix shifting between May and July) with pure sampling noise.
Both would show up as "more variance than multinomial predicts." So if anything, this method
should OVERSTATE overdispersion, not understate it -- which matters for how to read the result.

Method -- the standard multi-sample chi-square test of homogeneity (not a simple per-month
goodness-of-fit test), which matters because each venue's "true" rate has to be ESTIMATED from
its own few months of data:
  1. For a venue with M_v months of data (2 or 3), pbar_v,k = the device-count-weighted average
     quintile share across those months.
  2. X^2_v = sum over that venue's months of n_m * sum_k (p_m,k - pbar_v,k)^2 / pbar_v,k.
  3. Because pbar_v was fit FROM those same M_v months, the correct reference under the null
     (plain multinomial sampling, no overdispersion) is NOT (K-1) per month -- that undercorrects,
     since the estimated mean mechanically fits the observed points better than the true
     population mean would. The correct classical result (chi-square test of homogeneity across
     M_v dependent-on-the-same-reference groups) is E[X^2_v] = (K-1)*(M_v-1), summed per venue.
  4. VIF_obs = sum(X^2_v across all venues) / sum((K-1)*(M_v-1) across all venues).
  5. Dirichlet-multinomial's variance-inflation factor VIF(n) = (S+n)/(S+1) is linear in n, so
     E[VIF] over pooled data = VIF(n_bar), and S_hat = (n_bar - VIF_obs) / (VIF_obs - 1).

Run across the real May/June/July 2026 data (28,172 venues appearing in 2+ of the 3 months,
81,074 venue-months, sanity-checked against the 17,001 of those venues with 100+ devices in
every month they're present): VIF_obs = 0.91 both times. A ratio at or below 1.0 means real
month-to-month movement is no larger -- if anything slightly smaller -- than independent-sampling
noise alone predicts for these device counts. Because the comparison conflates sampling noise
with genuine behavioral change (which could only inflate the ratio, not shrink it), this is real
evidence against clustering showing up at a detectable level here, not just an inconclusive null
result -- so fit_mixing_model_2026_0X.py's visitor-sampling bootstrap is NOT given a
Dirichlet-multinomial correction; the data doesn't support one, and forcing one in would trade a
real number for an invented one. See README.md's Known Limitations section for the full writeup,
including what this test can and can't rule out.

Input: OUTPUT_DIR_<MM>/model_ready_2026_<MM>.pkl for MM in 05, 06, 07 (from each month's own
       build_features_2026_<MM>.py -- must include q1_devices..q5_devices).
Output: prints the full breakdown; saves {"S_hat", "rho_hat", "vif_obs", "vif_naive", "n_bar",
        "n_venue_months", "n_repeat_venues", "vif_obs_big_n", "n_venues_big_n"} to
        OUTPUT_DIR_06/overdispersion_estimate.pkl (June's output dir, arbitrarily -- this is a
        cross-month result, not specific to one month).
"""
import os
import pandas as pd
import numpy as np

OUTPUT_DIRS = {
    "05": os.environ.get("OUTPUT_DIR_05", "./output_05"),
    "06": os.environ.get("OUTPUT_DIR_06", "./output_06"),
    "07": os.environ.get("OUTPUT_DIR_07", "./output_07"),
}
QCOLS = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
K = 5

frames = {}
for mm, out_dir in OUTPUT_DIRS.items():
    path = os.path.join(out_dir, f"model_ready_2026_{mm}.pkl")
    df = pd.read_pickle(path)[["placekey"] + QCOLS].copy()
    df["n"] = df[QCOLS].sum(axis=1)
    df = df[df["n"] > 0].copy()
    for k in QCOLS:
        df[k.replace("_devices", "_p")] = df[k] / df["n"]
    df["month"] = mm
    frames[mm] = df
    print(f"{mm}: {len(df)} venues with n>0")

all_months = pd.concat(frames.values(), ignore_index=True)
pcols = [k.replace("_devices", "_p") for k in QCOLS]

counts = all_months.groupby("placekey").size()
repeat_pks = counts[counts >= 2].index
print(f"\n{len(repeat_pks)} venues appear in >=2 of the 3 months (out of "
      f"{all_months['placekey'].nunique()} total distinct venues)")
print("months-present distribution for repeat venues:")
print(counts[counts >= 2].value_counts().sort_index())

sub = all_months[all_months["placekey"].isin(repeat_pks)].copy()

def weighted_avg(g):
    w = g["n"].to_numpy()
    return pd.Series({pk: np.average(g[pk].to_numpy(), weights=w) for pk in pcols})

pbar = sub.groupby("placekey").apply(weighted_avg, include_groups=False)
pbar.columns = [f"pbar_{c}" for c in pcols]
sub = sub.merge(pbar, left_on="placekey", right_index=True)

pbar_arr = sub[[f"pbar_{c}" for c in pcols]].to_numpy()
p_arr = sub[pcols].to_numpy()
n_arr = sub["n"].to_numpy()

with np.errstate(invalid="ignore", divide="ignore"):
    terms = np.where(pbar_arr > 1e-12, (p_arr - pbar_arr) ** 2 / np.where(pbar_arr == 0, 1, pbar_arr), 0.0)
x2_per_month = n_arr * terms.sum(axis=1)

vif_naive = x2_per_month.mean() / (K - 1)
print(f"\n[NAIVE, biased] mean X^2 per venue-month: {x2_per_month.mean():.3f} vs expected {K-1} "
      f"-> VIF_naive={vif_naive:.3f} (under-corrects for pbar being fit from the same data)")

x2_by_venue = pd.Series(x2_per_month, index=sub.index).groupby(sub["placekey"]).sum()
m_by_venue = sub.groupby("placekey").size()
df_by_venue = (K - 1) * (m_by_venue - 1)
total_x2 = x2_by_venue.sum()
total_df = df_by_venue.sum()
vif_obs = total_x2 / total_df
n_bar = n_arr.mean()

print(f"\n[CORRECTED] total X^2 across {len(x2_by_venue)} venues: {total_x2:.1f}, "
      f"total degrees of freedom (K-1)*(M_v-1) summed: {total_df:.0f}")
print(f"VIF_obs (observed/expected variance ratio, properly corrected): {vif_obs:.4f}")
print(f"mean device count per venue-month in this comparison: {n_bar:.1f}")

if vif_obs <= 1.0:
    print(f"\nVIF_obs <= 1: no evidence of extra-multinomial clustering detectable here -- "
          f"S_hat set to a very large value (no correction applied to the visitor-sampling "
          f"bootstrap).")
    S_hat = 1e6
else:
    S_hat = (n_bar - vif_obs) / (vif_obs - 1)
    print(f"\nS_hat (fitted Dirichlet-multinomial concentration): {S_hat:.2f}")

rho_hat = 1.0 / (1.0 + S_hat)
print(f"implied intra-venue correlation rho = 1/(1+S): {rho_hat:.5f}")

big_pks = sub.groupby("placekey")["n"].min()
big_pks = big_pks[big_pks >= 100].index
bsub = sub[sub["placekey"].isin(big_pks)]
bx2_by_venue = x2_by_venue.loc[bsub["placekey"].unique()]
bm_by_venue = m_by_venue.loc[bsub["placekey"].unique()]
bdf = ((K - 1) * (bm_by_venue - 1)).sum()
bvif = bx2_by_venue.sum() / bdf
bn_bar = bsub["n"].mean()
print(f"\nsanity check, venues with n>=100 in every month present ({len(big_pks)} venues): "
      f"VIF_obs={bvif:.4f}, n_bar={bn_bar:.1f}")

pd.to_pickle(
    {"S_hat": S_hat, "rho_hat": rho_hat, "vif_obs": vif_obs, "vif_naive": vif_naive, "n_bar": n_bar,
     "n_venue_months": len(sub), "n_repeat_venues": len(repeat_pks), "vif_obs_big_n": bvif,
     "n_venues_big_n": len(big_pks)},
    os.path.join(OUTPUT_DIRS["06"], "overdispersion_estimate.pkl"),
)
print(f"\nsaved to {os.path.join(OUTPUT_DIRS['06'], 'overdispersion_estimate.pkl')}")
