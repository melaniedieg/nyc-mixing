"""
pool_persistence.py

Implements the persistence/pooling design from the "point 3" discussion: for every
venue, pools its per-month residual against a random-effects meta-analysis
estimate that separates genuine month-to-month/seasonal variation (tau^2, the
u_vm term) from visitor-sampling noise (the per-month var(r_vm), the epsilon_vm
term) -- r_vm = mu_v + u_vm + eps_vm. ADOPTED METHOD: REML (for tau^2) +
Hartung-Knapp-Sidik-Jonkman (for the pooled mean/CI/p-value) -- see the
"DerSimonian-Laird...read honestly" note below for why. DerSimonian-Laird
(mu_re/var_re/ci_re/tau2/I2/q_bh/fdr_sig_05) is kept as a diagnostic/comparison
column, not the reported result.

POINT ESTIMATE, fixed after external review (see README "Validation checks already
run" -- point-estimate correction): each month's r_vm is the OBSERVED spatial-CV
residual, `improbable_copresence_cv` = income_mixing_score - expected_mixing_cv, NOT
`resid_vs_mean` (the mean of the 500 visitor-sampling bootstrap draws). An earlier
version of this script used resid_vs_mean as r_vm. That was wrong in a specific,
diagnosable way: normalized entropy is concave, so the MEAN of resampled-entropy
draws is systematically a little lower than the entropy of the actually-observed
proportions (a finite-sample/Jensen's-inequality bias), meaning resid_vs_mean quietly
shifts every venue's point estimate downward, worse for low-device venues where the
bootstrap distribution is more skewed. The bootstrap draws are still exactly the
right tool for ESTIMATING VARIANCE (var(r_vm) = resid_vs_var, used below as the
inverse-variance weight) -- they just should never have stood in for the point
estimate itself. If a deliberate bias correction is wanted later (e.g.
2*r_observed - resid_vs_mean), it should be validated by simulation first and labeled
as a correction, not silently substituted.

WHAT THE RESULTING INTERVAL IS AND ISN'T (see README): var(r_vm) here captures
VISITOR-SAMPLING uncertainty only -- "if this venue had drawn a different,
equally-likely sample of its own visitors, how much would its residual move." It
holds the fitted RandomForest's prediction (expected_mixing_cv) completely fixed. It
does NOT capture uncertainty in that fitted expectation itself (how much the
prediction would change under a different training sample, different spatial folds,
different hyperparameters, or a different month's panel composition) -- the
model-refit bootstrap in fit_mixing_model_2026_0X.py measures a related but
different thing (ranking stability under row resampling, not a residual variance)
and is not yet combined into var(r_vm) here. So ci_fe/ci_re below are CONDITIONAL
intervals -- conditional on the fitted contextual model being exactly right -- not
full confidence intervals for the venue's true residual. Report them as such.

Per venue, reports:
  - n_months: how many months it cleared the eligibility floor (results_full
    only has rows for eligible venue-months -- a venue absent some month is
    simply not in that month's results_full, so it's excluded from the pool
    for that month rather than treated as "not mixed").
  - persistence_tier: a plain-language label for what n_months actually supports --
    see PERSISTENCE_TIERS below. None of the current data can earn "annual_persistence"
    (that needs fall/winter months, which don't exist yet); the ceiling right now is
    "current_period_persistence" for a venue eligible in all 4 of April-July.
  - mu_fe, var_fe, ci_fe_lo/hi: fixed-effect (no between-month variance term).
  - tau2, Q, I2: DerSimonian-Laird between-month variance estimate, Cochran's Q,
    and I^2 (% of total variation that's genuine month-to-month movement rather
    than visitor-sampling noise). With only 2-4 months per venue, tau2=0 means "no
    heterogeneity DETECTED" -- usually a power problem, not evidence of none; see
    README for why DerSimonian-Laird + a normal CI is known to be optimistic at
    this few observations. DIAGNOSTIC/COMPARISON ONLY -- kept for reference, not
    the reported result; see tau2_reml/mu_hk/var_hk/ci_hk_lo/hi below for the
    ADOPTED estimate (REML + Hartung-Knapp, which fixes exactly this optimism).
  - mu_re, var_re, ci_re_lo/hi: DerSimonian-Laird random-effects pooled estimate
    and CONDITIONAL CI (widens when tau2>0). DIAGNOSTIC/COMPARISON ONLY, same
    reason as above -- mu_hk/var_hk/ci_hk_lo/hi below are the adopted estimate.
  - tau2_reml, mu_hk, var_hk, ci_hk_lo/hi, p_hk, q_bh_hk, fdr_sig_05_hk, pred_lo/hi:
    the ADOPTED pooling result -- REML estimate of tau^2 (Viechtbauer 2005) plus
    Hartung-Knapp-Sidik-Jonkman's pooled mean/CI (t-distributed with m-1 df,
    widened per Wiksten et al. 2016 so it can't give an implausibly narrow
    interval), and the Benjamini-Hochberg FDR call (q_bh_hk/fdr_sig_05_hk)
    actually used downstream. pred_lo/hi is additionally a prediction interval
    for a hypothetical future month (defined at n_months>=3).
  - prob_positive_re: P(true effect > 0) under a normal approximation to the
    random-effects estimate -- a direct answer to "is this venue persistently
    more/less mixed than expected," not just a flag/no-flag call.
  - same_sign_months: how many of its eligible months have the same residual
    sign as mu_fe (a cheap, assumption-free persistence check alongside the
    formal CI).

Caveat, stated plainly (carried over from the fit_mixing_model patch and the
prior discussion): with only 4 months, tau^2 is estimated from very little
information and will be noisy for any single venue -- reported honestly
alongside mu_re rather than treated as precise. This is NOT yet a 12-month
or seasonal-contrast analysis (no winter month exists yet), and it does NOT
yet distinguish "absent from the POI universe" from "present but ineligible"
for venue-months outside results_full (that bookkeeping -- point 5 from the
"point 3" discussion -- would need the raw per-month analysis_pois universe,
not just model_ready/results_full, and hasn't been built).

Input:  OUTPUT_DIR/results_mixing_2026_<MM>.pkl, one per month (must be from the
        patched fit_mixing_model_2026_<MM>.py, i.e. must contain "results_full").
        Set OUTPUT_DIR to wherever build_features_2026_0X.py /
        fit_mixing_model_2026_0X.py wrote their outputs (same OUTPUT_DIR each
        month -- see those scripts' docstrings); default ./output.
Output: OUTPUT_DIR/pooled_persistence_2026.pkl/.csv -- one row per venue with >=1
        eligible month, sorted by mu_re.
"""
import math
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
#Full "YYYY_MM" period labels rather than bare month numbers, so a period outside 2026 (e.g.
#the "2025_12" winter month added to extend this past a single contiguous 2026 window) can be
#folded in without the dict comprehension silently assuming every period is "2026_<mm>". Order
#here is cosmetic (the "months" column below sorts these lexicographically, which matches
#chronological order for zero-padded YYYY_MM strings) but kept chronological for readability.
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
MONTH_PATHS = {
    mm: os.path.join(OUTPUT_DIR, f"results_mixing_{mm}.pkl") for mm in PERIODS
}
Z95 = 1.959963984540054

#Plain-language labels for what n_months actually supports, per external review: a venue
#tested in only 2 adjacent months is "repeated," not "persistent" in any seasonal sense, and
#nothing here can reach "annual" until fall/winter months exist. "current_period" is the
#honest ceiling for 4/4 eligible months drawn from one contiguous April-July window.
PERSISTENCE_TIERS = {
    1: "single_month_not_tested",
    2: "repeated_evidence",
    3: "provisional_persistence",
    4: "current_period_persistence",  #NOT "annual_persistence" -- would need all 4 seasons
}


def persistence_tier(n_months):
    return PERSISTENCE_TIERS.get(n_months, "current_period_persistence" if n_months > 4 else "single_month_not_tested")


#---- ADOPTED METHOD (decision made after external review): DerSimonian-Laird's closed-form
#tau^2 is known to be optimistic with only 2-4 observations per group, and a normal CI doesn't
#reflect how little data tau^2 itself was estimated from. REML + Hartung-Knapp is the
#better-behaved alternative at this sample size, and is now what's reported -- mu_re/var_re/
#ci_re/tau2/I2/q_bh above are retained only as a DerSimonian-Laird diagnostic/comparison. ----
def reml_tau2(r, v):
    """REML estimate of between-month variance tau^2, by directly maximizing the restricted
    log-likelihood over tau^2 (Viechtbauer 2005's REML criterion for a random-effects
    meta-analysis) -- a 1-D, well-behaved numerical optimization, not the DL closed-form moment
    estimator. m=len(r) must be >=2."""
    def neg_restricted_loglik(tau2):
        w = 1.0 / (v + tau2)
        sw = w.sum()
        mu = np.sum(w * r) / sw
        resid2 = (r - mu) ** 2
        ll = -0.5 * np.sum(np.log(v + tau2)) - 0.5 * np.log(sw) - 0.5 * np.sum(w * resid2)
        return -ll
    #upper bound generous relative to the data's own scale (income_mixing_score residuals are
    #bounded in roughly [-1,1], so tau2 in practice never approaches this) -- wide enough not to
    #truncate a real solution, bounded Brent search so it can't run away.
    upper = max(0.25, 20.0 * np.var(r))
    res = minimize_scalar(neg_restricted_loglik, bounds=(0.0, upper), method="bounded")
    return max(0.0, res.x)


def hartung_knapp(r, v, tau2_reml):
    """Hartung-Knapp-Sidik-Jonkman pooled mean/variance given REML tau^2: widens the random-effects
    variance by a factor reflecting how much the observed per-month residuals actually agree with
    each other (not just their nominal visitor-sampling variances), and floors it at the classical
    random-effects variance under the same weights (the IQWiG/Wiksten et al. 2016 modification --
    without it, HK can occasionally give an implausibly NARROW interval when a venue's few months
    happen to agree closely by chance). Use a t-distribution with m-1 df against this variance,
    not a normal one -- see ci_hk_lo/hi below."""
    m = len(r)
    w = 1.0 / (v + tau2_reml)
    sw = w.sum()
    mu_hk = np.sum(w * r) / sw
    q_stat = np.sum(w * (r - mu_hk) ** 2) / (m - 1)
    var_classical = 1.0 / sw
    var_hk = max(q_stat / sw, var_classical)
    return mu_hk, var_hk

frames = []
for mm, path in MONTH_PATHS.items():
    d = pd.read_pickle(path)
    rf = d["results_full"].copy()
    rf["month"] = mm
    frames.append(rf)
    print(f"{mm}: {len(rf)} eligible venues, results_full columns: {list(rf.columns)}")

long = pd.concat(frames, ignore_index=True)
long = long[long["resid_vs_var"] > 0].copy()  #guard against degenerate zero-variance rows
long["w"] = 1.0 / long["resid_vs_var"]

print(f"\ntotal venue-months: {len(long)}, distinct venues: {long['placekey'].nunique()}")

static_cols = ["location_name", "borough", "top_category"]
static = long.groupby("placekey")[static_cols].first()

records = []
for pk, g in long.groupby("placekey"):
    r = g["improbable_copresence_cv"].to_numpy()  #observed residual -- see point-estimate note above
    v = g["resid_vs_var"].to_numpy()  #visitor-sampling bootstrap variance only -- see CI caveat above
    w = g["w"].to_numpy()
    m = len(g)

    mu_fe = np.sum(w * r) / np.sum(w)
    var_fe = 1.0 / np.sum(w)

    if m >= 2:
        Q = np.sum(w * (r - mu_fe) ** 2)
        df = m - 1
        C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
        I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    else:
        Q, tau2, I2 = np.nan, np.nan, np.nan

    if m >= 2 and not np.isnan(tau2):
        w_re = 1.0 / (v + tau2)
        mu_re = np.sum(w_re * r) / np.sum(w_re)
        var_re = 1.0 / np.sum(w_re)
    else:
        mu_re, var_re = mu_fe, var_fe  #single month: RE collapses to that month's own estimate

    ci_fe = (mu_fe - Z95 * np.sqrt(var_fe), mu_fe + Z95 * np.sqrt(var_fe))
    ci_re = (mu_re - Z95 * np.sqrt(var_re), mu_re + Z95 * np.sqrt(var_re))
    z_re = mu_re / np.sqrt(var_re)
    prob_positive_re = 1.0 - 0.5 * (1 + math.erf(-z_re / np.sqrt(2)))
    #two-sided p-value for "this venue's pooled effect is zero" -- the input to the
    #multiple-testing correction below. Only meaningful when n_months>=2 (a single month
    #has no independent replication to test persistence against).
    p_re = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z_re) / np.sqrt(2))))
    same_sign = int(np.sum(np.sign(r) == np.sign(mu_fe))) if mu_fe != 0 else 0

    #---- REML + Hartung-Knapp + prediction interval: the ADOPTED pooling result, see note
    #above the function definitions. q_bh_hk/fdr_sig_05_hk (below) are what's reported;
    #q_bh/fdr_sig_05 (DL-based, above/tau2/mu_re/var_re/ci_re) are diagnostic/comparison only. ----
    tau2_reml = mu_hk = var_hk = ci_hk_lo = ci_hk_hi = p_hk = pred_lo = pred_hi = np.nan
    if m >= 2:
        tau2_reml = reml_tau2(r, v)
        mu_hk, var_hk = hartung_knapp(r, v, tau2_reml)
        t_crit_ci = t_dist.ppf(0.975, df=m - 1)
        ci_hk_lo, ci_hk_hi = mu_hk - t_crit_ci * np.sqrt(var_hk), mu_hk + t_crit_ci * np.sqrt(var_hk)
        t_stat_hk = mu_hk / np.sqrt(var_hk)
        p_hk = 2.0 * t_dist.sf(abs(t_stat_hk), df=m - 1)
        if m >= 3:  #prediction interval needs m-2 >= 1 df; undefined (NaN) at m=2
            t_crit_pi = t_dist.ppf(0.975, df=m - 2)
            half_width = t_crit_pi * np.sqrt(tau2_reml + var_hk)
            pred_lo, pred_hi = mu_hk - half_width, mu_hk + half_width

    records.append({
        "placekey": pk, "n_months": m, "persistence_tier": persistence_tier(m),
        "months": ",".join(sorted(g["month"])),
        "mu_fe": mu_fe, "var_fe": var_fe, "ci_fe_lo": ci_fe[0], "ci_fe_hi": ci_fe[1],
        "Q": Q, "tau2": tau2, "I2": I2,
        "mu_re": mu_re, "var_re": var_re, "ci_re_lo": ci_re[0], "ci_re_hi": ci_re[1],
        "p_re": p_re, "prob_positive_re": prob_positive_re, "same_sign_months": same_sign,
        #ADOPTED pooling-result columns (REML tau2 + Hartung-Knapp + prediction interval):
        "tau2_reml": tau2_reml, "mu_hk": mu_hk, "var_hk": var_hk,
        "ci_hk_lo": ci_hk_lo, "ci_hk_hi": ci_hk_hi, "p_hk": p_hk,
        "pred_lo": pred_lo, "pred_hi": pred_hi,
    })

pooled = pd.DataFrame(records).set_index("placekey").join(static)
pooled["ci_re_excludes_zero"] = (pooled["ci_re_lo"] > 0) | (pooled["ci_re_hi"] < 0)
pooled["ci_fe_excludes_zero"] = (pooled["ci_fe_lo"] > 0) | (pooled["ci_fe_hi"] < 0)
pooled = pooled.reset_index().sort_values("mu_re", ascending=False)


def benjamini_hochberg(pvals, alpha=0.05):
    """Standard BH step-up procedure. Returns (q_values, reject_at_alpha).
    q_values[i] is the BH-adjusted p-value -- the smallest FDR level at which
    that venue would be called significant -- so downstream code can just
    threshold q_bh at whatever alpha it wants without rerunning this."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    i = np.arange(1, m + 1)
    raw_q = ranked * m / i
    q_sorted = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0, 1)
    q = np.empty(m)
    q[order] = q_sorted
    return q, (q <= alpha)


#FDR correction applies to the "tested" family -- venues with >=2 eligible months, where a
#real cross-month comparison happened. A single-month venue has no independent replication
#to test persistence against (its mu_re/var_re is just that one month's own sampling
#uncertainty), so it's excluded from the family rather than diluting/strengthening it --
#otherwise this would silently re-introduce something close to the old point-in-time
#percentile screen for exactly the venues where persistence can't be assessed at all.
family_mask = (pooled["n_months"] >= 2).to_numpy()
pooled["q_bh"] = np.nan
q_vals, _ = benjamini_hochberg(pooled.loc[family_mask, "p_re"].to_numpy(), alpha=0.05)
pooled.loc[family_mask, "q_bh"] = q_vals
pooled["fdr_sig_05"] = pooled["q_bh"] <= 0.05  #NaN (n_months==1) compares False, as intended

#ADOPTED FDR call: same BH correction applied to the Hartung-Knapp p-values. q_bh/fdr_sig_05
#above (DL-based) are kept only as a diagnostic/comparison -- q_bh_hk/fdr_sig_05_hk are the
#numbers of record.
pooled["q_bh_hk"] = np.nan
q_vals_hk, _ = benjamini_hochberg(pooled.loc[family_mask, "p_hk"].to_numpy(), alpha=0.05)
pooled.loc[family_mask, "q_bh_hk"] = q_vals_hk
pooled["fdr_sig_05_hk"] = pooled["q_bh_hk"] <= 0.05

print(f"\n{len(pooled)} venues with >=1 eligible month pooled; "
      f"{family_mask.sum()} have >=2 eligible months (the tested family for FDR control).")
print(f"Of the tested family: {pooled['ci_re_excludes_zero'][family_mask].sum()} have an "
      f"UNADJUSTED 95% interval excluding zero (DL-based, diagnostic only); "
      f"{pooled['fdr_sig_05_hk'].sum()} survive the ADOPTED REML+Hartung-Knapp "
      f"Benjamini-Hochberg FDR control at q<=0.05 across all {family_mask.sum()} tests.")
for min_m in (2, 3, 4):
    sub = pooled[(pooled["n_months"] >= min_m)]
    print(f"  n_months>={min_m} ({len(sub)} venues): {sub['fdr_sig_05_hk'].sum()} survive q<=0.05")

print(f"\n[diagnostic/comparison only] DerSimonian-Laird survivors: {pooled['fdr_sig_05'].sum()} "
      f"vs. the ADOPTED REML+Hartung-Knapp survivors: {pooled['fdr_sig_05_hk'].sum()} (same "
      f"{family_mask.sum()}-venue tested family, q<=0.05 both ways). HK survives fewer, especially "
      f"among n_months==2 venues, where its t-distribution has only 1 df and so a much fatter tail "
      f"than DL's normal approximation -- this gap is the documented reason DL is retained only as "
      f"a comparison, not used for any reported finding.")
for min_m in (2, 3, 4):
    sub = pooled[(pooled["n_months"] >= min_m)]
    print(f"  [DL, diagnostic only] n_months>={min_m} ({len(sub)} venues): {sub['fdr_sig_05'].sum()} survive q<=0.05")

pooled.to_pickle(os.path.join(OUTPUT_DIR, "pooled_persistence_2026.pkl"))
pooled.to_csv(os.path.join(OUTPUT_DIR, "pooled_persistence_2026.csv"), index=False)
print(f"\nsaved {OUTPUT_DIR}/pooled_persistence_2026.{{pkl,csv}}")
print("\npersistence_tier counts (tested family, n_months>=2):")
print(pooled.loc[family_mask, "persistence_tier"].value_counts().to_string())

print("\n--- top 15 by mu_hk, RESTRICTED TO FDR-SURVIVING VENUES (n_months>=2, q_bh_hk<=0.05, ADOPTED) ---")
cols = ["placekey", "location_name", "borough", "n_months", "months", "mu_fe", "mu_hk",
        "tau2_reml", "p_hk", "q_bh_hk", "ci_hk_lo", "ci_hk_hi", "pred_lo", "pred_hi"]
surv = pooled[(pooled["n_months"] >= 2) & pooled["fdr_sig_05_hk"]]
print(surv.sort_values("mu_hk", ascending=False).head(15)[cols].to_string(index=False))

print("\n--- bottom 15 by mu_hk, RESTRICTED TO FDR-SURVIVING VENUES (n_months>=2, q_bh_hk<=0.05, ADOPTED) ---")
print(surv.sort_values("mu_hk").head(15)[cols].to_string(index=False))

print("\n--- named headline venues (HK columns = numbers of record, ADOPTED; DL columns = comparison only) ---")
hk_cols = ["placekey", "location_name", "borough", "n_months", "persistence_tier", "months",
           "mu_hk", "tau2_reml", "p_hk", "q_bh_hk", "ci_hk_lo", "ci_hk_hi", "pred_lo", "pred_hi",
           "mu_re", "tau2", "I2", "q_bh", "ci_re_lo", "ci_re_hi"]
for name_substr in ["Amenazel", "Chens Fresh", "Jacob Riis", "Hunts Point"]:
    hit = pooled[pooled["location_name"].str.contains(name_substr, case=False, na=False)]
    if len(hit):
        print(f"\n{name_substr}:")
        print(hit[hk_cols].to_string(index=False))
    else:
        print(f"\n{name_substr}: no match in pooled results")
