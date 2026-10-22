"""
pool_persistence.py

Implements the persistence/pooling design from the "point 3" discussion: for every
venue, pools its per-month residual against a random-effects meta-analysis
estimate that separates genuine month-to-month/seasonal variation (tau^2, the
u_vm term) from per-month uncertainty in r_vm itself (the epsilon_vm term) --
r_vm = mu_v + u_vm + eps_vm. ADOPTED METHOD: REML (for tau^2) +
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
right tool for ESTIMATING VARIANCE (var(r_vm), used below as part of the
inverse-variance weight) -- they just should never have stood in for the point
estimate itself. If a deliberate bias correction is wanted later (e.g.
2*r_observed - resid_vs_mean), it should be validated by simulation first and labeled
as a correction, not silently substituted.

POOLING VARIANCE, fixed after a second round of external review (see README "Known
limitations" -- Model prediction variance / the independence assumption): var(r_vm)
used to be resid_vs_var ALONE -- visitor-sampling uncertainty only, i.e. "if this
venue had drawn a different, equally-likely sample of its own visitors, how much
would its residual move," holding the fitted RandomForest's prediction
(expected_mixing_cv) completely fixed. That excluded uncertainty in the prediction
itself (how much it would move under a different, equally legitimate spatially-
blocked training sample), which turned out NOT to be a rounding-error-sized gap --
quantified directly (see README), it was bigger than the sampling variance already
included for roughly a third of a reference-flagged test sample of venues. ADOPTED
FIX: var_combined = resid_vs_var + model_pred_var, where model_pred_var comes from
estimate_model_prediction_variance.py's spatial-block bootstrap (same design as
sensitivity_spatial_block_bootstrap.py, which fixed the OTHER model-refit bootstrap
defect this same review caught). var_combined is what feeds tau2_reml/mu_hk/var_hk/
ci_hk/p_hk/q_bh_hk below -- the ADOPTED pooling result. The DerSimonian-Laird
diagnostic columns (mu_re/var_re/ci_re/tau2/I2/q_bh) are UNCHANGED and still use
resid_vs_var alone, since they were always diagnostic/comparison only, not the
reported result -- see README for why mixing two changes into one column would make
that comparison harder to read, not easier.

MISSING MODEL-REFIT VARIANCE, FIXED (Note 11, external review): var_combined above used to
silently treat a venue-month's missing model_pred_var (its grid cell was never drawn across
estimate_model_prediction_variance.py's bootstrap replicates -- see that script's docstring)
as if model-refit uncertainty were exactly ZERO for that venue-month, via a bare
.fillna(0.0). That's backwards: a venue-month with no model_pred_var estimate is one we know
LESS about, not one with zero refit uncertainty, and silently zero-filling it understated
var_combined, overweighted that month in the GLS pooling below, and could tip a borderline
venue's q_bh_hk under 0.05 on artificially narrowed uncertainty rather than real evidence.
ADOPTED FIX: var_combined is no longer filled; a venue-month missing model_pred_var is instead
excluded from the ADOPTED REML+Hartung-Knapp pooling specifically (n_months_adopted, computed
below, counts only venue-months with a real model_pred_var estimate). n_months/
persistence_tier/the DerSimonian-Laird diagnostic columns, which never depended on
model_pred_var, are UNCHANGED -- this is a different kind of missingness than failing the
income_known_devices eligibility floor (a venue simply not present in a month's data), not a
second instance of the same thing, so it gets its own, separately-reported exclusion rather
than being folded into n_months. The FDR-tested family (q_bh_hk/fdr_sig_05_hk) now requires
n_months_adopted>=2, not just n_months>=2, for the same reason a single-eligible-month venue
was already excluded. Effect on the published survivor count and the four named venues has NOT
yet been measured -- this fix has not been run against real data yet; see README "Known
limitations" until the pipeline is rerun with it in place.

POOLING COVARIANCE, FIXED (Note 10, a seventh round of external review -- previously an open
caveat here): the per-venue months pooled here used to be treated as independent replicates of
r_vm. The number this docstring previously cited for that ("+0.20 excess at lag 1, fading by lag
2+") turned out, on tracing it, to describe a DIFFERENT check -- the quintile-share
serial-correlation test behind Note 5's Dirichlet-multinomial correction, not a direct test of
r_vm itself; no committed script ever tested r_vm directly before now. estimate_temporal_
dependence.py does, across all 7 available lags, on the same 7,838-venue balanced panel: real
excess correlation concentrated at lag 1 (+0.156 raw / +0.180 variance-standardized over the
demeaning-artifact benchmark), fading to at-or-below that benchmark by lag 2 and beyond -- a
one-lag-only pattern, confirming the qualitative shape of the old (uncertainly-sourced) claim
while correcting its exact number and, more importantly, giving it a real, reproducible basis.

ADOPTED FIX: `reml_tau2`/`hartung_knapp` below now build an explicit GLS covariance
Sigma_v = diag(tau2 + var_combined) + rho * sqrt(var_combined_i * var_combined_j) on entries for
a pair of a venue's eligible months that are CALENDAR-adjacent (one month apart on the real Dec
2025-Jul 2026 timeline, not just "next in this venue's own eligible-months list" -- a venue
eligible in Feb and April has a 2-month gap, not lag 1), zero elsewhere. tau2 keeps its usual
meaning (occasion-level heterogeneity, independent across months, same convention the old
diagonal method already used) -- only the sampling/model-error component (var_combined) carries
the measured lag-1 correlation, matching the fact that it fades by lag 2+ rather than acting
like a persistent between-month signal. rho is FIXED at RHO_TEMPORAL = +0.1683 (mean of the
raw/standardized lag-1 excess estimates, read from estimate_temporal_dependence.py's
temporal_dependence_estimate.pkl -- same single-point-from-an-honestly-reported-range convention
as S_hat's adoption in Note 5), not re-estimated per venue -- there isn't remotely enough
per-venue data to fit rho and tau2 both. At rho=0 this is EXACTLY the old diagonal method
(checked to 1e-15, not assumed -- see note10/prototype_ar1_pooling.py, the prototype this was
validated with before being adopted here).

Effect, prototyped on a 2,100-venue stratified sample before being adopted: median variance
inflation ~1.24x (CIs widen ~11%), growing mildly with n_months (1.16x at m=2 to 1.28x at m=8 --
more months means more adjacent pairs). Point estimates (mu_hk) barely move (median shift
~0.0005-0.001). See README for the full-pipeline effect on the FDR survivor count and the four
named venues, measured after adopting this fix, not merely prototyped.

POOLING VARIANCE, fixed again after a THIRD round of external review (Note 5 -- see README
"Known limitations" -- the overdispersion check's independence assumption): resid_vs_var itself
used to come from fit_mixing_model_2026_0X.py's plain-multinomial visitor-sampling bootstrap,
read as a reasonable estimate because the original 3-month overdispersion check (May/June/July
2026 only) found VIF_obs=0.91 -- no detectable extra-multinomial clustering. That check's own
independence assumption (that a venue's different months are uncorrelated once you've accounted
for its fitted mean) turned out to be false once all 8 periods were available to test it on
directly: the corrected VIF on the full 8-period balanced panel is ~1.48 (real evidence FOR
clustering, reversing the old read -- see estimate_overdispersion.py's revised docstring for the
full derivation and the honest sensitivity range). ADOPTED FIX: resid_vs_var/resid_vs_mean now
come from estimate_visitor_sampling_variance_dm.py's Dirichlet-multinomial-corrected bootstrap
(same fixed expected_mixing_cv design as before, just with overdispersion-corrected draws) instead
of fit_mixing_model_2026_0X.py's plain-multinomial one. This correction is strongly size-dependent
-- it barely moves small, floor-eligible venues but can multiply variance several-fold for large,
high-traffic ones -- see that script's docstring. var_combined's formula (resid_vs_var +
model_pred_var) is unchanged; only what resid_vs_var itself measures has been corrected.

Per venue, reports:
  - n_months: how many months it cleared the eligibility floor (results_full
    only has rows for eligible venue-months -- a venue absent some month is
    simply not in that month's results_full, so it's excluded from the pool
    for that month rather than treated as "not mixed").
  - persistence_tier: a plain-language label for what n_months actually supports --
    see PERSISTENCE_TIERS below. None of the current data can earn "annual_persistence"
    (that needs fall/winter months, which don't exist yet); the ceiling right now is
    "current_period_persistence" for a venue eligible in all 8 of Dec 2025-Jul 2026.
  - mu_fe, var_fe, ci_fe_lo/hi: fixed-effect (no between-month variance term),
    computed from resid_vs_var alone (DIAGNOSTIC, not the ADOPTED weight).
  - tau2, Q, I2: DerSimonian-Laird between-month variance estimate, Cochran's Q,
    and I^2 (% of total variation that's genuine month-to-month movement rather
    than per-month noise). With only 2-8 months per venue, tau2=0 means "no
    heterogeneity DETECTED" -- usually a power problem, not evidence of none; see
    README for why DerSimonian-Laird + a normal CI is known to be optimistic at
    this few observations. DIAGNOSTIC/COMPARISON ONLY -- kept for reference, not
    the reported result; see tau2_reml/mu_hk/var_hk/ci_hk_lo/hi below for the
    ADOPTED estimate (REML + Hartung-Knapp over var_combined, which fixes both
    the DL-optimism problem AND the omitted-model-error problem).
  - mu_re, var_re, ci_re_lo/hi: DerSimonian-Laird random-effects pooled estimate
    and CONDITIONAL CI (widens when tau2>0), over resid_vs_var alone.
    DIAGNOSTIC/COMPARISON ONLY, same reason as above -- mu_hk/var_hk/ci_hk_lo/hi
    below are the adopted estimate.
  - model_pred_var: the model-refit/prediction variance folded into var_combined
    for this venue (mean across its eligible months, for transparency/inspection --
    the pooling itself uses each month's own value, not this summary).
  - n_months_adopted: how many of those n_months eligible months actually had a usable
    model_pred_var estimate and so were included in the ADOPTED REML+Hartung-Knapp pooling
    below (Note 11) -- can be LESS than n_months. tau2_reml/mu_hk/etc. are NaN when this is
    <2, same as n_months<2 did before this fix.
  - tau2_reml, mu_hk, var_hk, ci_hk_lo/hi, p_hk, q_bh_hk, fdr_sig_05_hk, pred_lo/hi:
    the ADOPTED pooling result -- REML estimate of tau^2 (Viechtbauer 2005) over
    var_combined, plus Hartung-Knapp-Sidik-Jonkman's pooled mean/CI (t-distributed
    with m-1 df, widened per Wiksten et al. 2016 so it can't give an implausibly
    narrow interval), and the Benjamini-Hochberg FDR call (q_bh_hk/fdr_sig_05_hk)
    actually used downstream. pred_lo/hi is additionally a prediction interval
    for a hypothetical future month (defined at n_months>=3).
  - prob_positive_re: P(true effect > 0) under a normal approximation to the
    DerSimonian-Laird random-effects estimate -- a direct answer to "is this venue
    persistently more/less mixed than expected," not just a flag/no-flag call.
  - same_sign_months: how many of its eligible months have the same residual
    sign as mu_fe (a cheap, assumption-free persistence check alongside the
    formal CI).

Caveat, stated plainly (carried over from the fit_mixing_model patch and the
prior discussion): with only 2-8 months, tau^2 is estimated from limited
information and will be noisy for any single venue -- reported honestly
alongside mu_hk rather than treated as precise. This is NOT yet a 12-month
or seasonal-contrast analysis (no fall/winter-beyond-December month exists yet),
and it does NOT yet distinguish "absent from the POI universe" from "present but
ineligible" for venue-months outside results_full (that bookkeeping -- point 5
from the "point 3" discussion -- would need the raw per-month analysis_pois
universe, not just model_ready/results_full, and hasn't been built).

Input:  OUTPUT_DIR/results_mixing_<PERIOD>.pkl, one per period (must be from the
        patched fit_mixing_model_2026_<MM>.py, i.e. must contain "results_full").
        OUTPUT_DIR/model_prediction_variance_<PERIOD>.csv, one per period (from
        estimate_model_prediction_variance.py -- now a REQUIRED step, not optional;
        run it for every period in PERIODS below before running this script).
        OUTPUT_DIR/visitor_sampling_variance_dm_<PERIOD>.csv, one per period (from
        estimate_visitor_sampling_variance_dm.py -- also REQUIRED since the Note 5 fix;
        supplies the overdispersion-corrected resid_vs_mean/resid_vs_var used below in
        place of results_mixing_<PERIOD>.pkl's plain-multinomial columns).
        OUTPUT_DIR/temporal_dependence_estimate.pkl (from estimate_temporal_dependence.py --
        also REQUIRED since the Note 10 fix; supplies RHO_TEMPORAL, the fixed lag-1
        excess-correlation point estimate the GLS pooling below uses).
        Set OUTPUT_DIR to wherever build_features_2026_0X.py /
        fit_mixing_model_2026_0X.py wrote their outputs (same OUTPUT_DIR each
        period -- see those scripts' docstrings); default ./output.
Output: OUTPUT_DIR/pooled_persistence_2026.pkl/.csv -- one row per venue with >=1
        eligible month, sorted by mu_hk.
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
#Model-refit/prediction variance, one file per period -- see estimate_model_prediction_variance.py.
MODEL_VAR_PATHS = {
    mm: os.path.join(OUTPUT_DIR, f"model_prediction_variance_{mm}.csv") for mm in PERIODS
}
#Overdispersion-corrected visitor-sampling variance, one file per period -- see
#estimate_visitor_sampling_variance_dm.py (Note 5 fix). Replaces results_mixing_<PERIOD>.pkl's
#plain-multinomial resid_vs_mean/resid_vs_var below.
VS_VAR_PATHS = {
    mm: os.path.join(OUTPUT_DIR, f"visitor_sampling_variance_dm_{mm}.csv") for mm in PERIODS
}
#Calendar position of each period (0..7), NOT a venue's own position in its eligible-months list
#-- two of a venue's eligible months are lag-1/"calendar-adjacent" only if their MONTH_IDX differ
#by exactly 1, so a venue eligible in Feb and April (idx 2 and 4) is correctly NOT lag-1, even
#though April is its next observed month. See Note 10 in the module docstring above.
MONTH_IDX = {mm: i for i, mm in enumerate(PERIODS)}
#Adopted lag-1 excess-correlation point estimate (Note 10) -- read from
#estimate_temporal_dependence.py's saved pickle, same env-override pattern as S_HAT below.
RHO_TEMPORAL = float(os.environ.get("RHO_TEMPORAL", pd.read_pickle(
    os.path.join(OUTPUT_DIR, "temporal_dependence_estimate.pkl"))["rho_lag1_point"]))
Z95 = 1.959963984540054

#Plain-language labels for what n_months actually supports, per external review: a venue
#tested in only 2 adjacent months is "repeated," not "persistent" in any seasonal sense, and
#nothing here can reach "annual" until fall/winter months exist. "current_period" is the
#honest ceiling for 8/8 eligible months drawn from one contiguous Dec 2025-Jul 2026 window.
PERSISTENCE_TIERS = {
    1: "single_month_not_tested",
    2: "repeated_evidence",
    3: "provisional_persistence",
    4: "current_period_persistence",  #NOT "annual_persistence" -- would need all 4 seasons
}


def persistence_tier(n_months):
    return PERSISTENCE_TIERS.get(n_months, "current_period_persistence" if n_months > 4 else "single_month_not_tested")


#---- ADOPTED METHOD (decision made after external review): DerSimonian-Laird's closed-form
#tau^2 is known to be optimistic with only 2-8 observations per group, and a normal CI doesn't
#reflect how little data tau^2 itself was estimated from. REML + Hartung-Knapp is the
#better-behaved alternative at this sample size, run over var_combined (see module docstring
#for why var_combined, not resid_vs_var alone). mu_re/var_re/ci_re/tau2/I2/q_bh (DL, over
#resid_vs_var alone) are retained only as a diagnostic/comparison. ----
def build_sigma(v, month_idx, tau2, rho):
    """GLS covariance for one venue's m eligible months (Note 10 -- see module docstring).
    Sigma_ii = tau2 + v_i, the SAME marginal variance the old diagonal method used (tau2 is
    still an occasion-level term, independent across months, not a single shared venue-level
    draw). Sigma_ij (i!=j) = rho*sqrt(v_i*v_j) when months i,j are CALENDAR-adjacent (their
    MONTH_IDX differ by exactly 1), else 0. At rho=0 this is exactly diag(tau2+v) -- the old
    method's implicit covariance -- which is what makes reml_tau2/hartung_knapp below exact
    generalizations rather than a different model; verified to 1e-15 in
    note10/prototype_ar1_pooling.py before being adopted."""
    m = len(v)
    gap = np.abs(month_idx[:, None] - month_idx[None, :])
    adj = (gap == 1).astype(float)
    np.fill_diagonal(adj, 0.0)
    sqrt_v = np.sqrt(v)
    off_diag = rho * sqrt_v[:, None] * sqrt_v[None, :] * adj
    return np.diag(tau2 + v) + off_diag


def reml_tau2(r, v, month_idx, rho=RHO_TEMPORAL):
    """REML estimate of between-month variance tau^2, by directly maximizing the restricted
    log-likelihood over tau^2 (Viechtbauer 2005's REML criterion for a random-effects
    meta-analysis, generalized (Note 10) to the GLS covariance Sigma_v(tau2, rho) from
    build_sigma above instead of a diagonal one) -- a 1-D, well-behaved numerical optimization,
    not the DL closed-form moment estimator. m=len(r) must be >=2."""
    m = len(r)
    def neg_restricted_loglik(tau2):
        Sigma = build_sigma(v, month_idx, tau2, rho)
        try:
            Sinv = np.linalg.inv(Sigma)
            sign, logdet = np.linalg.slogdet(Sigma)
        except np.linalg.LinAlgError:
            return 1e10
        ones = np.ones(m)
        sw = ones @ Sinv @ ones
        if sw <= 0 or sign <= 0:
            return 1e10
        mu = (ones @ Sinv @ r) / sw
        resid = r - mu
        ll = -0.5 * logdet - 0.5 * np.log(sw) - 0.5 * (resid @ Sinv @ resid)
        return -ll
    #upper bound generous relative to the data's own scale (income_mixing_score residuals are
    #bounded in roughly [-1,1], so tau2 in practice never approaches this) -- wide enough not to
    #truncate a real solution, bounded Brent search so it can't run away.
    upper = max(0.25, 20.0 * np.var(r))
    res = minimize_scalar(neg_restricted_loglik, bounds=(0.0, upper), method="bounded")
    return max(0.0, res.x)


def hartung_knapp(r, v, month_idx, tau2_reml, rho=RHO_TEMPORAL):
    """Hartung-Knapp-Sidik-Jonkman pooled mean/variance given REML tau^2: widens the random-effects
    variance by a factor reflecting how much the observed per-month residuals actually agree with
    each other (not just their nominal variances), and floors it at the classical random-effects
    variance under the same weights (the IQWiG/Wiksten et al. 2016 modification -- without it, HK
    can occasionally give an implausibly NARROW interval when a venue's few months happen to agree
    closely by chance). Generalized (Note 10) to the GLS covariance Sigma_v -- the diagonal
    weights w_i/sum(w) below become Sigma^-1-weighted sums, which is the standard GLS form and
    reduces exactly to the old diagonal formulas at rho=0. Use a t-distribution with m-1 df
    against this variance, not a normal one -- see ci_hk_lo/hi below."""
    m = len(r)
    Sigma = build_sigma(v, month_idx, tau2_reml, rho)
    Sinv = np.linalg.inv(Sigma)
    ones = np.ones(m)
    sw = ones @ Sinv @ ones
    mu_hk = (ones @ Sinv @ r) / sw
    resid = r - mu_hk
    q_stat = (resid @ Sinv @ resid) / (m - 1)
    var_classical = 1.0 / sw
    var_hk = max(q_stat / sw, var_classical)
    return mu_hk, var_hk

frames = []
for mm, path in MONTH_PATHS.items():
    d = pd.read_pickle(path)
    rf = d["results_full"].copy()
    rf["month"] = mm
    #Note 5 fix: drop the plain-multinomial resid_vs_mean/resid_vs_var results_full shipped with,
    #and replace with the Dirichlet-multinomial-corrected versions -- see module docstring and
    #estimate_visitor_sampling_variance_dm.py.
    rf = rf.drop(columns=["resid_vs_mean", "resid_vs_var"])
    vs = pd.read_csv(VS_VAR_PATHS[mm])[["placekey", "resid_vs_mean_dm", "resid_vs_var_dm"]]
    vs = vs.rename(columns={"resid_vs_mean_dm": "resid_vs_mean", "resid_vs_var_dm": "resid_vs_var"})
    rf = rf.merge(vs, on="placekey", how="left")
    n_missing_vs = rf["resid_vs_var"].isna().sum()
    if n_missing_vs:
        print(f"{mm}: WARNING {n_missing_vs} venues missing DM-corrected resid_vs_var "
              f"(unexpected -- every eligible venue is resampled every DM bootstrap draw, "
              f"unlike the grid-cell-block model_pred_var bootstrap)")
    mv = pd.read_csv(MODEL_VAR_PATHS[mm])[["placekey", "model_pred_var"]]
    rf = rf.merge(mv, on="placekey", how="left")
    n_missing = rf["model_pred_var"].isna().sum()
    if n_missing:
        print(f"{mm}: {n_missing} venues missing model_pred_var (cell never drawn across "
              f"estimate_model_prediction_variance.py's replicates) -- falling back to "
              f"resid_vs_var alone for those")
    frames.append(rf)
    print(f"{mm}: {len(rf)} eligible venues, results_full columns: {list(rf.columns)}")

long = pd.concat(frames, ignore_index=True)
long = long[long["resid_vs_var"] > 0].copy()  #guard against degenerate zero-variance rows
long["month_idx"] = long["month"].map(MONTH_IDX)  #calendar position -- see Note 10 above
#ADOPTED pooling variance: visitor-sampling (resid_vs_var) + model-refit/prediction
#(model_pred_var, fallback 0 for the rare venue-month where it couldn't be estimated -- see
#warning above). DL diagnostic columns below still use resid_vs_var alone -- see module
#docstring for why those two are kept separate rather than both switched over.
#Note 11: no .fillna(0.0) here anymore -- a venue-month missing model_pred_var now stays NaN
#in var_combined/w_combined instead of being silently treated as zero model-refit uncertainty.
#The per-venue loop below filters to non-NaN var_combined rows before running the ADOPTED
#REML+Hartung-Knapp pooling; see module docstring (Note 11).
long["var_combined"] = long["resid_vs_var"] + long["model_pred_var"]
long["w"] = 1.0 / long["resid_vs_var"]
long["w_combined"] = 1.0 / long["var_combined"]
n_missing_combined = int(long["var_combined"].isna().sum())
print(f"Note 11: {n_missing_combined} of {len(long)} venue-months "
      f"({n_missing_combined / len(long) * 100:.2f}%) have no model_pred_var estimate and are "
      f"excluded from the ADOPTED pooling below (n_months_adopted < n_months for those venues)")

print(f"\ntotal venue-months: {len(long)}, distinct venues: {long['placekey'].nunique()}")
print(f"median resid_vs_var (visitor-sampling only): {long['resid_vs_var'].median():.6f}")
print(f"median model_pred_var (model-refit, now folded in): {long['model_pred_var'].median():.6f}")
print(f"median var_combined: {long['var_combined'].median():.6f} "
      f"(median inflation factor vs. resid_vs_var alone: "
      f"{(long['var_combined'] / long['resid_vs_var']).median():.3f})")

static_cols = ["location_name", "borough", "top_category"]
static = long.groupby("placekey")[static_cols].first()

records = []
for pk, g in long.groupby("placekey"):
    r = g["improbable_copresence_cv"].to_numpy()  #observed residual -- see point-estimate note above
    v = g["resid_vs_var"].to_numpy()  #DIAGNOSTIC weight (DL columns below only)
    v_combined = g["var_combined"].to_numpy()  #ADOPTED weight (REML+HK below)
    midx = g["month_idx"].to_numpy()  #calendar position, for GLS lag-1 adjacency (Note 10)
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
    #two-sided p-value for "this venue's pooled effect is zero" -- DIAGNOSTIC only (DL-based);
    #q_bh/fdr_sig_05 below built from this are comparison columns, not the reported result.
    p_re = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z_re) / np.sqrt(2))))
    same_sign = int(np.sum(np.sign(r) == np.sign(mu_fe))) if mu_fe != 0 else 0

    #---- REML + Hartung-Knapp + prediction interval over var_combined: the ADOPTED pooling
    #result, see module docstring. q_bh_hk/fdr_sig_05_hk (below) are what's reported;
    #q_bh/fdr_sig_05 (DL-based, above) are diagnostic/comparison only.
    #Note 11: restricted to the subset of this venue's eligible months that have a real
    #var_combined (i.e. a real model_pred_var estimate) -- a venue-month whose grid cell was
    #never drawn by estimate_model_prediction_variance.py's bootstrap is excluded from THIS
    #pooling specifically, not treated as zero model-refit uncertainty. n_months_adopted can be
    #less than n_months; it does not affect n_months/persistence_tier/the DL diagnostic columns
    #above, none of which depend on model_pred_var. ----
    adopted_mask = ~np.isnan(v_combined)
    m_adopted = int(adopted_mask.sum())
    r_a, v_a, midx_a = r[adopted_mask], v_combined[adopted_mask], midx[adopted_mask]
    tau2_reml = mu_hk = var_hk = ci_hk_lo = ci_hk_hi = p_hk = pred_lo = pred_hi = np.nan
    if m_adopted >= 2:
        tau2_reml = reml_tau2(r_a, v_a, midx_a)
        mu_hk, var_hk = hartung_knapp(r_a, v_a, midx_a, tau2_reml)
        t_crit_ci = t_dist.ppf(0.975, df=m_adopted - 1)
        ci_hk_lo, ci_hk_hi = mu_hk - t_crit_ci * np.sqrt(var_hk), mu_hk + t_crit_ci * np.sqrt(var_hk)
        t_stat_hk = mu_hk / np.sqrt(var_hk)
        p_hk = 2.0 * t_dist.sf(abs(t_stat_hk), df=m_adopted - 1)
        if m_adopted >= 3:  #prediction interval needs m_adopted-2 >= 1 df; undefined at m_adopted=2
            t_crit_pi = t_dist.ppf(0.975, df=m_adopted - 2)
            half_width = t_crit_pi * np.sqrt(tau2_reml + var_hk)
            pred_lo, pred_hi = mu_hk - half_width, mu_hk + half_width

    records.append({
        "placekey": pk, "n_months": m, "persistence_tier": persistence_tier(m),
        "n_months_adopted": m_adopted,  #Note 11 -- see module docstring
        "months": ",".join(sorted(g["month"])),
        "mu_fe": mu_fe, "var_fe": var_fe, "ci_fe_lo": ci_fe[0], "ci_fe_hi": ci_fe[1],
        "Q": Q, "tau2": tau2, "I2": I2,
        "mu_re": mu_re, "var_re": var_re, "ci_re_lo": ci_re[0], "ci_re_hi": ci_re[1],
        "p_re": p_re, "prob_positive_re": prob_positive_re, "same_sign_months": same_sign,
        "model_pred_var": np.nanmean(g["model_pred_var"].to_numpy()),
        #ADOPTED pooling-result columns (REML tau2 + Hartung-Knapp + prediction interval,
        #over var_combined):
        "tau2_reml": tau2_reml, "mu_hk": mu_hk, "var_hk": var_hk,
        "ci_hk_lo": ci_hk_lo, "ci_hk_hi": ci_hk_hi, "p_hk": p_hk,
        "pred_lo": pred_lo, "pred_hi": pred_hi,
    })

pooled = pd.DataFrame(records).set_index("placekey").join(static)
pooled["ci_re_excludes_zero"] = (pooled["ci_re_lo"] > 0) | (pooled["ci_re_hi"] < 0)
pooled["ci_fe_excludes_zero"] = (pooled["ci_fe_lo"] > 0) | (pooled["ci_fe_hi"] < 0)
pooled = pooled.reset_index().sort_values("mu_hk", ascending=False)


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
#Note 11: requires n_months_adopted>=2 (a real ADOPTED pooled estimate exists), not just
#n_months>=2 -- see module docstring for why these two counts can now differ.
family_mask = (pooled["n_months_adopted"] >= 2).to_numpy()
pooled["q_bh"] = np.nan
q_vals, _ = benjamini_hochberg(pooled.loc[family_mask, "p_re"].to_numpy(), alpha=0.05)
pooled.loc[family_mask, "q_bh"] = q_vals
pooled["fdr_sig_05"] = pooled["q_bh"] <= 0.05  #NaN (n_months==1) compares False, as intended

#ADOPTED FDR call: same BH correction applied to the Hartung-Knapp p-values (over var_combined).
#q_bh/fdr_sig_05 above (DL-based, over resid_vs_var alone) are kept only as a
#diagnostic/comparison -- q_bh_hk/fdr_sig_05_hk are the numbers of record.
pooled["q_bh_hk"] = np.nan
q_vals_hk, _ = benjamini_hochberg(pooled.loc[family_mask, "p_hk"].to_numpy(), alpha=0.05)
pooled.loc[family_mask, "q_bh_hk"] = q_vals_hk
pooled["fdr_sig_05_hk"] = pooled["q_bh_hk"] <= 0.05

print(f"\n{len(pooled)} venues with >=1 eligible month pooled; "
      f"{family_mask.sum()} have >=2 eligible months (the tested family for FDR control).")
print(f"Of the tested family: {pooled['ci_re_excludes_zero'][family_mask].sum()} have an "
      f"UNADJUSTED 95% interval excluding zero (DL-based, diagnostic only); "
      f"{pooled['fdr_sig_05_hk'].sum()} survive the ADOPTED REML+Hartung-Knapp (var_combined) "
      f"Benjamini-Hochberg FDR control at q<=0.05 across all {family_mask.sum()} tests.")
for min_m in (2, 3, 4):
    sub = pooled[(pooled["n_months"] >= min_m)]
    print(f"  n_months>={min_m} ({len(sub)} venues): {sub['fdr_sig_05_hk'].sum()} survive q<=0.05")

print(f"\n[diagnostic/comparison only] DerSimonian-Laird over resid_vs_var alone survivors: "
      f"{pooled['fdr_sig_05'].sum()} vs. the ADOPTED REML+Hartung-Knapp over var_combined "
      f"survivors: {pooled['fdr_sig_05_hk'].sum()} (same {family_mask.sum()}-venue tested "
      f"family, q<=0.05 both ways -- this comparison conflates the DL-vs-HK method switch AND "
      f"the resid_vs_var-vs-var_combined variance switch; see README for each switch's effect "
      f"measured separately).")
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

print("\n--- named headline venues (HK columns over var_combined = numbers of record, ADOPTED; "
      "DL columns over resid_vs_var alone = comparison only) ---")
hk_cols = ["placekey", "location_name", "borough", "n_months", "persistence_tier", "months",
           "mu_hk", "tau2_reml", "p_hk", "q_bh_hk", "ci_hk_lo", "ci_hk_hi", "pred_lo", "pred_hi",
           "model_pred_var", "mu_re", "tau2", "I2", "q_bh", "ci_re_lo", "ci_re_hi"]
for name_substr in ["Amenazel", "Chens Fresh", "Jacob Riis", "Hunts Point"]:
    hit = pooled[pooled["location_name"].str.contains(name_substr, case=False, na=False)]
    if len(hit):
        print(f"\n{name_substr}:")
        print(hit[hk_cols].to_string(index=False))
    else:
        print(f"\n{name_substr}: no match in pooled results")
