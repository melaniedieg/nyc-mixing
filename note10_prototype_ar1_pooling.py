"""
prototype_ar1_pooling.py (Note 10 prototype -- NOT the adopted method, not wired into the
pipeline. Scoping step before deciding whether to rewrite pool_persistence.py/
consolidate_fragments.py.)

Generalizes the ADOPTED REML+Hartung-Knapp pooling from a diagonal-covariance one-way
random-effects model (months independent given tau2) to a GLS version with an explicit lag-1
correlation term in the within-venue error, matching what estimate_temporal_dependence.py found:
real excess serial correlation concentrated at lag 1 (~+0.16 to +0.18), fading to ~zero by lag 2+.

Model: Sigma_v = tau2*J + D^(1/2) C(rho) D^(1/2), where D = diag(var_combined per eligible month),
J = all-ones matrix (the shared random effect u_v, same as before), and C(rho) has 1s on the
diagonal and rho ONLY on entries for a pair of eligible months that are CALENDAR-adjacent (one
month apart on the actual Dec2025-Jul2026 timeline, not just "next in this venue's own eligible
list" -- a venue eligible in Feb and April has a 2-month calendar gap, not lag 1, even though
April is its next observed month). rho is fixed at a measured point estimate (same
single-point-from-an-honestly-reported-range convention as S_hat in Note 5), not re-estimated per
venue -- there isn't remotely enough per-venue data to estimate rho AND tau2 both.

REML and Hartung-Knapp generalize to this Sigma cleanly (standard GLS form -- see the
module-level comments in each function): the diagonal case already in pool_persistence.py is the
rho=0 special case of this same code, which is the main sanity check run below before trusting
anything else.

This script does NOT touch any committed output. It loads the same inputs pool_persistence.py
does, runs OLD (diagonal, adopted) vs NEW (GLS, prototype) side by side on a subset (the 4 named
venues plus a stratified random sample across n_months buckets), and reports whether the
generalization behaves sanely (reduces to the old numbers at rho=0, widens CIs as expected at
rho>0, doesn't blow up numerically) before any decision to rewrite the adopted scripts.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
MONTH_IDX = {mm: i for i, mm in enumerate(PERIODS)}  # calendar position, 0..7
RNG_SEED = 20261018
#Point estimate from estimate_temporal_dependence.py's fresh direct check on r_vm (mean of the
#raw-demeaned and variance-standardized lag-1 excess estimates, 0.156 and 0.180): see that
#script's output. Sensitivity range brackets the two.
RHO_POINT = 0.17
RHO_SENSITIVITY = (0.10, 0.17, 0.25)

# ---------------------------------------------------------------------------
# Load the exact same inputs pool_persistence.py does.
# ---------------------------------------------------------------------------
frames = []
for mm in PERIODS:
    d = pd.read_pickle(os.path.join(OUTPUT_DIR, f"results_mixing_{mm}.pkl"))
    rf = d["results_full"][["placekey", "improbable_copresence_cv", "location_name"]].copy()
    rf["month"] = mm
    vs = pd.read_csv(os.path.join(OUTPUT_DIR, f"visitor_sampling_variance_dm_{mm}.csv"))[
        ["placekey", "resid_vs_var_dm"]
    ].rename(columns={"resid_vs_var_dm": "resid_vs_var"})
    rf = rf.merge(vs, on="placekey", how="left")
    mv = pd.read_csv(os.path.join(OUTPUT_DIR, f"model_prediction_variance_{mm}.csv"))[
        ["placekey", "model_pred_var"]
    ]
    rf = rf.merge(mv, on="placekey", how="left")
    frames.append(rf)

long = pd.concat(frames, ignore_index=True)
long = long[long["resid_vs_var"] > 0].copy()
long["var_combined"] = long["resid_vs_var"] + long["model_pred_var"].fillna(0.0)
long["month_idx"] = long["month"].map(MONTH_IDX)
print(f"total venue-months: {len(long)}, distinct venues: {long['placekey'].nunique()}")

per_venue_m = long.groupby("placekey").size()


# ---------------------------------------------------------------------------
# OLD: exact copy of pool_persistence.py's adopted diagonal REML+HK (for side-by-side comparison).
# ---------------------------------------------------------------------------
def reml_tau2_diag(r, v):
    def negll(tau2):
        w = 1.0 / (v + tau2)
        sw = w.sum()
        mu = np.sum(w * r) / sw
        resid2 = (r - mu) ** 2
        ll = -0.5 * np.sum(np.log(v + tau2)) - 0.5 * np.log(sw) - 0.5 * np.sum(w * resid2)
        return -ll
    upper = max(0.25, 20.0 * np.var(r))
    res = minimize_scalar(negll, bounds=(0.0, upper), method="bounded")
    return max(0.0, res.x)


def hartung_knapp_diag(r, v, tau2):
    m = len(r)
    w = 1.0 / (v + tau2)
    sw = w.sum()
    mu_hk = np.sum(w * r) / sw
    q_stat = np.sum(w * (r - mu_hk) ** 2) / (m - 1)
    var_classical = 1.0 / sw
    var_hk = max(q_stat / sw, var_classical)
    return mu_hk, var_hk


# ---------------------------------------------------------------------------
# NEW (prototype): GLS generalization with a lag-1-only working correlation. At rho=0, Sigma
# reduces to tau2*J + diag(v), and every formula below reduces EXACTLY to the diagonal case above
# (checked, not assumed -- see sanity check in __main__).
# ---------------------------------------------------------------------------
def build_sigma(v, month_idx, tau2, rho):
    """Sigma_ii = tau2 + v_i (same marginal variance as the diagonal/adopted model -- tau2 is
    still an IID-across-months occasion effect, NOT a single shared venue-level draw, matching
    the standard random-effects meta-analysis convention the adopted method already uses).
    Sigma_ij (i!=j) = rho*sqrt(v_i*v_j) for a CALENDAR-adjacent pair of eligible months, else 0 --
    the lag-1-only excess correlation estimate_temporal_dependence.py found lives in the
    sampling/model-error component (v), not in a longer-range shared random effect (which would
    show equal correlation at every lag, not the observed lag-1-only decay). At rho=0 this is
    EXACTLY diag(tau2 + v), the adopted method's matrix -- see the sanity check in __main__."""
    m = len(v)
    gap = np.abs(month_idx[:, None] - month_idx[None, :])
    adj = (gap == 1).astype(float)
    np.fill_diagonal(adj, 0.0)
    sqrt_v = np.sqrt(v)
    off_diag = rho * sqrt_v[:, None] * sqrt_v[None, :] * adj
    return np.diag(tau2 + v) + off_diag


def reml_tau2_gls(r, v, month_idx, rho):
    m = len(r)
    def negll(tau2):
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
        quad = resid @ Sinv @ resid
        ll = -0.5 * logdet - 0.5 * np.log(sw) - 0.5 * quad
        return -ll
    upper = max(0.25, 20.0 * np.var(r))
    res = minimize_scalar(negll, bounds=(0.0, upper), method="bounded")
    return max(0.0, res.x)


def gls_hartung_knapp(r, v, month_idx, tau2, rho):
    m = len(r)
    Sigma = build_sigma(v, month_idx, tau2, rho)
    Sinv = np.linalg.inv(Sigma)
    ones = np.ones(m)
    sw = ones @ Sinv @ ones
    mu_hk = (ones @ Sinv @ r) / sw
    resid = r - mu_hk
    q_stat = (resid @ Sinv @ resid) / (m - 1)
    var_classical = 1.0 / sw
    var_hk = max(q_stat / sw, var_classical)
    return mu_hk, var_hk


def pool_one(r, v, month_idx, rho):
    """Full OLD-vs-NEW comparison for one venue. m>=2 required by caller."""
    m = len(r)
    tau2_old = reml_tau2_diag(r, v)
    mu_old, var_old = hartung_knapp_diag(r, v, tau2_old)
    tau2_new = reml_tau2_gls(r, v, month_idx, rho)
    mu_new, var_new = gls_hartung_knapp(r, v, month_idx, tau2_new, rho)

    def ci_p(mu, var):
        se = np.sqrt(var)
        t_crit = t_dist.ppf(0.975, df=m - 1)
        ci = (mu - t_crit * se, mu + t_crit * se)
        t_stat = mu / se
        p = 2.0 * t_dist.sf(abs(t_stat), df=m - 1)
        return ci, p

    ci_old, p_old = ci_p(mu_old, var_old)
    ci_new, p_new = ci_p(mu_new, var_new)
    return {
        "m": m, "tau2_old": tau2_old, "mu_old": mu_old, "var_old": var_old,
        "ci_old_lo": ci_old[0], "ci_old_hi": ci_old[1], "p_old": p_old,
        "tau2_new": tau2_new, "mu_new": mu_new, "var_new": var_new,
        "ci_new_lo": ci_new[0], "ci_new_hi": ci_new[1], "p_new": p_new,
        "var_ratio": var_new / var_old if var_old > 0 else np.nan,
    }


if __name__ == "__main__":
    # --- Sanity check: rho=0 must reproduce the diagonal method exactly. ---
    print("=== SANITY CHECK: rho=0 must exactly reproduce the adopted diagonal method ===")
    rng = np.random.default_rng(0)
    check_pks = rng.choice(long["placekey"].unique(), size=200, replace=False)
    max_diff_mu, max_diff_var, max_diff_tau2 = 0.0, 0.0, 0.0
    n_checked = 0
    for pk in check_pks:
        g = long[long["placekey"] == pk]
        if len(g) < 2:
            continue
        r = g["improbable_copresence_cv"].to_numpy()
        v = g["var_combined"].to_numpy()
        midx = g["month_idx"].to_numpy()
        res = pool_one(r, v, midx, rho=0.0)
        max_diff_mu = max(max_diff_mu, abs(res["mu_old"] - res["mu_new"]))
        max_diff_var = max(max_diff_var, abs(res["var_old"] - res["var_new"]))
        max_diff_tau2 = max(max_diff_tau2, abs(res["tau2_old"] - res["tau2_new"]))
        n_checked += 1
    print(f"checked {n_checked} venues: max|mu_old-mu_new|={max_diff_mu:.2e}, "
          f"max|var_old-var_new|={max_diff_var:.2e}, max|tau2_old-tau2_new|={max_diff_tau2:.2e}")
    print("(should all be ~1e-6 or smaller -- numerical-optimizer noise only, confirming the GLS "
          "code is a correct generalization, not a different model)\n")

    # --- Named venues, at the point-estimate rho and the sensitivity range. ---
    print("=== NAMED VENUES: old (diagonal, adopted) vs new (GLS, prototype) ===")
    for name_substr in ["Amenazel", "Chens Fresh", "Jacob Riis", "Hunts Point Floating"]:
        hit_pks = long[long["location_name"].str.contains(name_substr, case=False, na=False)]["placekey"].unique()
        for pk in hit_pks:
            g = long[long["placekey"] == pk]
            name = g["location_name"].iloc[0]
            r = g["improbable_copresence_cv"].to_numpy()
            v = g["var_combined"].to_numpy()
            midx = g["month_idx"].to_numpy()
            m = len(g)
            if m < 2:
                print(f"\n{name} ({pk}): only {m} eligible month(s), not poolable -- skipped")
                continue
            print(f"\n{name} ({pk}), m={m}, months={sorted(g['month'].tolist())}:")
            res0 = pool_one(r, v, midx, rho=0.0)
            print(f"  OLD (adopted):        mu_hk={res0['mu_old']:+.5f}  var_hk={res0['var_old']:.6f}  "
                  f"CI=[{res0['ci_old_lo']:+.5f}, {res0['ci_old_hi']:+.5f}]  p={res0['p_old']:.4f}  "
                  f"tau2={res0['tau2_old']:.6f}")
            for rho in RHO_SENSITIVITY:
                res = pool_one(r, v, midx, rho=rho)
                flag = " <-- CI endpoint sign flips" if (res0['ci_old_lo']>0 or res0['ci_old_hi']<0) != (res['ci_new_lo']>0 or res['ci_new_hi']<0) else ""
                print(f"  NEW rho={rho:.2f}:          mu_hk={res['mu_new']:+.5f}  var_hk={res['var_new']:.6f}  "
                      f"CI=[{res['ci_new_lo']:+.5f}, {res['ci_new_hi']:+.5f}]  p={res['p_new']:.4f}  "
                      f"tau2={res['tau2_new']:.6f}  var_ratio={res['var_ratio']:.3f}{flag}")

    # --- Stratified random subset across n_months buckets, at the point-estimate rho. ---
    print("\n=== STRATIFIED SUBSET (rho=0.17 point estimate): effect summary by n_months ===")
    rng2 = np.random.default_rng(RNG_SEED)
    rows = []
    for m_bucket in range(2, 9):
        pool = per_venue_m[per_venue_m == m_bucket].index.to_numpy()
        n_take = min(300, len(pool))
        sample = rng2.choice(pool, size=n_take, replace=False) if n_take else []
        for pk in sample:
            g = long[long["placekey"] == pk]
            r = g["improbable_copresence_cv"].to_numpy()
            v = g["var_combined"].to_numpy()
            midx = g["month_idx"].to_numpy()
            res = pool_one(r, v, midx, rho=RHO_POINT)
            sig_old = (res["ci_old_lo"] > 0) or (res["ci_old_hi"] < 0)
            sig_new = (res["ci_new_lo"] > 0) or (res["ci_new_hi"] < 0)
            rows.append({"placekey": pk, "m": m_bucket, "var_ratio": res["var_ratio"],
                         "mu_shift": res["mu_new"] - res["mu_old"],
                         "sig_old": sig_old, "sig_new": sig_new})
    sub = pd.DataFrame(rows)
    sub.to_csv(os.path.join(OUTPUT_DIR, "prototype_ar1_subset_results.csv"), index=False)
    for m_bucket, grp in sub.groupby("m"):
        flips_lost = ((grp["sig_old"]) & (~grp["sig_new"])).sum()
        flips_gained = ((~grp["sig_old"]) & (grp["sig_new"])).sum()
        print(f"n_months={m_bucket} (n={len(grp)}): median var_ratio={grp['var_ratio'].median():.3f}  "
              f"(IQR [{grp['var_ratio'].quantile(.25):.3f}, {grp['var_ratio'].quantile(.75):.3f}])  "
              f"median|mu_shift|={grp['mu_shift'].abs().median():.6f}  "
              f"unadj-CI sig->nonsig: {flips_lost}  nonsig->sig: {flips_gained}")
    print(f"\noverall (n={len(sub)}): median var_ratio={sub['var_ratio'].median():.3f}  "
          f"unadj-CI sig->nonsig: {((sub['sig_old']) & (~sub['sig_new'])).sum()}  "
          f"nonsig->sig: {((~sub['sig_old']) & (sub['sig_new'])).sum()}")
    print(f"\nsaved {OUTPUT_DIR}/prototype_ar1_subset_results.csv")
