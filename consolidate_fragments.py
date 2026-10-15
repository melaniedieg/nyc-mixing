"""
consolidate_fragments.py

Citywide POI-fragment consolidation (point 7). The Jacob Riis check showed the real
problem: three catalog entries ("Jacob Riis Park", "The People's Beach at Jacob Riis
Park", "Jacob Riis Beach") sit within 2-6 meters of each other -- genuine duplicate
fragments of the same beach -- while a FOURTH "Jacob Riis Beach" sits 14.6km away in
Brooklyn, a real, different, unrelated small park that just shares part of a name. A
naive name-substring match conflates the two cases. This script instead clusters by
BOTH geographic proximity AND name-token similarity (plus a same-category sanity
check), so it catches genuine duplicate fragments without merging unrelated venues
that happen to be nearby or share a name.

Method:
  1. Pull static attributes (location_name, top_category, latitude, longitude) for
     every placekey that was ELIGIBLE in at least one of the four months (the union
     of each month's model_ready.pkl) -- consolidation only matters for venues that
     actually enter the pooled analysis.
  2. Project to EPSG:2263 (NY State Plane feet, same projection already used for the
     grid-cell spatial CV blocking) and build a KD-tree; find all pairs within
     DIST_THRESHOLD_FT (75ft =~ 23m).
  3. For each candidate pair, require (a) same top_category, and (b) name-token
     Jaccard similarity >= NAME_JACCARD_THRESHOLD (0.4) after lowercasing and
     stripping common stopwords/punctuation. Both conditions must hold -- proximity
     alone would merge distinct businesses sharing a building/block; name similarity
     alone would merge distant same-named venues (exactly the Brooklyn Jacob Riis
     case).
  4. Connected components over the surviving edges = consolidated venue groups.
     Singletons pass through unchanged.
  5. For each month, re-derive EVERY consolidated group's quintile counts by summing
     its present fragments' q1..q5_devices (from model_ready), recompute its merged
     income_mixing_score from the vendor entropy formula, and set its "expected"
     reference as the device-count-weighted average of its fragments' own
     expected_mixing_cv (from results_full).
     CAVEAT, stated plainly: this is a combination rule over each fragment's ALREADY
     FITTED prediction, not a full model refit on a merged covariate row. A true
     refit would require rebuilding density/local-income-context features for the
     merged entity (which also changes its *neighbors'* density features, since two
     fewer POIs now occupy that footprint) -- a much bigger undertaking than this
     pass. This combination rule is reasonable for seeing whether consolidation
     changes the persistence picture, but isn't a substitute for a future full refit
     on consolidated features.
  6. Bootstrap the merged group's OWN visitor-sampling uncertainty (500 multinomial
     draws on the merged quintile counts, residual against the merged fixed
     expected_cv) -- the same method as the per-venue bootstrap, just applied to the
     combined counts.
  7. Replace the fragment rows with one merged row per group per month, and re-run
     the exact same pooling + Benjamini-Hochberg logic from pool_persistence.py.

Output: OUTPUT_DIR/consolidated_pooled_persistence_2026.{pkl,csv}, plus a fragment-group report
        (group_id, n_fragments, member placekeys/names, distances) for inspection. Set
        OUTPUT_DIR to the same directory build_features_2026_0X.py / fit_mixing_model_2026_0X.py
        wrote their per-month outputs to; default ./output.

POINT ESTIMATE, fixed after external review (see pool_persistence.py's docstring for the full
explanation): both the merged-group and singleton point estimates below are the OBSERVED
residual (score - expected_mixing_cv), not the visitor-sampling bootstrap mean -- the bootstrap
mean is a biased estimator of the point estimate (concave entropy -> Jensen's-inequality bias),
even though it's exactly right for estimating variance, which is still what resid_vs_var feeds
into the pooling weight below. The pooled CIs here are CONDITIONAL on the fitted model, same
caveat as pool_persistence.py -- they do not yet include model-refit/retraining uncertainty.
"""
import math
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
import networkx as nx

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")  #same dir build_features/fit_mixing wrote to
#Full "YYYY_MM" period labels -- see the matching comment in pool_persistence.py. Lets a
#period outside 2026 (e.g. "2025_12") be folded in without assuming every period is "2026_<mm>".
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
MODEL_READY_PATHS = {
    mm: os.path.join(OUTPUT_DIR, f"model_ready_{mm}.pkl") for mm in PERIODS
}
RESULTS_FULL_PATHS = {
    mm: os.path.join(OUTPUT_DIR, f"results_mixing_{mm}.pkl") for mm in PERIODS
}
QCOLS = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
DIST_THRESHOLD_FT = 75.0  #~23m
NAME_JACCARD_THRESHOLD = 0.4
N_BOOT_VS = 500
Z95 = 1.959963984540054
STOPWORDS = {
    "the", "a", "an", "at", "of", "and", "&", "inc", "llc", "corp", "corporation", "ny", "nyc",
    #street-type tokens: two DIFFERENT businesses on the same named street (e.g. "Knapp Street
    #Pizza" vs. "Knapp Street Bagel Cafe") otherwise inflate name-token Jaccard just from
    #sharing the street name, not from being the same venue.
    "street", "st", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive", "way",
    "pl", "place", "plaza", "court", "ct", "sq", "square", "pkwy", "parkway", "hwy", "highway",
    "ln", "lane",
    #NOTE: "park" deliberately NOT included here -- it's part of many venues' actual proper
    #names ("Jacob Riis Park"), unlike a pure street-suffix word, and removing it would have
    #dropped "Riis Park Visitor Center" out of the Jacob Riis group.
}


def name_tokens(name):
    if not isinstance(name, str):
        return set()
    toks = re.findall(r"[a-z0-9']+", name.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


#Same plain-language persistence tiers as pool_persistence.py -- see that script's docstring
#for the full rationale. Duplicated rather than imported since these are standalone scripts.
PERSISTENCE_TIERS = {
    1: "single_month_not_tested",
    2: "repeated_evidence",
    3: "provisional_persistence",
    4: "current_period_persistence",  #NOT "annual_persistence" -- would need all 4 seasons
}


def persistence_tier(n_months):
    return PERSISTENCE_TIERS.get(n_months, "current_period_persistence" if n_months > 4 else "single_month_not_tested")


#ADOPTED METHOD, same as pool_persistence.py -- see that script for the full explanation.
#mu_re/var_re/ci_re/tau2/I2/q_bh below are retained only as a DerSimonian-Laird
#diagnostic/comparison -- mu_hk/var_hk/ci_hk/q_bh_hk are the numbers of record.
def reml_tau2(r, v):
    def neg_restricted_loglik(tau2):
        w = 1.0 / (v + tau2)
        sw = w.sum()
        mu = np.sum(w * r) / sw
        resid2 = (r - mu) ** 2
        ll = -0.5 * np.sum(np.log(v + tau2)) - 0.5 * np.log(sw) - 0.5 * np.sum(w * resid2)
        return -ll
    upper = max(0.25, 20.0 * np.var(r))
    res = minimize_scalar(neg_restricted_loglik, bounds=(0.0, upper), method="bounded")
    return max(0.0, res.x)


def hartung_knapp(r, v, tau2_reml):
    m = len(r)
    w = 1.0 / (v + tau2_reml)
    sw = w.sum()
    mu_hk = np.sum(w * r) / sw
    q_stat = np.sum(w * (r - mu_hk) ** 2) / (m - 1)
    var_classical = 1.0 / sw
    var_hk = max(q_stat / sw, var_classical)
    return mu_hk, var_hk


#---- Step 1: static attributes, union across all 4 months' eligible venues ----
static_frames = []
model_ready = {}
for mm, path in MODEL_READY_PATHS.items():
    df = pd.read_pickle(path)
    model_ready[mm] = df
    static_frames.append(df[["placekey", "location_name", "top_category", "latitude", "longitude"]])

static_all = pd.concat(static_frames, ignore_index=True).drop_duplicates("placekey").reset_index(drop=True)
print(f"union of eligible placekeys across all {len(PERIODS)} periods ({', '.join(PERIODS)}): {len(static_all)}")

#---- Step 2: spatial pairs within DIST_THRESHOLD_FT ----
gdf = gpd.GeoDataFrame(
    static_all, geometry=gpd.points_from_xy(static_all["longitude"], static_all["latitude"]), crs="EPSG:4326"
).to_crs(epsg=2263)
xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
tree = cKDTree(xy)
pairs = tree.query_pairs(r=DIST_THRESHOLD_FT, output_type="ndarray")
print(f"candidate spatial pairs within {DIST_THRESHOLD_FT}ft: {len(pairs)}")

#---- Step 3: filter by same category + name-token Jaccard ----
names_tok = static_all["location_name"].apply(name_tokens).to_numpy()
cats = static_all["top_category"].to_numpy()
pks = static_all["placekey"].to_numpy()

edges = []
for i, j in pairs:
    if cats[i] != cats[j]:
        continue
    jac = jaccard(names_tok[i], names_tok[j])
    if jac >= NAME_JACCARD_THRESHOLD:
        d = np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1])
        edges.append((pks[i], pks[j], jac, d))
print(f"edges surviving same-category + name-Jaccard>={NAME_JACCARD_THRESHOLD}: {len(edges)}")

#---- Step 4: connected components ----
G = nx.Graph()
G.add_nodes_from(pks)
for a, b, jac, d in edges:
    G.add_edge(a, b, jaccard=jac, dist_ft=d)
components = [c for c in nx.connected_components(G) if len(c) > 1]
print(f"fragment groups found (size>1): {len(components)}, "
      f"covering {sum(len(c) for c in components)} placekeys")

group_rows = []
pk_to_group = {}
for gi, comp in enumerate(components):
    gid = f"GRP{gi:05d}"
    members = static_all[static_all["placekey"].isin(comp)]
    for pk in comp:
        pk_to_group[pk] = gid
    names_list = members["location_name"].tolist()
    #flag exact-normalized-name matches (e.g. two "Starbucks" 60ft apart) separately --
    #proximity + identical name is weaker evidence of "same physical venue, duplicate catalog
    #entry" than it is for a venue with a distinctive multi-word name, since it's also
    #consistent with two genuinely different branches of the same chain sitting close
    #together (adjacent subway entrances, across an intersection, etc.). Worth a manual
    #glance rather than auto-trusting.
    normalized = {" ".join(sorted(name_tokens(n))) for n in names_list}
    exact_name_match = len(normalized) == 1
    group_rows.append({
        "group_id": gid, "n_fragments": len(comp),
        "names": "; ".join(names_list),
        "placekeys": ",".join(members["placekey"].tolist()),
        "top_category": members["top_category"].iloc[0],
        "exact_name_match": exact_name_match,
    })
group_report = pd.DataFrame(group_rows).sort_values("n_fragments", ascending=False)
group_report.to_csv(os.path.join(OUTPUT_DIR, "fragment_groups_report.csv"), index=False)
print(f"\nsaved fragment group report: {OUTPUT_DIR}/fragment_groups_report.csv")
print(f"of {len(group_report)} groups, {group_report['exact_name_match'].sum()} are exact-name "
      f"matches (higher chain/franchise false-merge risk -- worth a manual glance, listed "
      f"separately in the saved report).")
print(group_report.head(20).to_string(index=False))

#sanity check: did the three real Jacob Riis Rockaway fragments land in one group, and
#the Brooklyn one get correctly excluded?
jr_pks = ["zzy-222@627-wgg-sbk", "zzy-228@627-tmh-49z", "zzy-22f@627-tmh-49z", "zzy-223@627-s92-gc5"]
print("\nJacob Riis sanity check -- group assignment per placekey:")
for pk in jr_pks:
    print(f"  {pk}: group={pk_to_group.get(pk, 'UNGROUPED (singleton)')}")

#---- Step 5/6: for each month, merge grouped fragments' quintile counts, recompute the
#merged score/residual, and bootstrap the merged entity's own visitor-sampling uncertainty.
#Ungrouped (singleton) venues pass through with their already-computed resid_vs_mean/var
#unchanged -- no need to redo work that consolidation doesn't touch. ----
LOG5 = np.log(5)
rng_merge = np.random.RandomState(7)
long_rows = []

for mm in MODEL_READY_PATHS:
    mr = model_ready[mm][["placekey", "location_name", "top_category"] + QCOLS + ["income_known_devices"]].copy()
    rfull = pd.read_pickle(RESULTS_FULL_PATHS[mm])["results_full"][["placekey", "borough", "expected_mixing_cv",
                                                                     "improbable_copresence_cv", "resid_vs_var"]]
    merged_in = mr.merge(rfull, on="placekey", how="inner")
    merged_in["group_id"] = merged_in["placekey"].map(pk_to_group)

    grouped_part = merged_in[merged_in["group_id"].notna()]
    singleton_part = merged_in[merged_in["group_id"].isna()].copy()
    singleton_part["out_id"] = singleton_part["placekey"]
    singleton_part["n_fragments_this_month"] = 1

    for gid, g in grouped_part.groupby("group_id"):
        q = g[QCOLS].to_numpy(dtype=float)
        q_total = q.sum()
        merged_q = q.sum(axis=0)
        if q_total <= 0:
            continue
        #merged score from combined counts, vendor entropy formula
        p = merged_q / q_total
        p_clipped = np.clip(p, 1e-12, None)
        merged_score = -(p_clipped * np.log(p_clipped)).sum() / LOG5
        #merged "expected" reference: device-count-weighted average of each present fragment's
        #OWN already-fitted expected_mixing_cv (see module docstring -- a combination rule
        #over existing predictions, not a refit on a merged covariate row)
        w = g["income_known_devices"].to_numpy(dtype=float)
        merged_expected_cv = np.average(g["expected_mixing_cv"].to_numpy(), weights=w)
        merged_resid = merged_score - merged_expected_cv

        #bootstrap the merged entity's own visitor-sampling uncertainty, same method as the
        #per-venue bootstrap: resample the COMBINED quintile counts, hold merged_expected_cv
        #fixed, recompute the residual each draw.
        draws = np.empty(N_BOOT_VS)
        for b in range(N_BOOT_VS):
            resampled = rng_merge.multinomial(int(round(q_total)), p)
            rp = np.clip(resampled / q_total, 1e-12, None)
            draws[b] = -(rp * np.log(rp)).sum() / LOG5 - merged_expected_cv

        long_rows.append({
            "placekey": gid, "location_name": " / ".join(sorted(g["location_name"].unique())),
            "borough": g["borough"].mode().iat[0], "top_category": g["top_category"].iloc[0],
            "month": mm,
            #point estimate = the OBSERVED merged residual (merged_score - merged_expected_cv),
            #NOT the bootstrap mean -- see pool_persistence.py's point-estimate note, same fix
            #applied here. draws.mean()/var() below is kept only as a bias diagnostic; var(ddof=1)
            #is still the right visitor-sampling variance to pool with.
            "resid_observed": merged_resid, "resid_vs_mean_bootstrap_diagnostic": draws.mean(),
            "resid_vs_var": draws.var(ddof=1), "n_fragments_this_month": len(g),
        })

    for _, row in singleton_part.iterrows():
        long_rows.append({
            "placekey": row["out_id"], "location_name": row["location_name"], "borough": row["borough"],
            "top_category": row["top_category"], "month": mm,
            "resid_observed": row["improbable_copresence_cv"],
            "resid_vs_var": row["resid_vs_var"], "n_fragments_this_month": 1,
        })
    print(f"month {mm}: {len(grouped_part.groupby('group_id'))} merged groups present, "
          f"{len(singleton_part)} singleton venues")

long = pd.DataFrame(long_rows)
long = long[long["resid_vs_var"] > 0].copy()
long["w"] = 1.0 / long["resid_vs_var"]
print(f"\nconsolidated venue-months: {len(long)}, distinct entities: {long['placekey'].nunique()} "
      f"(was 33767 distinct placekeys before consolidation)")


#---- Step 7: same pooling + Benjamini-Hochberg logic as pool_persistence.py, now run on the
#consolidated entity set instead of raw placekeys. ----
def benjamini_hochberg(pvals, alpha=0.05):
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


static_cols = ["location_name", "borough", "top_category"]
static = long.groupby("placekey")[static_cols].first()

records = []
for pk, g in long.groupby("placekey"):
    r = g["resid_observed"].to_numpy()
    v = g["resid_vs_var"].to_numpy()
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
        mu_re, var_re = mu_fe, var_fe

    ci_re = (mu_re - Z95 * np.sqrt(var_re), mu_re + Z95 * np.sqrt(var_re))
    z_re = mu_re / np.sqrt(var_re)
    prob_positive_re = 1.0 - 0.5 * (1 + math.erf(-z_re / np.sqrt(2)))
    p_re = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z_re) / np.sqrt(2))))

    #ADOPTED pooling result -- see note above reml_tau2/hartung_knapp.
    tau2_reml = mu_hk = var_hk = ci_hk_lo = ci_hk_hi = p_hk = pred_lo = pred_hi = np.nan
    if m >= 2:
        tau2_reml = reml_tau2(r, v)
        mu_hk, var_hk = hartung_knapp(r, v, tau2_reml)
        t_crit_ci = t_dist.ppf(0.975, df=m - 1)
        ci_hk_lo, ci_hk_hi = mu_hk - t_crit_ci * np.sqrt(var_hk), mu_hk + t_crit_ci * np.sqrt(var_hk)
        t_stat_hk = mu_hk / np.sqrt(var_hk)
        p_hk = 2.0 * t_dist.sf(abs(t_stat_hk), df=m - 1)
        if m >= 3:
            t_crit_pi = t_dist.ppf(0.975, df=m - 2)
            half_width = t_crit_pi * np.sqrt(tau2_reml + var_hk)
            pred_lo, pred_hi = mu_hk - half_width, mu_hk + half_width

    records.append({
        "placekey": pk, "n_months": m, "persistence_tier": persistence_tier(m),
        "months": ",".join(sorted(g["month"])),
        "mu_fe": mu_fe, "var_fe": var_fe, "Q": Q, "tau2": tau2, "I2": I2,
        "mu_re": mu_re, "var_re": var_re, "ci_re_lo": ci_re[0], "ci_re_hi": ci_re[1],
        "p_re": p_re, "prob_positive_re": prob_positive_re,
        "tau2_reml": tau2_reml, "mu_hk": mu_hk, "var_hk": var_hk,
        "ci_hk_lo": ci_hk_lo, "ci_hk_hi": ci_hk_hi, "p_hk": p_hk,
        "pred_lo": pred_lo, "pred_hi": pred_hi,
    })

pooled = pd.DataFrame(records).set_index("placekey").join(static).reset_index().sort_values("mu_re", ascending=False)
family_mask = (pooled["n_months"] >= 2).to_numpy()
pooled["q_bh"] = np.nan
q_vals, _ = benjamini_hochberg(pooled.loc[family_mask, "p_re"].to_numpy(), alpha=0.05)
pooled.loc[family_mask, "q_bh"] = q_vals
pooled["fdr_sig_05"] = pooled["q_bh"] <= 0.05
pooled["is_consolidated_group"] = pooled["placekey"].str.startswith("GRP")

#ADOPTED FDR call: same BH correction on the Hartung-Knapp p-values. q_bh_hk/fdr_sig_05_hk
#are the numbers of record; q_bh/fdr_sig_05 (DL-based) above are diagnostic/comparison only.
pooled["q_bh_hk"] = np.nan
q_vals_hk, _ = benjamini_hochberg(pooled.loc[family_mask, "p_hk"].to_numpy(), alpha=0.05)
pooled.loc[family_mask, "q_bh_hk"] = q_vals_hk
pooled["fdr_sig_05_hk"] = pooled["q_bh_hk"] <= 0.05
print(f"[diagnostic/comparison only] DL survivors: {pooled['fdr_sig_05'].sum()} vs. the ADOPTED "
      f"REML+Hartung-Knapp survivors: {pooled['fdr_sig_05_hk'].sum()} (same {family_mask.sum()}-entity family).")

print(f"\n{len(pooled)} consolidated entities pooled ({pooled['is_consolidated_group'].sum()} are "
      f"merged fragment groups, {(~pooled['is_consolidated_group']).sum()} are untouched singletons).")
print(f"Of the {family_mask.sum()} with >=2 eligible months: {pooled['fdr_sig_05_hk'].sum()} survive "
      f"the ADOPTED REML+Hartung-Knapp BH FDR control at q<=0.05 ({pooled['fdr_sig_05'].sum()} survive "
      f"under the DL diagnostic/comparison).")

pooled.to_pickle(os.path.join(OUTPUT_DIR, "consolidated_pooled_persistence_2026.pkl"))
pooled.to_csv(os.path.join(OUTPUT_DIR, "consolidated_pooled_persistence_2026.csv"), index=False)
print(f"\nsaved {OUTPUT_DIR}/consolidated_pooled_persistence_2026.{{pkl,csv}}")

jr_group_id = pk_to_group.get("zzy-222@627-wgg-sbk")  #looked up dynamically: nx component
#numbering isn't stable across runs, so a hardcoded "GRP#####" here would silently point at
#whatever group happened to land on that id in THIS run (bit us once already -- see git history).
print("\n--- Jacob Riis, consolidated (HK columns = numbers of record, ADOPTED; DL = comparison only) ---")
jr_cols = ["placekey", "location_name", "borough", "n_months", "persistence_tier", "months",
           "mu_hk", "tau2_reml", "q_bh_hk", "ci_hk_lo", "ci_hk_hi", "pred_lo", "pred_hi",
           "mu_re", "tau2", "I2", "q_bh", "ci_re_lo", "ci_re_hi"]
print(pooled[pooled["placekey"] == jr_group_id][jr_cols].to_string(index=False))
print("\n(the Brooklyn 'Jacob Riis Beach' is now just its own unaffected singleton entity:)")
print(pooled[pooled["placekey"] == "zzy-223@627-s92-gc5"][jr_cols].to_string(index=False))
