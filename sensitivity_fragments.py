"""
sensitivity_fragments.py

Point 6 (sensitivity analyses), part 1: stress-tests the two consolidation thresholds
(DIST_THRESHOLD_FT=75, NAME_JACCARD_THRESHOLD=0.4) chosen somewhat by eye in
consolidate_fragments.py. Reruns the full clustering -> merge -> bootstrap -> REML+Hartung-Knapp
-> Benjamini-Hochberg FDR pipeline (Steps 2-7 of consolidate_fragments.py) across a 42-combination
grid of nearby threshold values (distance in {40,50,60,75,100,125,150}ft x name-token Jaccard in
{0.25,0.3,0.35,0.4,0.5,0.6}), built on one shared KD-tree query at the widest radius rather than
rebuilding the tree per combination.

EXTENDED after external review (a sixth round): the original version of this script only checked
whether the grid moved the raw GROUP COUNT and the Jacob Riis Rockaway/Brooklyn sanity checks --
never whether it moved the actual FDR CONCLUSIONS this project reports (the ADOPTED REML+
Hartung-Knapp survivor count, and the four named venues' own significance calls). That's the gap
this version closes: every grid point now also reruns consolidate_fragments.py's Steps 5-7 (merge,
bootstrap, REML+Hartung-Knapp, BH FDR), not just its clustering step. Cost note: these steps only
touch the ~100-770 fragment GROUPS per period (cheap -- a 500-draw bootstrap over at most a few
hundred group-months) plus the same REML/Hartung-Knapp optimization over ~31,650 entities
consolidate_fragments.py already does once -- no RandomForest refit anywhere, so the full 42-point
grid runs in about 2 minutes end to end on top of a one-time ~1-second data load.

Checked at every grid point, not just the one combination that was picked: whether the unrelated
Brooklyn "Jacob Riis Beach" stays excluded, whether the Rockaway Jacob Riis cluster's 4 real
fragments stay correctly grouped together, the total ADOPTED FDR survivor count, and the
significance call (q_bh_hk<=0.05) for each of the four named venues (the Jacob Riis consolidated
group, Chens Fresh Food Market, Amenazel Organic, Hunts Point Floating Pier).

Input: model_ready_<PERIOD>.pkl, results_mixing_<PERIOD>.pkl, model_prediction_variance_<PERIOD>.csv,
       visitor_sampling_variance_dm_<PERIOD>.csv for all 8 periods, overdispersion_estimate.pkl
       (set OUTPUT_DIR, default ./output -- same directory every other pipeline script writes to).
Output: sensitivity_fragments_results.csv -- one row per (distance, Jaccard) grid combination.
"""
import math
import os
import re
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
import networkx as nx

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
MODEL_READY_PATHS = {mm: os.path.join(OUTPUT_DIR, f"model_ready_{mm}.pkl") for mm in PERIODS}
RESULTS_FULL_PATHS = {mm: os.path.join(OUTPUT_DIR, f"results_mixing_{mm}.pkl") for mm in PERIODS}
MODEL_VAR_PATHS = {mm: os.path.join(OUTPUT_DIR, f"model_prediction_variance_{mm}.csv") for mm in PERIODS}
VS_VAR_PATHS = {mm: os.path.join(OUTPUT_DIR, f"visitor_sampling_variance_dm_{mm}.csv") for mm in PERIODS}
S_HAT = float(os.environ.get("S_HAT", pd.read_pickle(
    os.path.join(OUTPUT_DIR, "overdispersion_estimate.pkl"))["S_hat"]))
QCOLS = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
N_BOOT_VS = 500
Z95 = 1.959963984540054
LOG5 = np.log(5)

STOPWORDS = {
    "the", "a", "an", "at", "of", "and", "&", "inc", "llc", "corp", "corporation", "ny", "nyc",
    "street", "st", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive", "way",
    "pl", "place", "plaza", "court", "ct", "sq", "square", "pkwy", "parkway", "hwy", "highway",
    "ln", "lane",
}
#the 4 real members of the Rockaway Jacob Riis group (GRP00228 in fragment_groups_report.csv):
#"Jacob Riis Park", "The People's Beach at Jacob Riis Park", "Jacob Riis Beach" (the 3 duplicate
#catalog entries for the beach itself -- see consolidate_fragments.py's docstring) plus "Riis Park
#Visitor Center", a 4th, differently-named POI close enough and name-similar enough to cluster in
#too. Not the same list as consolidate_fragments.py's own `jr_pks` sanity-check list, which drops
#the visitor center and instead appends the unrelated Brooklyn placekey for its own separate print.
JACOB_RIIS_ROCKAWAY_PKS = {"zzy-222@627-wgg-sbk", "zzy-228@627-tmh-49z", "zzy-22f@627-tmh-49z",
                           "zzy-22b@627-tmh-49z"}
JACOB_RIIS_BROOKLYN_PK = "zzy-223@627-s92-gc5"
NAMED_PKS = {
    "chens_fresh_food_market": "22v-223@627-s84-bhq",
    "amenazel_organic": "224-224@627-s6c-m8v",
    "hunts_point_floating_pier": "zzy-222@627-rwq-m49",
}

DIST_GRID = [40.0, 50.0, 60.0, 75.0, 100.0, 125.0, 150.0]
JAC_GRID = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


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


t0 = time.time()

#---- Step 1 (ONCE): static attributes + all per-period inputs, same as consolidate_fragments.py ----
static_frames = []
model_ready = {}
model_var = {mm: pd.read_csv(MODEL_VAR_PATHS[mm])[["placekey", "model_pred_var"]] for mm in PERIODS}
vs_var_dm = {
    mm: pd.read_csv(VS_VAR_PATHS[mm])[["placekey", "resid_vs_mean_dm", "resid_vs_var_dm"]]
    for mm in PERIODS
}
results_full = {}
for mm, path in MODEL_READY_PATHS.items():
    df = pd.read_pickle(path)
    model_ready[mm] = df[["placekey", "location_name", "top_category"] + QCOLS
                          + ["income_known_devices", "income_mixing_score"]].copy()
    static_frames.append(df[["placekey", "location_name", "top_category", "latitude", "longitude"]])
    results_full[mm] = pd.read_pickle(RESULTS_FULL_PATHS[mm])["results_full"][
        ["placekey", "borough", "expected_mixing_cv", "improbable_copresence_cv"]]

static_all = pd.concat(static_frames, ignore_index=True).drop_duplicates("placekey").reset_index(drop=True)
print(f"[{time.time()-t0:.0f}s] union of eligible placekeys across all {len(PERIODS)} periods: {len(static_all)}")

gdf = gpd.GeoDataFrame(
    static_all, geometry=gpd.points_from_xy(static_all["longitude"], static_all["latitude"]), crs="EPSG:4326"
).to_crs(epsg=2263)
xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
names_tok = static_all["location_name"].apply(name_tokens).to_numpy()
cats = static_all["top_category"].to_numpy()
pks = static_all["placekey"].to_numpy()

#widest distance needed, build the KD-tree pairs once at the max radius, then filter down per
#grid point; same-category + jaccard are also independent of the grid point's distance threshold,
#so both are precomputed once per pair rather than recomputed 42 times.
tree = cKDTree(xy)
max_pairs = tree.query_pairs(r=max(DIST_GRID), output_type="ndarray")
pair_dists = np.hypot(xy[max_pairs[:, 0], 0] - xy[max_pairs[:, 1], 0],
                       xy[max_pairs[:, 0], 1] - xy[max_pairs[:, 1], 1])
pair_same_cat = cats[max_pairs[:, 0]] == cats[max_pairs[:, 1]]
pair_jac = np.array([jaccard(names_tok[i], names_tok[j]) for i, j in max_pairs])
print(f"[{time.time()-t0:.0f}s] precomputed {len(max_pairs)} candidate pairs at the widest radius "
      f"({max(DIST_GRID)}ft), ready for the grid sweep")

rng_merge_seed = 7  #same seed consolidate_fragments.py uses, for comparability
results = []

for dist_thresh in DIST_GRID:
    for jac_thresh in JAC_GRID:
        tC0 = time.time()
        keep = (pair_dists <= dist_thresh) & pair_same_cat & (pair_jac >= jac_thresh)
        pairs_d = max_pairs[keep]

        G = nx.Graph()
        G.add_nodes_from(pks)
        G.add_edges_from([(pks[i], pks[j]) for i, j in pairs_d])
        components = [c for c in nx.connected_components(G) if len(c) > 1]
        n_groups = len(components)

        pk_to_group = {}
        for gi, comp in enumerate(components):
            gid = f"GRP{gi:05d}"
            for pk in comp:
                pk_to_group[pk] = gid

        riis_groups = {pk_to_group.get(pk, f"singleton:{pk}") for pk in JACOB_RIIS_ROCKAWAY_PKS}
        riis_correctly_grouped = len(riis_groups) == 1 and not str(list(riis_groups)[0]).startswith("singleton")
        brooklyn_group = pk_to_group.get(JACOB_RIIS_BROOKLYN_PK, "singleton")
        brooklyn_correctly_excluded = brooklyn_group not in riis_groups
        jr_group_id = list(riis_groups)[0] if riis_correctly_grouped else None

        #---- merge grouped fragments per period, bootstrap merged uncertainty (same method as
        #consolidate_fragments.py Steps 5/6) ----
        rng_merge = np.random.RandomState(rng_merge_seed)
        long_rows = []
        for mm in PERIODS:
            mr = model_ready[mm]
            merged_in = mr.merge(results_full[mm], on="placekey", how="inner") \
                          .merge(vs_var_dm[mm][["placekey", "resid_vs_var_dm"]]
                                 .rename(columns={"resid_vs_var_dm": "resid_vs_var"}), on="placekey", how="left") \
                          .merge(model_var[mm], on="placekey", how="left")
            merged_in["group_id"] = merged_in["placekey"].map(pk_to_group)
            grouped_part = merged_in[merged_in["group_id"].notna()]
            singleton_part = merged_in[merged_in["group_id"].isna()]

            for gid, g in grouped_part.groupby("group_id"):
                q = g[QCOLS].to_numpy(dtype=float)
                q_total = q.sum()
                if q_total <= 0:
                    continue
                w = g["income_known_devices"].to_numpy(dtype=float)
                merged_score = np.average(g["income_mixing_score"].to_numpy(), weights=w)
                merged_expected_cv = np.average(g["expected_mixing_cv"].to_numpy(), weights=w)
                merged_resid = merged_score - merged_expected_cv
                mpv = g["model_pred_var"].to_numpy(dtype=float)
                if np.isnan(mpv).all():
                    merged_model_pred_var = np.nan
                else:
                    valid = ~np.isnan(mpv)
                    merged_model_pred_var = np.average(mpv[valid], weights=w[valid])

                row_totals = q.sum(axis=1)
                row_totals_safe = np.where(row_totals > 0, row_totals, 1.0)
                q_shares = q / row_totals_safe[:, None]
                alphas = np.clip(S_HAT * q_shares, 1e-6, None)
                has_devices = row_totals > 0
                draws = np.empty(N_BOOT_VS)
                frag_scores = np.empty(len(g))
                for b in range(N_BOOT_VS):
                    for fi in range(len(g)):
                        if not has_devices[fi]:
                            frag_scores[fi] = np.nan
                            continue
                        p_tilde = rng_merge.dirichlet(alphas[fi])
                        resampled = rng_merge.multinomial(int(round(row_totals[fi])), p_tilde)
                        rp = np.clip(resampled / row_totals[fi], 1e-12, None)
                        frag_scores[fi] = -(rp * np.log(rp)).sum() / LOG5
                    valid = ~np.isnan(frag_scores)
                    draws[b] = np.average(frag_scores[valid], weights=w[valid]) - merged_expected_cv

                long_rows.append({
                    "placekey": gid, "month": mm, "resid_observed": merged_resid,
                    "resid_vs_var": draws.var(ddof=1), "model_pred_var": merged_model_pred_var,
                })

            for _, row in singleton_part.iterrows():
                long_rows.append({
                    "placekey": row["placekey"], "month": mm,
                    "resid_observed": row["improbable_copresence_cv"],
                    "resid_vs_var": row["resid_vs_var"], "model_pred_var": row["model_pred_var"],
                })

        long = pd.DataFrame(long_rows)
        long = long[long["resid_vs_var"] > 0].copy()
        long["var_combined"] = long["resid_vs_var"] + long["model_pred_var"].fillna(0.0)
        long["w"] = 1.0 / long["resid_vs_var"]

        #---- REML + Hartung-Knapp + BH FDR, same as consolidate_fragments.py Step 7 ----
        records = []
        for pk, g in long.groupby("placekey"):
            r = g["resid_observed"].to_numpy()
            v_combined = g["var_combined"].to_numpy()
            m = len(g)
            if m < 2:
                records.append({"placekey": pk, "n_months": m, "q_bh_hk": np.nan, "p_hk": np.nan})
                continue
            tau2_reml = reml_tau2(r, v_combined)
            mu_hk, var_hk = hartung_knapp(r, v_combined, tau2_reml)
            t_stat_hk = mu_hk / np.sqrt(var_hk)
            p_hk = 2.0 * t_dist.sf(abs(t_stat_hk), df=m - 1)
            records.append({"placekey": pk, "n_months": m, "mu_hk": mu_hk, "q_bh_hk": np.nan, "p_hk": p_hk})

        pooled = pd.DataFrame(records)
        family_mask = (pooled["n_months"] >= 2).to_numpy()
        q_vals_hk, sig_hk = benjamini_hochberg(pooled.loc[family_mask, "p_hk"].to_numpy(), alpha=0.05)
        pooled.loc[family_mask, "q_bh_hk"] = q_vals_hk
        pooled["fdr_sig_05_hk"] = pooled["q_bh_hk"] <= 0.05
        n_survivors = int(pooled["fdr_sig_05_hk"].sum())
        n_family = int(family_mask.sum())

        named_result = {}
        pooled_idx = pooled.set_index("placekey")
        for label, pk in NAMED_PKS.items():
            if pk in pooled_idx.index:
                named_result[f"{label}_q_bh_hk"] = pooled_idx.loc[pk, "q_bh_hk"]
                named_result[f"{label}_sig"] = bool(pooled_idx.loc[pk, "fdr_sig_05_hk"])
            else:
                named_result[f"{label}_q_bh_hk"] = np.nan
                named_result[f"{label}_sig"] = None
        if jr_group_id is not None and jr_group_id in pooled_idx.index:
            named_result["jacob_riis_q_bh_hk"] = pooled_idx.loc[jr_group_id, "q_bh_hk"]
            named_result["jacob_riis_sig"] = bool(pooled_idx.loc[jr_group_id, "fdr_sig_05_hk"])
        else:
            named_result["jacob_riis_q_bh_hk"] = np.nan
            named_result["jacob_riis_sig"] = None

        results.append({
            "dist_threshold_ft": dist_thresh, "jaccard_threshold": jac_thresh,
            "n_groups": n_groups,
            "jacob_riis_rockaway_correctly_grouped": riis_correctly_grouped,
            "jacob_riis_brooklyn_correctly_excluded": brooklyn_correctly_excluded,
            "n_family": n_family, "n_survivors_hk": n_survivors,
            **named_result,
        })
        print(f"[{time.time()-t0:.0f}s] dist={dist_thresh}ft jac={jac_thresh}: n_groups={n_groups}, "
              f"survivors={n_survivors}/{n_family} ({time.time()-tC0:.1f}s this combo)")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_fragments_results.csv"), index=False)
pd.set_option("display.width", 200)
print(f"\n[{time.time()-t0:.0f}s] FULL RESULTS:")
print(res_df.to_string(index=False))

chosen = res_df[(res_df.dist_threshold_ft == 75.0) & (res_df.jaccard_threshold == 0.4)]
print(f"\nchosen combination (75ft, 0.4): {chosen.to_string(index=False)}")
print(f"\nJacob Riis Rockaway group correct in {res_df['jacob_riis_rockaway_correctly_grouped'].sum()}/{len(res_df)} "
      f"grid combinations; Brooklyn correctly excluded in "
      f"{res_df['jacob_riis_brooklyn_correctly_excluded'].sum()}/{len(res_df)}")
print(f"group count range across grid: {res_df['n_groups'].min()}-{res_df['n_groups'].max()} "
      f"(chosen: {chosen['n_groups'].iloc[0]})")
print(f"ADOPTED FDR survivor count range across grid: {res_df['n_survivors_hk'].min()}-"
      f"{res_df['n_survivors_hk'].max()} (chosen: {chosen['n_survivors_hk'].iloc[0]}, "
      f"real production value: 4871)")
for label in NAMED_PKS:
    col = f"{label}_sig"
    print(f"{label}: significant in {res_df[col].sum()}/{len(res_df)} grid combinations "
          f"(chosen combo: {chosen[col].iloc[0]})")
print(f"jacob_riis (consolidated group): significant in {res_df['jacob_riis_sig'].sum()}/{len(res_df)} "
      f"grid combinations (chosen combo: {chosen['jacob_riis_sig'].iloc[0]})")
print(f"\n[{time.time()-t0:.0f}s] TOTAL DONE")
