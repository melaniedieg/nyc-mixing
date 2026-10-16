"""
sensitivity_eligibility_floor_fdr.py

Companion to sensitivity_eligibility_floor.py, added after external review (a sixth round):
that script only ever checked whether the 50-device eligibility floor moved CV R^2 for one month
(April). It never checked whether the floor choice moves the actual FDR CONCLUSIONS this project
reports -- the pooled, cross-period survivor count and the four named venues' own significance
calls -- which needs the full production pipeline (buffered CV -> model-refit-variance bootstrap
-> visitor-sampling variance -> REML+Hartung-Knapp pooling -> Benjamini-Hochberg FDR) rerun end to
end, across all 8 periods, at each floor. That is a materially bigger undertaking than a CV-only
check (real RandomForest refits, not just a clustering pass), so it is its own script rather than
folded into sensitivity_eligibility_floor.py -- same reasoning that split
estimate_model_prediction_variance.py and estimate_visitor_sampling_variance_dm.py out of
fit_mixing_model_2026_0X.py as their own required pipeline steps.

Method, at each floor in FLOORS (default 30, 75, 100, 150 -- the same five floors
sensitivity_eligibility_floor.py already tests, minus the published 50):
  1. Eligible venue set per period: RAISING the floor (75, 100, 150) filters the existing
     floor=50 model_ready_<PERIOD>.pkl (no rebuild -- see sensitivity_eligibility_floor.py's
     docstring for why that's valid). LOWERING it to 30 needs an actual rebuild via a patched
     copy of build_features_<PERIOD>.py (INCOME_KNOWN_DEVICE_THRESHOLD=30), run once per period.
  2. A lightweight, buffered-CV-ONLY refit (same BUFFER_FT=1500 spatial buffer as
     fit_mixing_model_2026_0X.py's PRIMARY CV -- see Note 7) for that floor's eligible set, each
     period -- skipping fit_mixing_model_2026_0X.py's two stability bootstraps, since neither
     feeds the pooling/FDR pipeline (which reads resid_vs_var from estimate_visitor_sampling_
     variance_dm.py and model_pred_var from estimate_model_prediction_variance.py instead).
  3. estimate_visitor_sampling_variance_dm.py's own DM-corrected bootstrap, reusing the SAME
     adopted S_hat from overdispersion_estimate.pkl at every floor (a stated simplification --
     re-deriving S_hat separately per floor would need its own multi-period balanced-panel check,
     which this script does not attempt; S_hat is a property of the visitor-sampling process, not
     expected to shift much with the eligibility floor).
  4. estimate_model_prediction_variance.py's own 15-replicate model-refit bootstrap (the
     expensive step -- reruns the full buffered GroupKFold(5) 15 times per period per floor).
  5. pool_persistence.py's own REML+Hartung-Knapp pooling and Benjamini-Hochberg FDR, across all
     8 periods, for that floor.

Compared against the RAW (pre-fragment-consolidation) pool_persistence.py output at the
published floor=50, not the consolidated numbers in consolidate_fragments.py/the README headline
-- fragment consolidation is a separate dimension already stress-tested on its own
(sensitivity_fragments.py), so this isolates the floor's own effect rather than conflating the two.

Compute cost, stated plainly: ~3-3.5 hours across all 4 floors x 8 periods on the data sizes seen
so far, dominated by step 4's 15-replicate bootstrap. Resumable by design -- every intermediate
file is written once and the corresponding step is skipped on a rerun if that file already exists,
so an interrupted run can just be restarted with this same script.

Input:  model_ready_<PERIOD>.pkl / overdispersion_estimate.pkl in OUTPUT_DIR (the published
        floor=50 production outputs); build_features_<PERIOD>.py, one per period, alongside this
        script (needed only for the floor=30 rebuild).
Output: FLOOR_OUTPUT_DIR/floor<F>/ for each floor F -- the same file set pool_persistence.py's own
        pipeline produces (results_mixing_<PERIOD>.pkl, visitor_sampling_variance_dm_<PERIOD>.csv,
        model_prediction_variance_<PERIOD>.csv, pooled_persistence_2026.{pkl,csv}) -- plus
        sensitivity_eligibility_floor_fdr_results.csv, a one-row-per-floor summary (n_venues,
        n_family, n_survivors, survivor rate, and the three non-grouped named venues' mu_hk/
        q_bh_hk/significance at that floor).
"""
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")  #published floor=50 production outputs
FLOOR_OUTPUT_DIR = os.environ.get("FLOOR_OUTPUT_DIR", "./output_floor_sensitivity")
SCRIPT_DIR = os.environ.get("SCRIPT_DIR", ".")  #where build_features_<PERIOD>.py / estimate_*.py live
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
FLOORS = [int(f) for f in os.environ.get("FLOORS", "75,100,150,30").split(",")]
GRID_CELL_FT = 2000
BUFFER_FT = 1500  #Note 7 fix -- same buffer as the adopted production PRIMARY CV
S_HAT = float(os.environ.get("S_HAT", pd.read_pickle(
    os.path.join(OUTPUT_DIR, "overdispersion_estimate.pkl"))["S_hat"]))

NAMED_PKS = {  #the three non-grouped named venues (Jacob Riis is a merged group -- see
               #sensitivity_fragments.py for how the floor interacts with consolidation instead)
    "chens_fresh_food_market": "22v-223@627-s84-bhq",
    "amenazel_organic": "224-224@627-s6c-m8v",
    "hunts_point_floating_pier": "zzy-222@627-rwq-m49",
}

feats_num_race = ["log_devices", "income_proxy_v2", "log_population",
                   "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_nh", "racial_diversity_index"]
feats_num_full = feats_num_race + [
    "log_poi_density_250m", "log_poi_density_500m", "log_same_category_density_250m",
    "log_dist_nearest_subway_m", "log_subway_count_500m", "log_subway_count_800m",
    "log_tourist_density_500m", "log_tourist_density_800m",
    "log_chain_density_500m", "log_chain_density_800m",
    "local_income_proxy_250m", "local_income_proxy_500m",
]
feats_cat = ["top_category", "borough"]


def buffered_primary_cv(model_ready_path, label):
    """Lightweight variant of fit_mixing_model_2026_0X.py's PRIMARY CV block only (buffered
    GroupKFold, same spec) -- skips the SECONDARY random KFold and both stability bootstraps,
    neither of which feeds the pooling/FDR pipeline. Returns a results_full-shaped DataFrame."""
    t0 = time.time()
    df = pd.read_pickle(model_ready_path)
    y = df["income_mixing_score"]
    X_full = df[feats_num_full + feats_cat]
    pre = ColumnTransformer([("num", StandardScaler(), feats_num_full),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])
    sample_weight_full = df["income_known_devices"].values

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=2263)
    gx = (gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
    gy = (gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
    grid_cell = (gx + "_" + gy).values

    xy_all = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    tree_all = cKDTree(xy_all)
    n_groups = len(np.unique(grid_cell))
    gkf_main = GroupKFold(n_splits=min(5, n_groups), shuffle=True, random_state=42)
    y_pred_cv = np.full(len(df), np.nan)
    fold_r2 = []
    for tr_idx, te_idx in gkf_main.split(X_full, y, groups=grid_cell):
        test_tree = cKDTree(xy_all[te_idx])
        near_test = test_tree.query_ball_tree(tree_all, r=BUFFER_FT)
        near_test_idx = set()
        for lst in near_test:
            near_test_idx.update(lst)
        tr_idx_buf = np.array([i for i in tr_idx if i not in near_test_idx])
        fold_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
        fold_pipe.fit(X_full.iloc[tr_idx_buf], y.iloc[tr_idx_buf], reg__sample_weight=sample_weight_full[tr_idx_buf])
        pred = fold_pipe.predict(X_full.iloc[te_idx])
        y_pred_cv[te_idx] = pred
        fold_r2.append(r2_score(y.iloc[te_idx], pred, sample_weight=sample_weight_full[te_idx]))
    cv_full = np.array(fold_r2)
    print(f"[{time.time()-t0:.0f}s] {label}: buffered PRIMARY CV R2={cv_full.mean():.4f} "
          f"(+/-{cv_full.std():.4f}), n={len(df)}, {n_groups} grid cells")

    results_full = df[["placekey", "borough", "location_name", "top_category"]].copy()
    results_full["expected_mixing_cv"] = y_pred_cv
    results_full["improbable_copresence_cv"] = df["income_mixing_score"].to_numpy() - y_pred_cv
    #placeholder columns: pool_persistence.py unconditionally drops these two (plain-multinomial
    #BOOTSTRAP 2/2 columns) before merging in the DM-corrected replacement -- this CV-only pass
    #never computes them (skipped on purpose, see module docstring), so NaN placeholders only.
    results_full["resid_vs_mean"] = np.nan
    results_full["resid_vs_var"] = np.nan
    return results_full, cv_full


def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


floor_summaries = []
for floor in FLOORS:
    floor_dir = os.path.join(FLOOR_OUTPUT_DIR, f"floor{floor}")
    os.makedirs(floor_dir, exist_ok=True)
    od_link = os.path.join(floor_dir, "overdispersion_estimate.pkl")
    if not os.path.exists(od_link):
        os.symlink(os.path.abspath(os.path.join(OUTPUT_DIR, "overdispersion_estimate.pkl")), od_link)

    for mm in PERIODS:
        model_ready_path = os.path.join(floor_dir, f"model_ready_{mm}.pkl")
        if not os.path.exists(model_ready_path):
            if floor > 50:
                #RAISING the floor: filter the existing floor=50 model_ready -- no rebuild needed,
                #every surviving venue has identical feature values (see
                #sensitivity_eligibility_floor.py's docstring for the full reasoning).
                df50 = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{mm}.pkl"))
                df50[df50["income_known_devices"] >= floor].to_pickle(model_ready_path)
            else:
                #LOWERING the floor: an actual rebuild via a patched build_features_<PERIOD>.py,
                #run once, output pointed straight at this floor's directory.
                build_script = os.path.join(SCRIPT_DIR, f"build_features_{mm}.py")
                run(f"OUTPUT_DIR={floor_dir} INCOME_KNOWN_DEVICE_THRESHOLD_OVERRIDE={floor} "
                    f"python3 {build_script}")
                # build_features_<PERIOD>.py writes model_ready_<PERIOD>.pkl under that floor's
                # threshold once INCOME_KNOWN_DEVICE_THRESHOLD_OVERRIDE patches its constant --
                # see sensitivity_eligibility_floor.py's docstring for the original one-off version
                # of this same rebuild.

        results_mixing_path = os.path.join(floor_dir, f"results_mixing_{mm}.pkl")
        if not os.path.exists(results_mixing_path):
            results_full, cv_full = buffered_primary_cv(model_ready_path, f"floor{floor}_{mm}")
            pd.to_pickle({"results_full": results_full, "cv_full": cv_full}, results_mixing_path)

        vs_var_path = os.path.join(floor_dir, f"visitor_sampling_variance_dm_{mm}.csv")
        if not os.path.exists(vs_var_path):
            run(f"MONTH={mm} OUTPUT_DIR={floor_dir} S_HAT={S_HAT} "
                f"python3 {os.path.join(SCRIPT_DIR, 'estimate_visitor_sampling_variance_dm.py')}")

        model_var_path = os.path.join(floor_dir, f"model_prediction_variance_{mm}.csv")
        if not os.path.exists(model_var_path):
            run(f"MONTH={mm} OUTPUT_DIR={floor_dir} "
                f"python3 {os.path.join(SCRIPT_DIR, 'estimate_model_prediction_variance.py')}")

    pooled_path = os.path.join(floor_dir, "pooled_persistence_2026.csv")
    if not os.path.exists(pooled_path):
        run(f"OUTPUT_DIR={floor_dir} python3 {os.path.join(SCRIPT_DIR, 'pool_persistence.py')}")

    pooled = pd.read_csv(pooled_path)
    n_family = int((pooled["n_months"] >= 2).sum())
    n_survivors = int(pooled["fdr_sig_05_hk"].sum())
    row = {"floor": floor, "n_venues": len(pooled), "n_family": n_family,
           "n_survivors": n_survivors, "survivor_rate_pct": 100 * n_survivors / n_family}
    pooled_idx = pooled.set_index("placekey")
    for label, pk in NAMED_PKS.items():
        if pk in pooled_idx.index:
            r = pooled_idx.loc[pk]
            row[f"{label}_n_months"] = int(r["n_months"])
            row[f"{label}_mu_hk"] = r["mu_hk"]
            row[f"{label}_q_bh_hk"] = r["q_bh_hk"]
            row[f"{label}_sig"] = bool(r["fdr_sig_05_hk"]) if not pd.isna(r["fdr_sig_05_hk"]) else None
        else:
            row[f"{label}_n_months"] = 0
            row[f"{label}_mu_hk"] = np.nan
            row[f"{label}_q_bh_hk"] = np.nan
            row[f"{label}_sig"] = None
    floor_summaries.append(row)
    print(f"floor={floor}: {len(pooled)} venues, {n_family} tested (n_months>=2), "
          f"{n_survivors} survive ADOPTED FDR ({row['survivor_rate_pct']:.2f}%)")

#reference row: the published floor=50, from the already-adopted RAW (pre-consolidation)
#pool_persistence.py output -- not rerun here, just read, since it's the production baseline.
pub = pd.read_csv(os.path.join(OUTPUT_DIR, "pooled_persistence_2026.csv"))
n_family_50 = int((pub["n_months"] >= 2).sum())
n_survivors_50 = int(pub["fdr_sig_05_hk"].sum())
row50 = {"floor": 50, "n_venues": len(pub), "n_family": n_family_50,
         "n_survivors": n_survivors_50, "survivor_rate_pct": 100 * n_survivors_50 / n_family_50}
pub_idx = pub.set_index("placekey")
for label, pk in NAMED_PKS.items():
    if pk in pub_idx.index:
        r = pub_idx.loc[pk]
        row50[f"{label}_n_months"] = int(r["n_months"])
        row50[f"{label}_mu_hk"] = r["mu_hk"]
        row50[f"{label}_q_bh_hk"] = r["q_bh_hk"]
        row50[f"{label}_sig"] = bool(r["fdr_sig_05_hk"]) if not pd.isna(r["fdr_sig_05_hk"]) else None
floor_summaries.append(row50)

summary = pd.DataFrame(floor_summaries).sort_values("floor")
summary.to_csv(os.path.join(FLOOR_OUTPUT_DIR, "sensitivity_eligibility_floor_fdr_results.csv"), index=False)
pd.set_option("display.width", 200)
print("\nFULL SUMMARY (floor=50 is the published baseline, read not rerun):")
print(summary.to_string(index=False))
print(f"\nsurvivor rate range across floors 30-150: {summary['survivor_rate_pct'].min():.2f}%-"
      f"{summary['survivor_rate_pct'].max():.2f}%")
print(f"saved {FLOOR_OUTPUT_DIR}/sensitivity_eligibility_floor_fdr_results.csv")
