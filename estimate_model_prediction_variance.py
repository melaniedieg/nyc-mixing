"""
estimate_model_prediction_variance.py

Point 6 (sensitivity analyses), part 5 -- promoted from a one-off check to a required monthly
pipeline step after external review: pool_persistence.py's/consolidate_fragments.py's pooling
weight used to be resid_vs_var (visitor-sampling uncertainty) ALONE, which holds the fitted
RandomForest's prediction (expected_mixing_cv) completely fixed and so never captures how much
that prediction itself would move under a different, equally legitimate spatially-blocked
training sample. Reviewed and quantified directly (see README "Known limitations" -- Model
prediction variance): for 31% of a test sample of reference-flagged venues, this omitted
model-refit variance was actually BIGGER than the sampling variance already in the published CI,
not a rounding-error-sized gap.

This script estimates that model-refit/prediction variance for EVERY eligible venue in one
month, so pool_persistence.py can combine it with resid_vs_var into one inverse-variance weight
(var_combined = resid_vs_var + model_pred_var) instead of using resid_vs_var alone -- this is now
the ADOPTED pooling variance; see that script's docstring.

Method: resample grid CELLS (not venues) with replacement -- same spatial-block design as
sensitivity_spatial_block_bootstrap.py, which fixed the OTHER model-refit bootstrap defect this
same external review caught (fit_mixing_model_2026_0X.py's "model-refit stability" check; see
that script and the README bullet above this one). Each replicate reruns the real grid-cell-
blocked, BUFFERED GroupKFold(5) (Note 7 fix -- same BUFFER_FT=1500 spatial buffer as
fit_mixing_model_2026_0X.py's own PRIMARY CV; see that script's docstring and the README's "No
spatial buffer between CV folds" for the full derivation) so every out-of-fold prediction stays
genuinely held out, then records every venue's predicted value that replicate (averaged across a
cell's occurrences if its cell was drawn more than once). The variance of those predictions
across replicates is the empirical model-refit/prediction-uncertainty estimate for that
venue-month, in the same units as resid_vs_var so the two can be summed directly.

Compute cost, stated plainly: this reruns the full grid-cell-blocked GroupKFold(5) N_BOOT times
(15 by default -- 5x the model fits of a single published CV run), ~10-15 minutes per month on
the data sizes seen so far (10k-30k venues). This is now a REQUIRED step before pool_persistence.py
/consolidate_fragments.py for any new month, not an optional sensitivity check -- see README's
"Run order per month" for where it goes in the monthly pipeline.

Tracking EVERY venue instead of a reference-flagged subset costs nothing extra per replicate (the
bottleneck is the 5 RandomForest refits per replicate, not how many venues' predictions get read
off afterward), which is what makes running this for the full universe, every month, affordable
at all.

Input:  OUTPUT_DIR/model_ready_<MONTH>.pkl (same file fit_mixing_model_2026_0X.py reads)
Output: OUTPUT_DIR/model_prediction_variance_<MONTH>.csv -- one row per venue eligible that
        month: placekey, month, n_draws (how many of the N_BOOT replicates actually drew that
        venue's grid cell -- a block bootstrap doesn't draw every cell every time), and
        model_pred_var (NaN if n_draws<2; pool_persistence.py falls back to resid_vs_var alone
        for those few venue-months, which in practice has been a vanishingly small share -- see
        the per-month "saved N venues, M with >=2 draws" print below).
"""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

MONTH = os.environ["MONTH"]  #e.g. "2026_07" or "2025_12" -- no default, this always runs per-month
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
GRID_CELL_FT = 2000  #same spatial block size as fit_mixing_model_2026_0X.py's own CV
BUFFER_FT = 1500  #Note 7 fix -- same spatial buffer as fit_mixing_model_2026_0X.py's PRIMARY CV
N_BOOT = int(os.environ.get("N_BOOT", "15"))

t0 = time.time()

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

df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{MONTH}.pkl"))
print(f"[{time.time()-t0:.0f}s] {MONTH}: {len(df)} venues, N_BOOT={N_BOOT}")

_gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
).to_crs(epsg=2263)
_gx = (_gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
_gy = (_gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
df = df.copy()
df["_grid_cell"] = (_gx + "_" + _gy).values
unique_cells = np.unique(df["_grid_cell"].values)
n_cells = len(unique_cells)
print(f"[{time.time()-t0:.0f}s] {n_cells} distinct {GRID_CELL_FT}ft grid cells")

pre = ColumnTransformer([("num", StandardScaler(), feats_num_full),
                          ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])

rng = np.random.RandomState(7)
cells_by_group = {cell: sub for cell, sub in df.groupby("_grid_cell")}
all_pks = df["placekey"].to_numpy()
preds_by_venue = {pk: [] for pk in all_pks}

for b in range(N_BOOT):
    sampled_cells = rng.choice(unique_cells, size=n_cells, replace=True)
    boot_df = pd.concat([cells_by_group[c] for c in sampled_cells], ignore_index=True)

    Xb = boot_df[feats_num_full + feats_cat]
    yb = boot_df["income_mixing_score"]
    wb = boot_df["income_known_devices"].values
    groups_b = boot_df["_grid_cell"].values  #original cell id -- every draw of the same cell
                                              #stays together in one fold (see
                                              #sensitivity_spatial_block_bootstrap.py docstring
                                              #for why this matters)

    #Note 7 fix: same spatial buffer as fit_mixing_model_2026_0X.py's PRIMARY CV, applied fresh
    #to this replicate's own (resampled, possibly duplicated) rows -- a cell drawn more than
    #once still needs its buffer computed against this replicate's actual row positions.
    _bgdf = gpd.GeoDataFrame(
        boot_df, geometry=gpd.points_from_xy(boot_df["longitude"], boot_df["latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=2263)
    _bxy = np.column_stack([_bgdf.geometry.x.to_numpy(), _bgdf.geometry.y.to_numpy()])
    _btree = cKDTree(_bxy)

    n_groups_b = len(np.unique(groups_b))
    gkf = GroupKFold(n_splits=min(5, n_groups_b))
    y_pred_boot = np.full(len(boot_df), np.nan)
    for tr_idx, te_idx in gkf.split(Xb, yb, groups=groups_b):
        _test_tree = cKDTree(_bxy[te_idx])
        _near_test = _test_tree.query_ball_tree(_btree, r=BUFFER_FT)
        _near_test_idx = set()
        for _lst in _near_test:
            _near_test_idx.update(_lst)
        tr_idx_buf = np.array([i for i in tr_idx if i not in _near_test_idx])
        pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=b, n_jobs=-1))])
        pipe.fit(Xb.iloc[tr_idx_buf], yb.iloc[tr_idx_buf], reg__sample_weight=wb[tr_idx_buf])
        y_pred_boot[te_idx] = pipe.predict(Xb.iloc[te_idx])

    boot_df = boot_df.assign(_pred=y_pred_boot)
    pred_this_rep = boot_df.groupby("placekey")["_pred"].mean()
    for pk, val in pred_this_rep.items():
        preds_by_venue[pk].append(val)

    print(f"[{time.time()-t0:.0f}s] {MONTH} replicate {b+1}/{N_BOOT} done "
          f"({len(boot_df)} rows, {n_groups_b} distinct cells present)")

rows = []
for pk, preds in preds_by_venue.items():
    preds = np.array(preds)
    rows.append({"placekey": pk, "month": MONTH, "n_draws": len(preds),
                 "model_pred_mean": preds.mean() if len(preds) else np.nan,
                 "model_pred_var": preds.var(ddof=1) if len(preds) >= 2 else np.nan})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUTPUT_DIR, f"model_prediction_variance_{MONTH}.csv"), index=False)
print(f"\n[{time.time()-t0:.0f}s] {MONTH}: saved {len(out)} venues, "
      f"{out['model_pred_var'].notna().sum()} with >=2 draws "
      f"(median model_pred_var={out['model_pred_var'].median():.6f})")
print(f"saved {OUTPUT_DIR}/model_prediction_variance_{MONTH}.csv")
