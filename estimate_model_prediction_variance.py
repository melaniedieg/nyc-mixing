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

VARIABLE, LOW DRAW COUNT, FIXED (Note 12, external review): the original design resampled grid
CELLS with replacement into a per-replicate boot_df, then ran a fresh GroupKFold(5) WITHIN that
replicate's (variably-sized, variably-composed) resampled universe -- so a venue only got a
prediction that replicate if its own grid cell happened to be drawn into boot_df at all. Since an
individual cell has only a ~63.2% chance of appearing in any one bootstrap draw (standard
non-selection-probability math: 1-(1-1/n)^n -> 1-1/e as n grows), draw counts across the default
N_BOOT=15 replicates came out highly variable in practice on real data -- median 10, min 3, with
110 of ~27k July venues drawing fewer than 5 of the 15 replicates. A handful of venues even risked
n_draws<2 (model_pred_var undefined, silently falls back to resid_vs_var alone) if they were ever
unlucky across all N_BOOT draws, though that never actually happened on real data checked so far.

The fix: separate "which cells make it into training" (the thing that should bootstrap-vary) from
"which venues get a held-out prediction" (which should NOT depend on the luck of a cell being
resampled). The grid-cell-blocked, BUFFERED GroupKFold(5) split is now computed ONCE, up front,
over the FULL venue universe (same BUFFER_FT=1500 spatial buffer and the same
shuffle=True/random_state=42 fold assignment as fit_mixing_model_2026_0X.py's own PRIMARY CV --
see that script's docstring -- so this script's fold membership is reproducible the same way).
Every venue is in exactly one of the 5 folds every replicate, so every venue gets exactly one
genuinely held-out prediction every replicate: n_draws is now always exactly N_BOOT for every
venue, by construction.

The bootstrap-resampling signal moves into training instead: each replicate still draws grid
cells with replacement (same rng, same mechanism as before), but instead of physically duplicating
rows into a resampled boot_df, each row's sample_weight for that replicate's fold-fits is
multiplied by how many times ITS cell was drawn this replicate (0 if the cell wasn't drawn at
all). A cell drawn 0 times contributes weight 0 and is -- in effect, not in name -- excluded from
that replicate's training data, the same as if its rows had never been duplicated into boot_df;
a cell drawn twice gets 2x the influence, also the same as physical duplication would have given
it. RandomForestRegressor's internal bootstrap=True row-resampling respects sample_weight (rows
are drawn proportional to it), so this reproduces the original cell-level bootstrap's training-set
variability exactly, without ever touching which venues are eligible to be predicted.

Net effect: the variance recorded per venue is still "how much would this venue's predicted value
move under a different, equally legitimate bootstrap resample of the training cells" -- the same
quantity as before -- but now estimated from a FULL N_BOOT draws for every venue instead of a
Binomial(N_BOOT, ~0.632)-distributed, sometimes-as-low-as-3 subset of them. This also makes the
script faster: the spatial buffer exclusion set depends only on which rows are in which fold
(fixed) and all rows' fixed positions, so it is computed ONCE instead of being recomputed from
scratch, per replicate, against a resampled geometry, as before.

Method: resample grid CELLS (not venues) with replacement -- same spatial-block design as
sensitivity_spatial_block_bootstrap.py, which fixed the OTHER model-refit bootstrap defect this
same external review caught (fit_mixing_model_2026_0X.py's "model-refit stability" check; see
that script and the README bullet above this one). Each replicate refits the real grid-cell-
blocked, BUFFERED GroupKFold(5) (Note 7 fix; Note 12 fix above for how the split itself is now
fixed and the bootstrap moves into sample_weight) and every venue's out-of-fold prediction that
replicate goes straight into that venue's running list -- no averaging across duplicate cell
occurrences needed anymore, since cells are never physically duplicated. The variance of those
N_BOOT predictions per venue is the empirical model-refit/prediction-uncertainty estimate for that
venue-month, in the same units as resid_vs_var so the two can be summed directly.

Compute cost, stated plainly: this reruns the full grid-cell-blocked GroupKFold(5) N_BOOT times
(15 by default -- 5x the model fits of a single published CV run), ~10-15 minutes per month on
the data sizes seen so far (10k-30k venues) -- the Note 12 fix changes WHERE that cost goes
(no more per-replicate geometry/tree rebuild) but not materially how many RandomForest fits
happen, so this remains a REQUIRED step before pool_persistence.py/consolidate_fragments.py for
any new month, not an optional sensitivity check -- see README's "Run order per month" for where
it goes in the monthly pipeline.

Tracking EVERY venue instead of a reference-flagged subset costs nothing extra per replicate (the
bottleneck is the 5 RandomForest refits per replicate, not how many venues' predictions get read
off afterward), which is what makes running this for the full universe, every month, affordable
at all.

Input:  OUTPUT_DIR/model_ready_<MONTH>.pkl (same file fit_mixing_model_2026_0X.py reads)
Output: OUTPUT_DIR/model_prediction_variance_<MONTH>.csv -- one row per venue eligible that
        month: placekey, month, n_draws (now always exactly N_BOOT, kept as a column for
        continuity with pool_persistence.py and as a sanity check -- see Note 12 above for why
        this is no longer variable), and model_pred_var (no longer ever NaN on the grounds of a
        low draw count; see Note 12).
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

X_full = df[feats_num_full + feats_cat]
y_full = df["income_mixing_score"]
w_known = df["income_known_devices"].to_numpy(dtype=float)  #the pre-existing, non-bootstrap weight
groups_full = df["_grid_cell"].to_numpy()
all_pks = df["placekey"].to_numpy()
n_rows = len(df)

#Note 12: the grid-cell-blocked, buffered GroupKFold(5) split is now fixed ONCE over the full
#venue universe, with the same shuffle=True/random_state=42 fold assignment as
#fit_mixing_model_2026_0X.py's own PRIMARY CV, so every venue lands in exactly one fold every
#replicate -- see module docstring.
_xy_all = np.column_stack([_gdf.geometry.x.to_numpy(), _gdf.geometry.y.to_numpy()])
_tree_all = cKDTree(_xy_all)
n_splits = min(5, n_cells)
gkf_main = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_of_row = np.full(n_rows, -1, dtype=int)
tr_idx_buf_by_fold = []
te_idx_by_fold = []
for fold_i, (tr_idx, te_idx) in enumerate(gkf_main.split(X_full, y_full, groups=groups_full)):
    fold_of_row[te_idx] = fold_i
    #Note 7 fix, computed once (fixed geometry/fixed test set -> fixed buffer exclusion set):
    #same buffered-CV exclusion as fit_mixing_model_2026_0X.py's PRIMARY CV -- excludes any
    #training-candidate venue within BUFFER_FT of ANY test-fold venue, not just the test cell's
    #own venues.
    _test_tree = cKDTree(_xy_all[te_idx])
    _near_test = _test_tree.query_ball_tree(_tree_all, r=BUFFER_FT)
    _near_test_idx = set()
    for _lst in _near_test:
        _near_test_idx.update(_lst)
    tr_idx_buf = np.array([i for i in tr_idx if i not in _near_test_idx])
    tr_idx_buf_by_fold.append(tr_idx_buf)
    te_idx_by_fold.append(te_idx)
    print(f"[{time.time()-t0:.0f}s] fold {fold_i}: {len(te_idx)} test venues, "
          f"{len(tr_idx_buf)}/{len(tr_idx)} train venues survive the {BUFFER_FT}ft buffer")

assert (fold_of_row >= 0).all(), "every row should land in exactly one fold"

rng = np.random.RandomState(7)
preds_matrix = np.full((N_BOOT, n_rows), np.nan)

for b in range(N_BOOT):
    #Note 12: same cell-level bootstrap draw as before, but it now feeds a per-row sample_weight
    #multiplier instead of physically resampling rows into a replicate-specific boot_df. A cell
    #drawn 0 times this replicate gets weight 0 (in effect excluded from training, same as if its
    #rows were never duplicated into boot_df); a cell drawn k times gets k x its normal weight.
    sampled_cells = rng.choice(unique_cells, size=n_cells, replace=True)
    cells_drawn, draw_counts = np.unique(sampled_cells, return_counts=True)
    draw_count_by_cell = dict(zip(cells_drawn, draw_counts))
    row_draw_count = np.array([draw_count_by_cell.get(c, 0) for c in groups_full], dtype=float)
    row_weight = w_known * row_draw_count

    y_pred_rep = np.full(n_rows, np.nan)
    for fold_i in range(n_splits):
        tr_idx_buf = tr_idx_buf_by_fold[fold_i]
        te_idx = te_idx_by_fold[fold_i]
        sw = row_weight[tr_idx_buf]
        pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=b, n_jobs=-1))])
        pipe.fit(X_full.iloc[tr_idx_buf], y_full.iloc[tr_idx_buf], reg__sample_weight=sw)
        y_pred_rep[te_idx] = pipe.predict(X_full.iloc[te_idx])

    assert not np.isnan(y_pred_rep).any(), "every venue should get a prediction every replicate"
    preds_matrix[b, :] = y_pred_rep

    print(f"[{time.time()-t0:.0f}s] {MONTH} replicate {b+1}/{N_BOOT} done "
          f"({len(cells_drawn)}/{n_cells} distinct cells drawn >=1x, "
          f"{n_cells - len(cells_drawn)} cells drawn 0x)")

model_pred_mean = preds_matrix.mean(axis=0)
model_pred_var = preds_matrix.var(axis=0, ddof=1)
out = pd.DataFrame({
    "placekey": all_pks,
    "month": MONTH,
    "n_draws": N_BOOT,  #Note 12: always exactly N_BOOT now, by construction -- see module docstring
    "model_pred_mean": model_pred_mean,
    "model_pred_var": model_pred_var,
})
out.to_csv(os.path.join(OUTPUT_DIR, f"model_prediction_variance_{MONTH}.csv"), index=False)
print(f"\n[{time.time()-t0:.0f}s] {MONTH}: saved {len(out)} venues, "
      f"{out['model_pred_var'].notna().sum()} with >=2 draws "
      f"(median model_pred_var={out['model_pred_var'].median():.6f})")
print(f"saved {OUTPUT_DIR}/model_prediction_variance_{MONTH}.csv")
