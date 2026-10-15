"""
fit_balance_model_2026_07.py

Robustness check, NOT independent confirmation: fits the same feature set and RandomForest
spec against `balance_score` (a second, related measure of how evenly a venue's visitors are
spread across income quintiles) instead of `income_mixing_score`. An earlier version of this
docstring said a venue counted as a "doubly-robust outlier" if it was also extreme here -- that
language is wrong and has been removed. Checked directly: among venues flagged by
`fit_mixing_model_2026_07.py`, this model's residual z-score correlates with the
income_mixing_score model's residual z-score at ~0.98 (computed below, if that month's
fit_mixing_model output is present). That's not independent evidence -- the two targets are
overwhelmingly tracking the same underlying signal -- so this script does not implement its own
borough-proportional flagging rule; it only fits the model and reports the correlation, which is
the actual cross-metric check. See README.md's Known limitations for the full caveat.

CV here is grid-cell-blocked GroupKFold (same ~2000ft/610m cells as
fit_mixing_model_2026_07.py), not a random row-level KFold -- z_balance feeds
cross_metric_corr, the actual cross-metric check, so it should not benefit from the
same spatial-leakage inflation the mixing-model CV had before that fix.

Input:  OUTPUT_DIR/model_ready_2026_07.pkl (from build_features_2026_07.py)
        OUTPUT_DIR/results_mixing_2026_07.pkl (from fit_mixing_model_2026_07.py, optional --
        enables the correlation check; the fit itself runs without it)
Output: OUTPUT_DIR/results_balance_2026_07.pkl
        {"results_bal": ..., "cv_bal": ..., "cv_bal_random": ..., "cross_metric_corr": float or None}
"""
import os
import pandas as pd, numpy as np, time
import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score

GRID_CELL_FT = 2000  #~610m -- same spatial block size as fit_mixing_model_2026_{mm}.py

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.width", 160)
t0 = time.time()

df = pd.read_pickle(os.path.join(OUTPUT_DIR, "model_ready_2026_07.pkl"))

feats_num_race = ["log_devices", "income_proxy_v2", "log_population",
                   "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_nh", "racial_diversity_index"]
feats_num_full = feats_num_race + [
    "log_poi_density_250m", "log_poi_density_500m", "log_same_category_density_250m",
    "log_dist_nearest_subway_m", "log_subway_count_500m", "log_subway_count_800m",
    "log_tourist_density_500m", "log_tourist_density_800m",
    "log_chain_density_500m", "log_chain_density_800m",
    #Spatially-smoothed local income context (see build_features_2026_07.py) -- unlike
    #income_proxy_v2, which comes from one administrative CBG, these describe the average
    #income_proxy_v2 of nearby venues' own assigned CBGs within 250m/500m, so a corridor that
    #straddles or draws from several CBGs isn't reduced to just the one it happens to sit in.
    "local_income_proxy_250m", "local_income_proxy_500m",
]
feats_cat = ["top_category", "borough"]
y_bal = df["balance_score"]

X_full = df[feats_num_full + feats_cat]
pre = ColumnTransformer([("num", StandardScaler(), feats_num_full), ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])
rf_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])

sample_weight_full = df["income_known_devices"].values

_gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
).to_crs(epsg=2263)
_gx = (_gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
_gy = (_gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
grid_cell = (_gx + "_" + _gy).values

#PRIMARY: grid-cell-blocked GroupKFold. Feeds resid_bal / z_balance / cross_metric_corr below.
gkf_main = GroupKFold(n_splits=5, shuffle=True, random_state=42)
pred_bal = np.full(len(df), np.nan)
fold_r2 = []
for tr_idx, te_idx in gkf_main.split(X_full, y_bal, groups=grid_cell):
    fold_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
    fold_pipe.fit(X_full.iloc[tr_idx], y_bal.iloc[tr_idx], reg__sample_weight=sample_weight_full[tr_idx])
    p = fold_pipe.predict(X_full.iloc[te_idx])
    pred_bal[te_idx] = p
    fold_r2.append(r2_score(y_bal.iloc[te_idx], p, sample_weight=sample_weight_full[te_idx]))
cv_bal = np.array(fold_r2)
print(f"[{time.time()-t0:.0f}s] PRIMARY (grid-cell GroupKFold) BALANCE-TARGET CV R2={cv_bal.mean():.4f} "
      f"(+/-{cv_bal.std():.4f}) [device-count weighted, spatially blocked]")

#SECONDARY, reported for comparison only -- not used for resid_bal/z_balance below.
kf_random = KFold(5, shuffle=True, random_state=42)
fold_r2_random = []
for tr_idx, te_idx in kf_random.split(X_full):
    fold_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
    fold_pipe.fit(X_full.iloc[tr_idx], y_bal.iloc[tr_idx], reg__sample_weight=sample_weight_full[tr_idx])
    p = fold_pipe.predict(X_full.iloc[te_idx])
    fold_r2_random.append(r2_score(y_bal.iloc[te_idx], p, sample_weight=sample_weight_full[te_idx]))
cv_bal_random = np.array(fold_r2_random)
print(f"[{time.time()-t0:.0f}s] SECONDARY (random row-level KFold, comparison only) "
      f"BALANCE-TARGET CV R2={cv_bal_random.mean():.4f} (+/-{cv_bal_random.std():.4f})")

resid_bal = y_bal.values - pred_bal
borough = df["borough"].values

def zscore_within_borough(resid, borough):
    z = np.empty_like(resid)
    for bor in np.unique(borough):
        mask = borough == bor
        r = resid[mask]
        z[mask] = (r - r.mean()) / r.std()
    return z

z_bal = zscore_within_borough(resid_bal, borough)
results_bal = df[["placekey", "location_name", "borough", "top_category", "balance_score"]].copy()
results_bal["expected_balance_cv"] = pred_bal
results_bal["z_balance"] = z_bal

#---- the actual cross-metric check: correlate this model's residual z with the mixing
#model's residual z, among venues the mixing model flagged. This is what "cross-metric
#confirmed" means in practice -- see the docstring above and README's Known limitations for
#why it isn't treated as an independent bar. ----
mixing_path = os.path.join(OUTPUT_DIR, "results_mixing_2026_07.pkl")
cross_metric_corr = None
if os.path.exists(mixing_path):
    mixing = pd.read_pickle(mixing_path)
    flagged = pd.concat([mixing["stable_top"], mixing["stable_bottom"]])[["placekey", "improbable_copresence_z"]]
    merged = flagged.merge(results_bal[["placekey", "z_balance"]], on="placekey", how="inner")
    cross_metric_corr = merged["improbable_copresence_z"].corr(merged["z_balance"])
    print(f"[{time.time()-t0:.0f}s] correlation between income_mixing_score residual z and "
          f"balance_score residual z, among {len(merged)} flagged venues: r={cross_metric_corr:.3f}")
else:
    print(f"[{time.time()-t0:.0f}s] {mixing_path} not found -- run fit_mixing_model_2026_07.py "
          f"first to compute the cross-metric correlation; saving the fit without it")

pd.to_pickle({"results_bal": results_bal, "cv_bal": cv_bal, "cv_bal_random": cv_bal_random,
              "cross_metric_corr": cross_metric_corr},
             os.path.join(OUTPUT_DIR, "results_balance_2026_07.pkl"))
print(f"[{time.time()-t0:.0f}s] DONE")
