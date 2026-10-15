"""
sensitivity_neighborhood_diversity.py

Point 6 (sensitivity analyses), part 3: tests a specific omitted-variable concern raised in
external review -- the contextual baseline model (fit_mixing_model_2026_0X.py) conditions on a
block group's MEAN income (income_proxy_v2) and racial composition, but not on how SPREAD OUT
that block group's income distribution is. Two neighborhoods can have identical mean income and
very different income distributions; if income spread itself predicts mixing, a venue's residual
could partly reflect omitted neighborhood diversity rather than anything about the venue itself --
exactly the kind of confound the model is supposed to net out.

The predictor already exists and didn't need building: `census_new.pkl` (the same file
build_features_2026_0X.py already reads for income_proxy_v2) has had
`neighborhood_income_diversity_v2` in it the whole time -- normalized Shannon entropy (0-1) of a
block group's 16 ACS household-income brackets, computed in the same cell as income_proxy_v2 in
income_mixing_model.ipynb. It was explored in the notebook (cells 15-20) but never carried into
build_features_2026_0X.py's column selection, so no production model has ever seen it. This script
adds it back in and checks whether it matters.

Method: load a month's already-built model_ready_2026_0X.pkl, merge in
neighborhood_income_diversity_v2 by `geoid` (no rebuild needed -- it's a static per-block-group
value, same join key already used for income_proxy_v2), and rerun the EXACT fit_mixing_model
spec and CV scheme (same features plus this one, same RandomForest hyperparameters, same
income_known_devices sample weighting, same grid-cell-blocked GroupKFold(5)) twice: once without
the new feature (reproducing the published baseline) and once with it added. Compares (a) CV R²,
(b) the new feature's RF importance against the other numeric features, and (c) the
spatial-CV-residual z-score (within borough, same convention as the published results) for the
four headline named venues, since a predictor that matters would be expected to move at least one
of them.

Tested so far: July 2026 only. This is a first/representative check, not an exhaustive one --
rerunning across more periods (the MONTH env var below) would strengthen it further if the result
ever looked borderline; it didn't here (see README "Known limitations" for the numbers).

Input:  OUTPUT_DIR/model_ready_2026_<MM>.pkl (already-built, from build_features_2026_<MM>.py)
        DATA_DIR/census_new.pkl (already has neighborhood_income_diversity_v2 -- see above)
Output: sensitivity_neighborhood_diversity_results.csv -- one row per model variant (CV R² stats
        plus the new feature's RF importance), and
        sensitivity_neighborhood_diversity_named_venues.csv -- one row per headline venue, with
        its residual z-score under each variant.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

MONTH = os.environ.get("MONTH", "2026_07")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
GRID_CELL_FT = 2000
NEW_FEAT = "neighborhood_income_diversity_v2"

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

# The four headline venues this check is specifically meant to stress-test (see README
# "Validation checks already run" / "Known limitations" for why these four).
NAMED_VENUES = {
    "224-224@627-s6c-m8v": "Amenazel Organic",
    "22v-223@627-s84-bhq": "Chens Fresh Food Market",
    "zzy-222@627-rwq-m49": "Hunts Point Floating Pier",
    "zzy-222@627-wgg-sbk": "Jacob Riis Park",
    "zzy-228@627-tmh-49z": "The People's Beach at Jacob Riis Park",
    "zzy-22f@627-tmh-49z": "Jacob Riis Beach (Rockaways fragment)",
}


def grid_cells(df):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=2263)
    gx = (gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
    gy = (gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
    return (gx + "_" + gy).values


df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{MONTH}.pkl"))
census = pd.read_pickle(os.path.join(DATA_DIR, "census_new.pkl"))[["geoid", NEW_FEAT]]
n_before = len(df)
df = df.merge(census, on="geoid", how="left")
missing = df[NEW_FEAT].isna().mean()
print(f"{MONTH}: {n_before} venues; {NEW_FEAT} missing for {missing:.2%} -- "
      f"dropping those so both variants are compared on the identical row set")
df = df[df[NEW_FEAT].notna()].reset_index(drop=True)

y = df["income_mixing_score"]
sample_weight = df["income_known_devices"].values
grid_cell = grid_cells(df)


def run_cv(feat_list, label):
    X = df[feat_list + feats_cat]
    pre = ColumnTransformer([("num", StandardScaler(), feat_list),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = np.full(len(df), np.nan)
    fold_r2 = []
    for tr_idx, te_idx in gkf.split(X, y, groups=grid_cell):
        pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx], reg__sample_weight=sample_weight[tr_idx])
        pred = pipe.predict(X.iloc[te_idx])
        y_pred_cv[te_idx] = pred
        fold_r2.append(r2_score(y.iloc[te_idx], pred, sample_weight=sample_weight[te_idx]))
    fold_r2 = np.array(fold_r2)
    print(f"{label}: CV R2={fold_r2.mean():.4f} (+/-{fold_r2.std():.4f}), "
          f"folds={np.round(fold_r2, 4).tolist()}")
    return y_pred_cv, fold_r2


pred_baseline, r2_baseline = run_cv(feats_num_full, "BASELINE (published feature set)")
pred_plus, r2_plus = run_cv(feats_num_full + [NEW_FEAT], f"PLUS {NEW_FEAT}")

df["resid_baseline"] = y.values - pred_baseline
df["resid_plus"] = y.values - pred_plus
for bor in df["borough"].unique():
    m = df["borough"] == bor
    for col in ("resid_baseline", "resid_plus"):
        r = df.loc[m, col]
        df.loc[m, col + "_z"] = (r - r.mean()) / r.std()

# full-data (non-CV) RF, just to rank the new feature's importance against the others
X_full = df[feats_num_full + [NEW_FEAT] + feats_cat]
pre_full = ColumnTransformer([("num", StandardScaler(), feats_num_full + [NEW_FEAT]),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])
pipe_full = Pipeline([("pre", pre_full), ("reg", RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
pipe_full.fit(X_full, y, reg__sample_weight=sample_weight)
numeric_feats = feats_num_full + [NEW_FEAT]
importances = pipe_full.named_steps["reg"].feature_importances_[:len(numeric_feats)]
imp_df = pd.DataFrame({"feature": numeric_feats, "importance": importances}).sort_values(
    "importance", ascending=False).reset_index(drop=True)
new_feat_rank = int(imp_df.index[imp_df["feature"] == NEW_FEAT][0]) + 1
print(f"\n{NEW_FEAT} RF importance rank: {new_feat_rank} of {len(numeric_feats)} numeric features")
print(imp_df.to_string(index=False))

results = pd.DataFrame([
    {"variant": "baseline", "n_venues": len(df), "cv_r2_mean": r2_baseline.mean(),
     "cv_r2_std": r2_baseline.std(), "cv_r2_min": r2_baseline.min(), "cv_r2_max": r2_baseline.max(),
     f"{NEW_FEAT}_importance_rank": np.nan},
    {"variant": f"plus_{NEW_FEAT}", "n_venues": len(df), "cv_r2_mean": r2_plus.mean(),
     "cv_r2_std": r2_plus.std(), "cv_r2_min": r2_plus.min(), "cv_r2_max": r2_plus.max(),
     f"{NEW_FEAT}_importance_rank": new_feat_rank},
])
results.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_neighborhood_diversity_results.csv"), index=False)

named_rows = []
print("\nnamed venues, residual z-score before vs after adding the diversity predictor:")
for pk, name in NAMED_VENUES.items():
    row = df[df["placekey"] == pk]
    if len(row) == 0:
        print(f"  {name}: not eligible in {MONTH}")
        continue
    r = row.iloc[0]
    delta = r["resid_plus_z"] - r["resid_baseline_z"]
    print(f"  {name}: baseline_z={r['resid_baseline_z']:.4f}  plus_diversity_z={r['resid_plus_z']:.4f}  "
          f"delta={delta:+.4f}")
    named_rows.append({"placekey": pk, "location_name": name, "month": MONTH,
                        "resid_z_baseline": r["resid_baseline_z"], "resid_z_plus_diversity": r["resid_plus_z"],
                        "delta": delta})
pd.DataFrame(named_rows).to_csv(
    os.path.join(OUTPUT_DIR, "sensitivity_neighborhood_diversity_named_venues.csv"), index=False)

print(f"\nsaved {OUTPUT_DIR}/sensitivity_neighborhood_diversity_results.csv and "
      f"sensitivity_neighborhood_diversity_named_venues.csv")
