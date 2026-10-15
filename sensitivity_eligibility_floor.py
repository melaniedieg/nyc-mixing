"""
sensitivity_eligibility_floor.py

Point 6 (sensitivity analyses), part 2: stress-tests INCOME_KNOWN_DEVICE_THRESHOLD=50, the
minimum number of income-known visiting devices a venue needs to be eligible at all
(build_features_2026_0X.py). The number was chosen as "low enough to keep a usable number of
venues, high enough that a single quintile isn't being inferred from a handful of devices" --
this script checks whether the model's out-of-sample fit is actually sensitive to that choice,
in both directions, for April 2026.

Two different mechanics, because of how build_features_2026_0X.py is structured: it computes
every spatial/context feature (density, subway, tourist, chain, local-income) on the FULL POI
universe BEFORE the eligibility filter is applied as a final dataframe subset. That means:

  - RAISING the floor (75, 100, 150) is just filtering MORE rows out of the existing
    floor=50 model_ready_2026_04.pkl -- no rebuild needed, and every surviving venue has
    IDENTICAL feature values to the floor=50 run (same model inputs, fewer eligible rows).
  - LOWERING the floor (30) needs an actual rebuild, because venues between 30 and 49 devices
    were never in the floor=50 dataframe at all. Ran separately via a patched copy of
    build_features_2026_04.py with INCOME_KNOWN_DEVICE_THRESHOLD=30 (41s, cheap, since the
    expensive spatial-feature computation doesn't change).

Both directions reuse the EXACT model spec and CV scheme from fit_mixing_model_2026_04.py
(same features, same RandomForest hyperparameters, same income_known_devices sample weighting,
same 2000ft grid-cell-blocked GroupKFold(5)) so the resulting R^2 values are directly comparable
to the published floor=50 baseline (mean 0.6927, fold range 0.635-0.735) -- this script does not
invent a different, easier-to-pass check.

Input:
  FLOOR50_PATH   -- the real, already-built floor=50 model_ready_2026_04.pkl
  FLOOR30_PATH   -- a floor=30 rebuild (patched INCOME_KNOWN_DEVICE_THRESHOLD, otherwise
                    identical build_features_2026_04.py)
Output: sensitivity_eligibility_floor_results.csv -- one row per floor value tested.
"""
import os
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

FLOOR50_PATH = os.environ.get("FLOOR50_PATH", "./output/model_ready_2026_04.pkl")
FLOOR30_PATH = os.environ.get("FLOOR30_PATH", "./output/model_ready_2026_04_floor30.pkl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
GRID_CELL_FT = 2000

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


def grid_cells(df):
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=2263)
    gx = (gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
    gy = (gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
    return (gx + "_" + gy).values


def run_cv(df, label, t0):
    X = df[feats_num_full + feats_cat]
    y = df["income_mixing_score"]
    w = df["income_known_devices"].values
    pre = ColumnTransformer([("num", StandardScaler(), feats_num_full),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])
    cells = grid_cells(df)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2 = []
    for tr_idx, te_idx in gkf.split(X, y, groups=cells):
        pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx], reg__sample_weight=w[tr_idx])
        pred = pipe.predict(X.iloc[te_idx])
        fold_r2.append(r2_score(y.iloc[te_idx], pred, sample_weight=w[te_idx]))
    fold_r2 = np.array(fold_r2)
    print(f"[{time.time()-t0:.0f}s] {label}: n={len(df)}, CV R2={fold_r2.mean():.4f} "
          f"(+/-{fold_r2.std():.4f}), folds={np.round(fold_r2, 4).tolist()}")
    return fold_r2


t0 = time.time()
results = []

#--- direction 1: RAISE the floor by subsetting the existing floor=50 dataframe (same rows,
#same features, just a tighter final filter -- no rebuild) ---
df50 = pd.read_pickle(FLOOR50_PATH)
print(f"[{time.time()-t0:.0f}s] loaded floor=50 base: n={len(df50)}")
for floor in (50, 75, 100, 150):
    sub = df50[df50["income_known_devices"] >= floor].copy()
    fold_r2 = run_cv(sub, f"floor={floor} (subset of floor=50 build)", t0)
    results.append({"floor": floor, "source": "subset_of_floor50_build", "n_venues": len(sub),
                     "cv_r2_mean": fold_r2.mean(), "cv_r2_std": fold_r2.std(),
                     "cv_r2_min": fold_r2.min(), "cv_r2_max": fold_r2.max()})

#--- direction 2: LOWER the floor using the separately-rebuilt floor=30 dataframe (new venues
#between 30-49 devices weren't in the floor=50 build at all) ---
df30 = pd.read_pickle(FLOOR30_PATH)
print(f"[{time.time()-t0:.0f}s] loaded floor=30 rebuild: n={len(df30)}")
fold_r2_30 = run_cv(df30, "floor=30 (separate rebuild)", t0)
results.append({"floor": 30, "source": "floor30_rebuild", "n_venues": len(df30),
                 "cv_r2_mean": fold_r2_30.mean(), "cv_r2_std": fold_r2_30.std(),
                 "cv_r2_min": fold_r2_30.min(), "cv_r2_max": fold_r2_30.max()})

res_df = pd.DataFrame(results).sort_values("floor").reset_index(drop=True)
res_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_eligibility_floor_results.csv"), index=False)
pd.set_option("display.width", 160)
print(f"\n[{time.time()-t0:.0f}s] FULL RESULTS:")
print(res_df.to_string(index=False))
print(f"\n[{time.time()-t0:.0f}s] TOTAL DONE")
