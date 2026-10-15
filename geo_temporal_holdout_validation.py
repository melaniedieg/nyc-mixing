"""
geo_temporal_holdout_validation.py

The existing grid-cell-blocked GroupKFold in fit_mixing_model_2026_0X.py tests whether the
model generalizes across SPACE within one month. It never tests whether the model generalizes
across TIME -- every month is independently refit and evaluated only against its own held-out
venues from that same month. This script runs that missing check.

A first attempt (train on three months pooled, test on the fourth -- see git history / repo
comments if this ever needs re-deriving) gave a misleadingly strong result (mean weighted
R^2=0.78, HIGHER than the within-month spatial CV's 0.69-0.73) because of venue overlap: 92.4%
of April's venues also appear in at least one other month, and 21,557 of 33,767 distinct venues
across all four months (64%) appear in all four. Context features (income_proxy_v2, race,
density, subway distance, local income) are derived from a venue's fixed physical location, so
they barely change month to month -- meaning the model had effectively already seen most
"held-out" venues' context during training, just paired with a different month's realized score.
That's the exact same kind of leakage that justified grid-cell-blocking the within-month CV in
the first place, just showing up across months instead of across space.

This version blocks on BOTH axes at once: a venue is assigned to exactly one of 5 grid-cell-based
groups (reusing the same ~2000ft spatial blocking already used for the within-month CV, but now
applied to the POOLED set of all distinct venues across all 4 months, not per month), and for
each held-out month, the model is trained ONLY on other months' rows for venues in the other 4
groups, then tested on the held-out month's rows for venues in the held-out group -- so a test
venue was never seen in ANY training month, and nearby venues stay together exactly like the
existing within-month check already does. This is the honest "doubly held out" version: new
venues AND a new month, simultaneously -- and it comes back close to the within-month spatial-CV
numbers (mean weighted R^2=0.7263, range 0.698-0.743 across the four held-out months, vs.
0.69-0.73 within-month), which is the reassuring result: no evidence the model is overfit to a
specific month, once venue overlap is actually blocked rather than ignored.

Input: model_ready_2026_0X.pkl for all four months (set MODEL_READY_DIR, default ./output,
matching the OUTPUT_DIR each month's build_features_2026_0X.py writes to).
Output: geo_temporal_holdout_results.csv -- one row per held-out month.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

MODEL_READY_DIR = os.environ.get("MODEL_READY_DIR", "./output")
MODEL_READY_PATHS = {
    mm: os.path.join(MODEL_READY_DIR, f"model_ready_2026_{mm}.pkl") for mm in ("04", "05", "06", "07")
}
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
ALL_COLS = feats_num_full + feats_cat + ["income_mixing_score", "income_known_devices",
                                          "placekey", "latitude", "longitude"]

frames = {}
for mm, path in MODEL_READY_PATHS.items():
    d = pd.read_pickle(path)
    d = d[ALL_COLS].dropna(subset=feats_num_full + feats_cat + ["income_mixing_score"]).copy()
    d["month"] = mm
    frames[mm] = d

#one lat/long per distinct placekey (first occurrence across months -- same physical venue,
#coordinates don't move between months) -> grid cell -> one of 5 venue-level groups
all_rows = pd.concat(frames.values(), ignore_index=True)
venue_loc = all_rows.drop_duplicates(subset="placekey", keep="first")[["placekey", "latitude", "longitude"]]
gdf = gpd.GeoDataFrame(venue_loc, geometry=gpd.points_from_xy(venue_loc["longitude"], venue_loc["latitude"]),
                        crs="EPSG:4326").to_crs(epsg=2263)
gx = (gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
gy = (gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
venue_loc["grid_cell"] = (gx + "_" + gy).values
print(f"distinct venues: {len(venue_loc)}, distinct grid cells: {venue_loc['grid_cell'].nunique()}")

gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
venue_loc = venue_loc.reset_index(drop=True)
dummy_y = np.zeros(len(venue_loc))
fold_of = np.full(len(venue_loc), -1)
for fold_idx, (_, te_idx) in enumerate(gkf.split(venue_loc, dummy_y, groups=venue_loc["grid_cell"])):
    fold_of[te_idx] = fold_idx
venue_loc["venue_group"] = fold_of
pk_to_group = dict(zip(venue_loc["placekey"], venue_loc["venue_group"]))
print("venues per group:", venue_loc["venue_group"].value_counts().sort_index().to_dict())

for mm in frames:
    frames[mm]["venue_group"] = frames[mm]["placekey"].map(pk_to_group)

all_months = list(MODEL_READY_PATHS.keys())
HELD_OUT_VENUE_GROUP = 0  #test venues: this group, in the held-out month only
results = []

for held_out in all_months:
    train_months = [m for m in all_months if m != held_out]
    train = pd.concat(
        [frames[m][frames[m]["venue_group"] != HELD_OUT_VENUE_GROUP] for m in train_months],
        ignore_index=True,
    )
    test = frames[held_out][frames[held_out]["venue_group"] == HELD_OUT_VENUE_GROUP]

    overlap = set(train["placekey"]) & set(test["placekey"])
    assert not overlap, f"leakage: {len(overlap)} venues in both train and test"

    pre = ColumnTransformer([
        ("num", StandardScaler(), feats_num_full),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat),
    ])
    pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])

    Xtr, ytr, wtr = train[feats_num_full + feats_cat], train["income_mixing_score"], train["income_known_devices"].values
    pipe.fit(Xtr, ytr, reg__sample_weight=wtr)

    Xte, yte, wte = test[feats_num_full + feats_cat], test["income_mixing_score"], test["income_known_devices"].values
    pred = pipe.predict(Xte)
    r2_w = r2_score(yte, pred, sample_weight=wte)
    r2_u = r2_score(yte, pred)

    results.append({"held_out_month": held_out, "n_train": len(train), "n_test": len(test),
                     "r2_doubly_heldout_weighted": r2_w, "r2_doubly_heldout_unweighted": r2_u})
    print(f"held out {held_out}, new-venues-only test (n={len(test)}, trained on {train_months} "
          f"n={len(train)}): doubly-held-out R2 (weighted)={r2_w:.4f}, (unweighted)={r2_u:.4f}")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUTPUT_DIR, "geo_temporal_holdout_results.csv"), index=False)
print("\n", res_df.to_string(index=False))
print(f"\nmean doubly-held-out (weighted) R2: {res_df['r2_doubly_heldout_weighted'].mean():.4f} "
      f"(+/-{res_df['r2_doubly_heldout_weighted'].std():.4f})")
