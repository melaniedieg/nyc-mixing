"""
sensitivity_spatial_block_bootstrap.py

Point 6 (sensitivity analyses), part 4: external review flagged that the "model-refit
stability" bootstrap in fit_mixing_model_2026_0X.py (BOOTSTRAP 1/2) does not reproduce the
model's own published scoring procedure, and ignores the project's own documented spatial
dependence.

What it does today (still shipping, unchanged, in fit_mixing_model_2026_07.py): resamples
individual VENUES (rows) with replacement, ignoring that neighbor venues' residuals are
correlated (documented ~0.52-0.61 -- the entire reason the published CV itself is grid-cell
GroupKFold rather than a plain KFold), refits ONE RandomForest on that resampled set, and
predicts on X_full -- every venue, including the ones whose rows were resampled into this
same model's own training set. That's an in-sample prediction for those venues, not the
out-of-fold prediction the published stability check is supposed to be testing the robustness
of, and it never re-applies the spatial blocking at all.

The reviewer's suggested fix: resample spatial BLOCKS (grid cells), not rows, and repeat the
actual grid-cell-blocked GroupKFold(5) spatial cross-fitting inside each bootstrap replicate,
so every prediction used to re-flag a venue stays genuinely held out.

This script implements that, against the exact published July 2026 reference flags:
  1. Resample the ~N distinct grid cells WITH replacement (not venues). A cell and every
     venue inside it travel together -- preserving the spatial-block structure the published
     CV itself relies on. Some cells are drawn more than once (their venues appear more than
     once in that replicate); some aren't drawn at all.
  2. On the resulting bootstrap replicate, rerun the published GroupKFold(5), grouped by the
     cell's ORIGINAL id (not by draw), so every occurrence of a given cell -- even if it was
     drawn twice -- lands in the same fold together. That's what keeps a venue's own
     neighborhood from leaking across train/test within a replicate, exactly as in the
     published CV.
  3. Residuals are z-scored within borough and re-flagged with the published
     borough_proportional_flags(), and we tally how often each of the REAL reference-flagged
     venues (from the actual production run, results_mixing_2026_07.pkl) reappears flagged --
     counting only the iterations where that venue's cell was actually drawn at all (a venue
     whose cell never gets drawn isn't evidence either way that iteration).
  4. Reports this corrected stability_pct against the ORIGINAL (flawed) row-resampling
     bootstrap's own stability_pct for the same venues, so any overstatement is visible
     directly, including for the four headline named venues.

Compute note: the original bootstrap fits ONE model per iteration (N_BOOT=30, 100 trees)
because it predicts in-sample. This corrected version refits the full 5-fold CV per
iteration (5x the model fits per draw), so N_BOOT_CORRECTED is deliberately smaller --
see README for the count actually used and why a smaller N is still a fair, like-for-like
test of the MECHANISM (the comparison is "does using the right procedure change the answer",
not "which bootstrap has the tighter interval").

Tested so far: July 2026 only, matching the other sensitivity checks in this series.

Input:  OUTPUT_DIR/model_ready_2026_07.pkl, OUTPUT_DIR/results_mixing_2026_07.pkl (for the
        real reference flags and the original bootstrap's own stability_pct, for comparison)
Output: sensitivity_spatial_block_bootstrap_results.csv -- one row per reference-flagged
        venue: stability_pct_original (flawed, row-level, as shipped) vs
        stability_pct_corrected (spatial-block) side by side, plus how many iterations each
        venue's cell was actually drawn in (n_iters_drawn).
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

MONTH = os.environ.get("MONTH", "2026_07")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
GRID_CELL_FT = 2000
PCT = 0.5
N_BOOT_CORRECTED = int(os.environ.get("N_BOOT_CORRECTED", "20"))

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


def borough_proportional_flags(z, borough, pct):
    """Identical to fit_mixing_model_2026_07.py's function of the same name."""
    top_mask = np.zeros(len(z), dtype=bool)
    bottom_mask = np.zeros(len(z), dtype=bool)
    z = np.asarray(z)
    borough = np.asarray(borough)
    for bor in np.unique(borough):
        idx = np.where(borough == bor)[0]
        k = max(1, int(round(len(idx) * pct / 100.0)))
        order = np.argsort(z[idx])
        bottom_mask[idx[order[:k]]] = True
        top_mask[idx[order[-k:]]] = True
    return top_mask, bottom_mask


df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{MONTH}.pkl"))
results = pd.read_pickle(os.path.join(OUTPUT_DIR, f"results_mixing_{MONTH}.pkl"))
stable_top, stable_bottom = results["stable_top"], results["stable_bottom"]

top_ref = stable_top["placekey"].tolist()
bottom_ref = stable_bottom["placekey"].tolist()
orig_stability = dict(zip(stable_top["placekey"], stable_top["stability_pct"]))
orig_stability.update(dict(zip(stable_bottom["placekey"], stable_bottom["stability_pct"])))
flag_kind = {pk: "top" for pk in top_ref}
flag_kind.update({pk: "bottom" for pk in bottom_ref})
name_lookup = dict(zip(stable_top["placekey"], stable_top["location_name"]))
name_lookup.update(dict(zip(stable_bottom["placekey"], stable_bottom["location_name"])))

print(f"[{time.time()-t0:.0f}s] {len(df)} venues; {len(top_ref)} reference-flagged top, "
      f"{len(bottom_ref)} reference-flagged bottom (from the published production run)")

# same grid-cell blocking as fit_mixing_model_2026_07.py
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

rng = np.random.RandomState(42)
top_counts_corr = {pk: 0 for pk in top_ref}
bottom_counts_corr = {pk: 0 for pk in bottom_ref}
top_drawn = {pk: 0 for pk in top_ref}
bottom_drawn = {pk: 0 for pk in bottom_ref}

cells_by_group = {cell: sub for cell, sub in df.groupby("_grid_cell")}

for b in range(N_BOOT_CORRECTED):
    sampled_cells = rng.choice(unique_cells, size=n_cells, replace=True)
    boot_df = pd.concat([cells_by_group[c] for c in sampled_cells], ignore_index=True)

    Xb = boot_df[feats_num_full + feats_cat]
    yb = boot_df["income_mixing_score"]
    wb = boot_df["income_known_devices"].values
    groups_b = boot_df["_grid_cell"].values  # original cell id -- every draw of the same
                                              # cell stays together in one fold, so a venue's
                                              # own neighborhood never splits train/test

    n_groups_b = len(np.unique(groups_b))
    gkf = GroupKFold(n_splits=min(5, n_groups_b))
    y_pred_boot = np.full(len(boot_df), np.nan)
    for tr_idx, te_idx in gkf.split(Xb, yb, groups=groups_b):
        pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=b, n_jobs=-1))])
        pipe.fit(Xb.iloc[tr_idx], yb.iloc[tr_idx], reg__sample_weight=wb[tr_idx])
        y_pred_boot[te_idx] = pipe.predict(Xb.iloc[te_idx])

    resid_boot = yb.to_numpy() - y_pred_boot
    boro_b = boot_df["borough"].to_numpy()
    resid_z_boot = np.empty_like(resid_boot)
    for bor in np.unique(boro_b):
        mask = boro_b == bor
        r = resid_boot[mask]
        resid_z_boot[mask] = (r - r.mean()) / r.std()

    top_mask_b, bottom_mask_b = borough_proportional_flags(resid_z_boot, boro_b, PCT)
    pk_b = boot_df["placekey"].to_numpy()
    top_pks_b = set(pk_b[top_mask_b])
    bottom_pks_b = set(pk_b[bottom_mask_b])
    drawn_pks = set(pk_b)

    for pk in top_ref:
        if pk in drawn_pks:
            top_drawn[pk] += 1
            if pk in top_pks_b:
                top_counts_corr[pk] += 1
    for pk in bottom_ref:
        if pk in drawn_pks:
            bottom_drawn[pk] += 1
            if pk in bottom_pks_b:
                bottom_counts_corr[pk] += 1

    print(f"[{time.time()-t0:.0f}s] spatial-block bootstrap {b+1}/{N_BOOT_CORRECTED} done "
          f"({len(boot_df)} rows from {n_cells} cell-draws, {n_groups_b} distinct cells present)")

rows = []
for pk in top_ref + bottom_ref:
    drawn = top_drawn.get(pk, bottom_drawn.get(pk))
    counts = top_counts_corr.get(pk, bottom_counts_corr.get(pk))
    corrected_pct = (counts / drawn * 100) if drawn > 0 else np.nan
    rows.append({
        "placekey": pk, "location_name": name_lookup.get(pk), "flag_kind": flag_kind[pk],
        "stability_pct_original_rowboot": orig_stability[pk],
        "stability_pct_corrected_spatialblock": corrected_pct,
        "n_iters_drawn": drawn, "n_iters_total": N_BOOT_CORRECTED,
    })
out = pd.DataFrame(rows).sort_values("flag_kind")
out["delta"] = out["stability_pct_corrected_spatialblock"] - out["stability_pct_original_rowboot"]
out.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_spatial_block_bootstrap_results.csv"), index=False)

print(f"\n[{time.time()-t0:.0f}s] SUMMARY across {len(out)} reference-flagged venues "
      f"(N_BOOT_CORRECTED={N_BOOT_CORRECTED}):")
print(f"  mean stability_pct, original row-resampling bootstrap:   {out['stability_pct_original_rowboot'].mean():.1f}%")
print(f"  mean stability_pct, corrected spatial-block bootstrap:   {out['stability_pct_corrected_spatialblock'].mean():.1f}%")
print(f"  mean delta (corrected - original):                       {out['delta'].mean():+.1f} pts")
print(f"  venues where corrected >=80% but original <80%: {((out['stability_pct_corrected_spatialblock']>=80)&(out['stability_pct_original_rowboot']<80)).sum()}")
print(f"  venues where original >=80% but corrected <80%: {((out['stability_pct_original_rowboot']>=80)&(out['stability_pct_corrected_spatialblock']<80)).sum()}")
print("\nfull table:")
print(out.to_string(index=False))
print(f"\nsaved {OUTPUT_DIR}/sensitivity_spatial_block_bootstrap_results.csv")
