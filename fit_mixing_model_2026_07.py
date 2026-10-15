"""
fit_mixing_model_2026_07.py

Fits the final RandomForest model against `income_mixing_score` for 2026-07,
reports CV R^2, borough-normalized residual z-scores, and TWO complementary bootstrap
stability checks on venues flagged as statistical outliers: a model-refit bootstrap (rows
resampled with replacement, same as before) and a visitor-sampling bootstrap (each venue's
own quintile composition resampled -- see the long comment above that block). The first asks
"is the model's ranking stable if refit on different rows"; the second asks "would THIS
VENUE's own score look different under a different sample of its own visitors" -- a flagged
venue is a stronger finding when it clears both, not just one.

CV is spatially blocked: venues are bucketed into ~2000ft (610m) grid cells (projected
to EPSG:2263, NY State Plane feet) and PRIMARY R^2 / expected_mixing_cv / residuals /
z-scores / outlier flagging all come from GroupKFold(5) grouped by grid cell, not a
random row-level KFold. Nearby venues' residuals are correlated (documented ~0.52-0.61
among neighbors), so a random row-level split lets a venue's near-neighbors leak into
its own training fold, inflating CV R^2 relative to genuine generalization to an unseen
neighborhood. The original random KFold(5, shuffle=True, random_state=42) is still fit
and reported as a SECONDARY comparison number only -- it does not feed expected_mixing_cv,
residuals, z-scores, or flagging anywhere below.

FIXED (Note 7, a fifth round of external review -- see README Known limitations, "No spatial
buffer between CV folds"): grid-cell blocking alone says nothing about cells on OPPOSITE sides
of a fold boundary sitting directly adjacent with a 0ft gap, and residuals are measurably
autocorrelated well past one cell width. The PRIMARY CV below now also enforces BUFFER_FT, a
spatial buffer between train and test within each fold -- see the long comment at that block
for the full derivation, the real correlogram numbers, and why BUFFER_FT=1500 (not larger) was
adopted.

Outlier flagging uses a borough-proportional rule: the top/bottom PCT% of venues
WITHIN EACH BOROUGH (by residual z-score), rather than a flat top/bottom-N citywide.
A flat citywide count doesn't scale with borough size (Staten Island and Manhattan
get the same number of "slots" despite very different venue counts) and is sensitive
to month-to-month noise near the cutoff -- this was adopted after finding that some of
the project's flagship venues (Jacob Riis Beach, Chens Fresh Food Market) were extreme
for their own borough but narrowly missed a flat citywide top/bottom-30 cutoff. See
README.md's Known limitations for why that same observation is NOT treated as validating
the specific 0.5% threshold (see the sensitivity table there instead).

The RandomForest is fit weighted by `income_known_devices`, down-weighting noisy,
low-device venues (see the diagnostic cell in income_mixing_model.ipynb).

CORRECTED (this patch): the visitor-sampling bootstrap (BOOTSTRAP 2/2 below) now compares
each resampled draw against the SAME fixed, out-of-fold expected_mixing_cv used for the
published headline residual, rather than refitting a new RandomForest per draw and predicting
in-sample. The earlier version (a) graded each draw against a model that had seen that draw's
own resampled answer -- an easier, different estimand than the real out-of-fold CV residual --
and (b) only ever kept an aggregated stability percentage, discarding the per-draw residuals a
real confidence interval needs. See the long comment at BOOTSTRAP 2/2 below for the full
explanation.

Input:  OUTPUT_DIR/model_ready_2026_07.pkl (from build_features_2026_07.py -- must
        include q1_devices..q5_devices; see that script's docstring)
Output: OUTPUT_DIR/results_mixing_2026_07.pkl
        {"stable_top": ..., "stable_bottom": ..., "imp_df": ..., "cv_full": ...,
         "cv_random": ..., "grid_cell_ft": ..., "pct": ..., "n_boot_visitor_sampling": ...,
         "results_full": ..., "resid_vs_draws_flagged": ...}
        cv_full is the PRIMARY (grid-cell GroupKFold) R^2 array; cv_random is the SECONDARY
        random-KFold comparison array. stable_top/stable_bottom carry both
        "stability_pct" (model-refit bootstrap) and "stability_pct_visitor_sampling", plus now
        "resid_vs_mean"/"resid_vs_var" (visitor-sampling bootstrap residual mean/variance per
        venue, against the FIXED expected_mixing_cv -- see BOOTSTRAP 2/2 below). results_full
        carries the same two columns for EVERY venue, not just the flagged subset, which is
        what a later cross-month persistence/pooling analysis needs -- a venue flagged in some
        months but not others still needs a comparable mean/variance in every month.
        resid_vs_draws_flagged has the raw per-draw residuals (one row per draw) for flagged
        venues only, kept for diagnostics.
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
from scipy.spatial import cKDTree

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCT = 0.5  #top/bottom 0.5% of venues, computed separately within each borough
GRID_CELL_FT = 2000  #~610m -- spatial block size for GroupKFold (see docstring)
BUFFER_FT = 1500  #Note 7 fix -- spatial buffer between train/test in the PRIMARY CV (see below)

pd.set_option("display.width", 160)
t0 = time.time()

df = pd.read_pickle(os.path.join(OUTPUT_DIR, "model_ready_2026_07.pkl"))
print("shape:", df.shape)

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
y = df["income_mixing_score"]

X_full = df[feats_num_full + feats_cat]
pre = ColumnTransformer([("num", StandardScaler(), feats_num_full), ("cat", OneHotEncoder(handle_unknown="ignore"), feats_cat)])
rf_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])

sample_weight_full = df["income_known_devices"].values

#Grid-cell spatial blocking: project to NY State Plane feet, bucket into GRID_CELL_FT
#cells, so nearby venues (whose residuals are correlated) land in the same fold instead
#of splitting across train/test.
_gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
).to_crs(epsg=2263)
_gx = (_gdf.geometry.x // GRID_CELL_FT).astype(int).astype(str)
_gy = (_gdf.geometry.y // GRID_CELL_FT).astype(int).astype(str)
grid_cell = (_gx + "_" + _gy).values
print(f"[{time.time()-t0:.0f}s] grid cells: {len(np.unique(grid_cell))} distinct "
      f"{GRID_CELL_FT}ft cells for {len(df)} venues")

#Note 7 fix: a spatial buffer, on top of grid-cell grouping -- see the module docstring and
#README Known limitations ("No spatial buffer between CV folds") for the full background.
#Grid-cell GroupKFold keeps a whole cell on one side of a fold, but cells on OPPOSITE sides of
#a fold boundary sit directly adjacent with a 0ft gap, and residuals are measurably
#autocorrelated well past one cell width -- a correlogram on real April 2026 residuals: r=0.56
#at 0-250ft, 0.44 at 250-500ft, 0.33 at 500-1000ft, 0.25 at 1000-1500ft, 0.23 at 1500-2000ft,
#0.16 at 2000-3000ft, fading to 0.07 by 3000-5000ft (see README for the full table). BUFFER_FT
#additionally excludes any training-candidate venue within BUFFER_FT of ANY test-fold venue --
#standard "buffered CV" (Roberts et al. 2017) -- rather than just excluding the test cell's own
#venues as before.
#
#BUFFER_FT=1500 is a deliberate compromise, not the distance where autocorrelation fully
#vanishes, stated plainly: at this venue density, a buffer wide enough to clear the
#correlogram's tail (3000ft+) excludes 85-90%+ of each fold's training venues and starts
#starving individual boroughs (Bronx, Staten Island) of training data entirely -- checked
#directly (per-borough training counts collapse to 122-337 venues at 3000ft, against 772+ at
#1500ft). 1500ft keeps a workable training set while still cutting the near-boundary
#correlation roughly in half relative to no buffer at all (0.25 at the fold edge vs. 0.56 at
#true proximity).
#
#Fit manually per fold (rather than via cross_val_score/cross_val_predict) so sample_weight
#can be split the same way X and y are for each fold.
#
#PRIMARY: grid-cell-blocked, BUFFERED GroupKFold. This is what feeds expected_mixing_cv,
#residuals, z-scores, and outlier flagging below.
_xy_all = np.column_stack([_gdf.geometry.x.to_numpy(), _gdf.geometry.y.to_numpy()])
_tree_all = cKDTree(_xy_all)
gkf_main = GroupKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = np.full(len(df), np.nan)
fold_r2 = []
fold_n_excluded = []
for tr_idx, te_idx in gkf_main.split(X_full, y, groups=grid_cell):
    _test_tree = cKDTree(_xy_all[te_idx])
    _near_test = _test_tree.query_ball_tree(_tree_all, r=BUFFER_FT)
    _near_test_idx = set()
    for _lst in _near_test:
        _near_test_idx.update(_lst)
    tr_idx_buf = np.array([i for i in tr_idx if i not in _near_test_idx])
    fold_n_excluded.append(len(tr_idx) - len(tr_idx_buf))
    fold_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
    fold_pipe.fit(X_full.iloc[tr_idx_buf], y.iloc[tr_idx_buf], reg__sample_weight=sample_weight_full[tr_idx_buf])
    pred = fold_pipe.predict(X_full.iloc[te_idx])
    y_pred_cv[te_idx] = pred
    fold_r2.append(r2_score(y.iloc[te_idx], pred, sample_weight=sample_weight_full[te_idx]))
cv_full = np.array(fold_r2)
print(f"[{time.time()-t0:.0f}s] buffer={BUFFER_FT}ft: mean train venues excluded per fold = "
      f"{np.mean(fold_n_excluded):.0f} ({np.mean(fold_n_excluded)/len(df)*100:.0f}% of n)")
print(f"\n[{time.time()-t0:.0f}s] PRIMARY (grid-cell, buffered GroupKFold) CV R2={cv_full.mean():.4f} "
      f"(+/-{cv_full.std():.4f}) [device-count weighted, spatially blocked + buffered]")

#SECONDARY, reported for comparison only -- NOT used for residuals/z-scores/flagging.
#This is the original random row-level KFold; it can overstate out-of-sample R^2 because
#spatially/residually-correlated neighbor venues can split across its train/test folds.
kf_random = KFold(5, shuffle=True, random_state=42)
fold_r2_random = []
for tr_idx, te_idx in kf_random.split(X_full):
    fold_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1))])
    fold_pipe.fit(X_full.iloc[tr_idx], y.iloc[tr_idx], reg__sample_weight=sample_weight_full[tr_idx])
    pred = fold_pipe.predict(X_full.iloc[te_idx])
    fold_r2_random.append(r2_score(y.iloc[te_idx], pred, sample_weight=sample_weight_full[te_idx]))
cv_random = np.array(fold_r2_random)
print(f"[{time.time()-t0:.0f}s] SECONDARY (random row-level KFold, reported for comparison "
      f"only -- not used below) CV R2={cv_random.mean():.4f} (+/-{cv_random.std():.4f})")

results_cv = df.copy()
results_cv["expected_mixing_cv"] = y_pred_cv
results_cv["improbable_copresence_cv"] = results_cv["income_mixing_score"] - results_cv["expected_mixing_cv"]

print(f"\n[{time.time()-t0:.0f}s] raw residual stats by borough:")
print(results_cv.groupby("borough")["improbable_copresence_cv"].agg(["count", "mean", "std", "min", "max"]))

rf_pipe.fit(X_full, y, reg__sample_weight=sample_weight_full)
names = rf_pipe.named_steps["pre"].get_feature_names_out()
imps = rf_pipe.named_steps["reg"].feature_importances_
imp_df = pd.DataFrame({"feature": names, "importance": imps}).sort_values("importance", ascending=False)
print(f"\n[{time.time()-t0:.0f}s] top 20 feature importances:")
print(imp_df.head(20).to_string(index=False))

#residuals are z-scored WITHIN borough so venues are compared to similar peers, not
#citywide (otherwise every venue in a borough with a different baseline looks "unusual")
results_cv["improbable_copresence_z"] = (
    results_cv.groupby("borough")["improbable_copresence_cv"]
    .transform(lambda s: (s - s.mean()) / s.std())
)

def borough_proportional_flags(z, borough, pct):
    """Returns (top_mask, bottom_mask): top/bottom pct% of venues, computed WITHIN each borough."""
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

top_mask_ref, bottom_mask_ref = borough_proportional_flags(
    results_cv["improbable_copresence_z"].values, results_cv["borough"].values, PCT
)
print(f"\nBorough-proportional flags (top/bottom {PCT}% per borough):")
print("Top borough counts:", results_cv.loc[top_mask_ref, "borough"].value_counts().to_dict())
print("Bottom borough counts:", results_cv.loc[bottom_mask_ref, "borough"].value_counts().to_dict())

cols = ["placekey", "location_name", "borough", "top_category", "income_known_devices",
        "income_mixing_score", "expected_mixing_cv", "improbable_copresence_cv", "improbable_copresence_z"]

#---- BOOTSTRAP 1/2, model-refit stability: resample venues (rows) with replacement, refit,
#re-normalize per iteration (borough-renormalized), and re-apply the SAME borough-proportional
#rule to see how often each reference flagged venue re-appears. This tests whether the MODEL's
#ranking is stable under refitting on different training rows -- it does NOT test whether a
#given venue's own score would look different under a different sample of its own visitors
#(see BOOTSTRAP 2/2 below for that). ----
N_BOOT = 30
top_ref = results_cv.loc[top_mask_ref, "placekey"].tolist()
bottom_ref = results_cv.loc[bottom_mask_ref, "placekey"].tolist()
top_counts = {pk: 0 for pk in top_ref}
bottom_counts = {pk: 0 for pk in bottom_ref}

placekeys = df["placekey"].values
borough_arr = df["borough"].values
n = len(df)
rng = np.random.RandomState(0)

for b in range(N_BOOT):
    idx = rng.randint(0, n, size=n)
    Xb, yb = X_full.iloc[idx], y.iloc[idx]
    boot_pipe = Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=b, n_jobs=-1))])
    boot_pipe.fit(Xb, yb, reg__sample_weight=sample_weight_full[idx])
    resid = y.values - boot_pipe.predict(X_full)
    resid_z = np.empty_like(resid)
    for bor in np.unique(borough_arr):
        mask = borough_arr == bor
        r = resid[mask]
        resid_z[mask] = (r - r.mean()) / r.std()
    top_mask_b, bottom_mask_b = borough_proportional_flags(resid_z, borough_arr, PCT)
    top_pks_b = set(placekeys[top_mask_b])
    bottom_pks_b = set(placekeys[bottom_mask_b])
    for pk in top_ref:
        if pk in top_pks_b:
            top_counts[pk] += 1
    for pk in bottom_ref:
        if pk in bottom_pks_b:
            bottom_counts[pk] += 1
    print(f"[{time.time()-t0:.0f}s] model-refit bootstrap {b+1}/{N_BOOT} done")

results_cv["stability_pct"] = np.nan
for pk, c in {**top_counts, **bottom_counts}.items():
    results_cv.loc[results_cv["placekey"] == pk, "stability_pct"] = c / N_BOOT * 100

#---- BOOTSTRAP 2/2, visitor-sampling stability: each venue's own income_mixing_score comes
#from a finite sample of visiting devices (as few as 50, the eligibility floor) sorted into
#5 income quintiles. The bootstrap above never varies that -- it resamples WHICH VENUES go
#into the training set, but every venue keeps its own exact, fixed score throughout. This
#block instead asks: if this same venue had drawn a different, equally-likely sample of
#income_known_devices visitors from the same underlying population, would its score -- and
#therefore its flagged status -- look the same?
#
#For each iteration, every venue's quintile counts are redrawn as
#Multinomial(n=income_known_devices, p=its own observed q1_pct..q5_pct), treating the
#observed shares as the true population rate (the standard nonparametric bootstrap assumption
#for categorical/compositional count data -- equivalent to resampling that venue's own
#devices' quintile labels with replacement). Each venue's income_mixing_score is then
#recomputed from the resampled shares using the vendor's own formula -- normalized Shannon
#entropy of the quintile shares -- verified byte-for-bit against the real column before this
#was written (agreement to <1e-10 on real June 2026 data), not re-derived or approximated.
#
#Each draw's residual is taken against expected_cv_arr -- the SAME fixed, out-of-fold,
#spatially-cross-validated prediction used for the published headline residual
#(expected_mixing_cv) -- NOT a freshly refit model's in-sample prediction of that same
#resampled draw. An earlier version of this block refit a new RandomForest on
#(X_full, resampled_y) and predicted on that same X_full, which (a) let the model see each
#draw's own resampled answer while "grading" it -- an easier, different estimand than the real
#out-of-fold CV residual -- and (b) only ever kept an aggregated stability percentage,
#discarding the per-draw residuals a real confidence interval needs. Holding expected_mixing_cv
#fixed and only re-measuring y isolates pure visitor-sampling noise around the actual published
#residual. It also means this block no longer needs a RandomForest refit per draw, which is
#why N_BOOT_VS below is far larger than the model-refit bootstrap's N_BOOT.
#
#Caveat, stated plainly: this assumes each income-known device's home-quintile draw is i.i.d.
#given the venue's observed rate. If visits cluster -- multiple devices from the same
#household, workplace, or tour group visiting together -- true sampling variance would run
#higher than a plain multinomial implies, and this would understate it.
#SUPERSEDED (Note 5, a third round of external review -- see README Known limitations, "The
#overdispersion check's independence assumption"): estimate_overdispersion.py's original 3-month
#check (May/June/July 2026 only) found VIF_obs=0.91 and read that as no detectable clustering --
#but that check never tested whether a venue's different months are independent of each other,
#an assumption that turned out to be false once all 8 periods existed to test it on. The
#corrected finding is VIF~1.48 (real evidence FOR clustering, reversing the old read). This
#block's plain-multinomial draws below are UNCHANGED and still produce resid_vs_mean/resid_vs_var
#in results_full, but pool_persistence.py/consolidate_fragments.py no longer read those columns --
#they now use estimate_visitor_sampling_variance_dm.py's Dirichlet-multinomial-corrected bootstrap
#instead (same fixed expected_mixing_cv design, just with the overdispersion correction applied).
#This block was left as-is rather than edited in place since nothing downstream still depends on
#its output; see the README bullet above for the full derivation and real numbers.
quintile_cols = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
q_counts = df[quintile_cols].to_numpy(dtype=float)
q_totals = q_counts.sum(axis=1)
mismatch = np.abs(q_totals - df["income_known_devices"].to_numpy()) > 1
if mismatch.any():
    print(f"WARNING: {mismatch.sum()} venues where q1..q5_devices sum disagrees with "
          f"income_known_devices by >1 -- their visitor-sampling draw uses the q-column sum, "
          f"not income_known_devices, so their resampled N is still internally consistent")
with np.errstate(invalid="ignore", divide="ignore"):
    q_pvals = np.where(q_totals[:, None] > 0, q_counts / np.where(q_totals[:, None] == 0, 1, q_totals[:, None]), 0.2)
LOG5 = np.log(5)

N_BOOT_VS = 500  #cheap now that it doesn't refit a RandomForest per draw -- see note above
expected_cv_arr = results_cv["expected_mixing_cv"].to_numpy()  #FIXED across every draw
resid_vs_draws = np.full((len(df), N_BOOT_VS), np.nan)  #[venue, draw], full venue universe
top_counts_vs = {pk: 0 for pk in top_ref}
bottom_counts_vs = {pk: 0 for pk in bottom_ref}
rng2 = np.random.RandomState(1)

for b in range(N_BOOT_VS):
    resampled_counts = np.vstack([
        rng2.multinomial(int(round(n_b)), p_b) if n_b > 0 else np.zeros(5)
        for n_b, p_b in zip(q_totals, q_pvals)
    ]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled_props = np.where(
            q_totals[:, None] > 0,
            resampled_counts / np.where(q_totals[:, None] == 0, 1, q_totals[:, None]),
            0.2,
        )
    resampled_props_clipped = np.clip(resampled_props, 1e-12, None)
    entropy = -(resampled_props_clipped * np.log(resampled_props_clipped)).sum(axis=1)
    y_vs = entropy / LOG5

    resid_vs = y_vs - expected_cv_arr  #against the FIXED cv expectation, not a per-draw refit
    resid_vs_draws[:, b] = resid_vs

    resid_z_vs = np.empty_like(resid_vs)
    for bor in np.unique(borough_arr):
        mask = borough_arr == bor
        r = resid_vs[mask]
        resid_z_vs[mask] = (r - r.mean()) / r.std()
    top_mask_vs, bottom_mask_vs = borough_proportional_flags(resid_z_vs, borough_arr, PCT)
    top_pks_vs = set(placekeys[top_mask_vs])
    bottom_pks_vs = set(placekeys[bottom_mask_vs])
    for pk in top_ref:
        if pk in top_pks_vs:
            top_counts_vs[pk] += 1
    for pk in bottom_ref:
        if pk in bottom_pks_vs:
            bottom_counts_vs[pk] += 1
    if (b + 1) % 50 == 0 or b == N_BOOT_VS - 1:
        print(f"[{time.time()-t0:.0f}s] visitor-sampling bootstrap {b+1}/{N_BOOT_VS} done")

results_cv["stability_pct_visitor_sampling"] = np.nan
for pk, c in {**top_counts_vs, **bottom_counts_vs}.items():
    results_cv.loc[results_cv["placekey"] == pk, "stability_pct_visitor_sampling"] = c / N_BOOT_VS * 100

#Full-universe visitor-sampling summary stats -- saved for EVERY venue, not just the flagged
#subset, so a venue flagged in some months but not others still has a comparable mean/variance
#in every month. resid_vs_var (the variance of resid_vs across the N_BOOT_VS draws) is the
#per-venue-month sampling variance a later inverse-variance pooling step needs.
results_cv["resid_vs_mean"] = resid_vs_draws.mean(axis=1)
results_cv["resid_vs_var"] = resid_vs_draws.var(axis=1, ddof=1)
results_cv["n_boot_vs"] = N_BOOT_VS

stable_top = results_cv[results_cv["placekey"].isin(top_ref)].sort_values("improbable_copresence_z", ascending=False)
stable_bottom = results_cv[results_cv["placekey"].isin(bottom_ref)].sort_values("improbable_copresence_z")

n_top, n_bottom = len(stable_top), len(stable_bottom)
print(f"\n[{time.time()-t0:.0f}s] MODEL-REFIT bootstrap -- "
      f"Top ({n_top} venues): {(stable_top['stability_pct'] >= 80).sum()}/{n_top} at >=80% stability, "
      f"{(stable_top['stability_pct'] < 50).sum()}/{n_top} below 50%, mean={stable_top['stability_pct'].mean():.1f}%")
print(f"Bottom ({n_bottom} venues): {(stable_bottom['stability_pct'] >= 80).sum()}/{n_bottom} at >=80% stability, "
      f"{(stable_bottom['stability_pct'] < 50).sum()}/{n_bottom} below 50%, mean={stable_bottom['stability_pct'].mean():.1f}%")

print(f"\n[{time.time()-t0:.0f}s] VISITOR-SAMPLING bootstrap -- "
      f"Top ({n_top} venues): {(stable_top['stability_pct_visitor_sampling'] >= 80).sum()}/{n_top} at >=80% stability, "
      f"{(stable_top['stability_pct_visitor_sampling'] < 50).sum()}/{n_top} below 50%, "
      f"mean={stable_top['stability_pct_visitor_sampling'].mean():.1f}%")
print(f"Bottom ({n_bottom} venues): {(stable_bottom['stability_pct_visitor_sampling'] >= 80).sum()}/{n_bottom} at >=80% stability, "
      f"{(stable_bottom['stability_pct_visitor_sampling'] < 50).sum()}/{n_bottom} below 50%, "
      f"mean={stable_bottom['stability_pct_visitor_sampling'].mean():.1f}%")

both_top = ((stable_top["stability_pct"] >= 80) & (stable_top["stability_pct_visitor_sampling"] >= 80)).sum()
both_bottom = ((stable_bottom["stability_pct"] >= 80) & (stable_bottom["stability_pct_visitor_sampling"] >= 80)).sum()
print(f"\n[{time.time()-t0:.0f}s] ROBUST UNDER BOTH bootstraps (>=80% on each) -- "
      f"Top: {both_top}/{n_top}, Bottom: {both_bottom}/{n_bottom}")

print("\nSTABLE TOP (borough-proportional):")
print(stable_top[cols + ["stability_pct", "stability_pct_visitor_sampling"]].to_string(index=False))
print("\nSTABLE BOTTOM (borough-proportional):")
print(stable_bottom[cols + ["stability_pct", "stability_pct_visitor_sampling"]].to_string(index=False))

results_full_cols = ["placekey", "location_name", "borough", "top_category", "income_known_devices",
                      "income_mixing_score", "expected_mixing_cv", "improbable_copresence_cv",
                      "improbable_copresence_z", "resid_vs_mean", "resid_vs_var"]
results_full = results_cv[results_full_cols].copy()

flagged_pks = set(top_ref) | set(bottom_ref)
flagged_mask = results_cv["placekey"].isin(flagged_pks).to_numpy()
resid_vs_draws_flagged = pd.DataFrame(
    resid_vs_draws[flagged_mask],
    index=results_cv.loc[flagged_mask, "placekey"].values,
)

pd.to_pickle({"stable_top": stable_top, "stable_bottom": stable_bottom, "imp_df": imp_df,
              "cv_full": cv_full, "cv_random": cv_random, "grid_cell_ft": GRID_CELL_FT, "pct": PCT,
              "n_boot_visitor_sampling": N_BOOT_VS, "results_full": results_full,
              "resid_vs_draws_flagged": resid_vs_draws_flagged},
             os.path.join(OUTPUT_DIR, "results_mixing_2026_07.pkl"))
print(f"\n[{time.time()-t0:.0f}s] TOTAL DONE")
