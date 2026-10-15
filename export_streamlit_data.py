"""
Rebuilds the streamlit_app/data CSVs from the CURRENT, most-defensible pipeline outputs:
  - per-period expected_mixing_score / CV R^2 come straight from results_mixing_<period>.pkl's
    `expected_mixing_cv` / `cv_full`, i.e. the PRIMARY spatially-blocked (GroupKFold over ~610m
    grid cells) cross-validation -- not a re-fit plain-KFold model. Plain KFold (`cv_random` in
    the same pickle) overstates R^2 because neighboring venues' residuals are correlated
    (0.52-0.61), so it is deliberately not used here.
  - per-venue residual z-score is `improbable_copresence_z` from the same pickle -- already
    z-scored within borough using the borough-proportional convention (replaces the old script's
    own ad hoc `groupby("borough").transform(...)`).
  - evidence tier is the four-tier `persistence_tier` (single_month_not_tested /
    repeated_evidence / provisional_persistence / current_period_persistence) plus the
    REML+Hartung-Knapp, Benjamini-Hochberg FDR-corrected `fdr_sig_05_hk` / `q_bh_hk` / `mu_hk`
    from consolidated_pooled_persistence_2026.csv -- replaces the old two-tier
    stable_single_month/confirmed_multi_month scheme, which was based on a since-superseded
    one-off bootstrap-outlier flag file (three_month_stable_outliers_PROPORTIONAL_CORRECTED.csv)
    that only ever covered May-July 2026.
  - fragment-grouped venues (e.g. Jacob Riis Park's several POI records across months) are
    looked up via fragment_groups_report.csv's placekey->group_id membership, since
    consolidated_pooled_persistence_2026.csv keys a grouped venue by its GRP#####id, not by any
    one fragment's own placekey. Verified 100% join coverage against every period's own venue list.
  - a group's several fragments can ALL be present in the same period (confirmed: Jacob Riis
    Park's 3 fragments were all present in both April and July 2026) -- left unmerged, that's one
    physical place drawn as 2-3 map dots and counted 2-3x in every aggregate. This version
    collapses each group to one row per period: devices sum across its fragments (devices are
    genuinely additive); the displayed scores/location come from whichever one fragment had the
    most devices that period (its "primary" fragment) rather than an invented blended statistic,
    since averaging a z-score across fragments wouldn't itself be a valid z-score; and the display
    name is the canonical multi-name one already in consolidated_pooled_persistence_2026.csv
    (e.g. "Jacob Riis Park / Riis Park Visitor Center / The People's Beach at Jacob Riis Park")
    rather than whichever single fragment's name happened to be in that period's raw data.
  - `display_category` fixes one specific, verified mislabeling: 91% of venues filed under
    "Museums, Historical Sites, and Similar Institutions" are actually sub-categorized as
    "Nature Parks and Other Similar Institutions" (2,213 of 2,435 in July 2026) -- a category chart
    grouped by the raw top_category would read as "museums mix well" when it's really describing
    park visitation. Only top_category is relabeled for display; sub_category and the underlying
    data are untouched.

Covers all 8 currently-pooled periods: Dec 2025 through Jul 2026.
"""
import pickle
import pandas as pd

SHARED = "shared/output"

PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")


def period_key(p):
    """'2025_12' -> '2025-12' (matches the dashed key the app's MONTH_LABELS use)."""
    y, m = p.split("_")
    return f"{y}-{m}"


#--- fragment placekey -> group_id lookup -----------------------------------------------------
fg = pd.read_csv(f"{SHARED}/fragment_groups_report.csv")
frag_to_group = {}
for _, row in fg.iterrows():
    gid = row["group_id"]
    for pk in str(row["placekeys"]).split(","):
        frag_to_group[pk.strip()] = gid

#--- consolidated persistence / significance table, keyed by placekey-or-group_id -------------
cp = pd.read_csv(f"{SHARED}/consolidated_pooled_persistence_2026.csv")
cp_tier_cols = ["placekey", "location_name", "n_months", "persistence_tier", "months",
                "mu_hk", "q_bh_hk", "fdr_sig_05_hk"]
cp_lookup = cp[cp_tier_cols].set_index("placekey")

#known, verified mislabeling: this top_category is 91% nature parks by sub_category, not museums
MUSEUM_PARK_TOP = "Museums, Historical Sites, and Similar Institutions"
MUSEUM_PARK_SUB = "Nature Parks and Other Similar Institutions"
MUSEUM_PARK_DISPLAY = 'Parks & Nature Sites (filed under "Museums, Historical Sites")'

stats_rows = []

for period in PERIODS:
    key = period_key(period)

    with open(f"{SHARED}/results_mixing_{period}.pkl", "rb") as f:
        res = pickle.load(f)
    rf = res["results_full"].copy()
    cv_full = res["cv_full"]
    cv_r2 = float(cv_full.mean())

    md = pd.read_pickle(f"{SHARED}/model_ready_{period}.pkl")
    md_cols = md[["placekey", "sub_category", "latitude", "longitude", "balance_score"]]

    out = rf.merge(md_cols, on="placekey", how="left")

    out = out.rename(columns={
        "latitude": "lat", "longitude": "lon",
        "expected_mixing_cv": "expected_mixing_score",
        "improbable_copresence_z": "mixing_residual_z",
    })

    #sub_category fallback to top_category (matches the ~7% missing note in the app's README)
    out["sub_category"] = out["sub_category"].fillna(out["top_category"])

    #redirect fragment-member placekeys to their group's consolidated row
    out["join_key"] = out["placekey"].map(lambda pk: frag_to_group.get(pk, pk))
    tiers = cp_lookup.reindex(out["join_key"].values).reset_index(drop=True)

    out["persistence_tier"] = tiers["persistence_tier"].fillna("single_month_not_tested").values
    out["n_months"] = tiers["n_months"].fillna(1).astype(int).values
    out["months"] = tiers["months"].fillna(period).values
    out["mu_hk"] = tiers["mu_hk"].values
    out["q_bh_hk"] = tiers["q_bh_hk"].values
    out["fdr_sig_05_hk"] = tiers["fdr_sig_05_hk"].fillna(False).astype(bool).values
    #canonical display name -- for a grouped venue this is already the combined
    #"A / B / C" name consolidate_fragments.py built; for a singleton it's just
    #that venue's own name, so it's safe to use unconditionally. Overwrite in
    #place (not a separate column renamed in later) -- a rename alongside an
    #existing same-named column produces two columns both called
    #"location_name"/"placekey", and a plain column selection on that silently
    #keeps the ORIGINAL (per-fragment) one instead of the one just computed.
    out["location_name"] = tiers["location_name"].fillna(out["location_name"]).values

    n_before = len(out)
    group_sizes = out.groupby("join_key")["placekey"].transform("size")
    n_groups_affected = out.loc[group_sizes > 1, "join_key"].nunique()

    #one row per join_key: devices sum (genuinely additive); every other
    #per-period number/location comes from the single fragment with the most
    #devices that period ("primary"), not a blended average -- see module
    #docstring for why.
    out = out.sort_values("income_known_devices", ascending=False)
    primary = out.groupby("join_key", as_index=False).first()
    devices_sum = out.groupby("join_key")["income_known_devices"].sum()
    primary["income_known_devices"] = primary["join_key"].map(devices_sum).values
    out = primary

    if n_groups_affected:
        print(f"  collapsed {n_before - len(out)} duplicate fragment row(s) down to "
              f"{n_groups_affected} group row(s) present this period")

    out["display_category"] = out["top_category"].where(
        ~((out["top_category"] == MUSEUM_PARK_TOP) & (out["sub_category"] == MUSEUM_PARK_SUB)),
        MUSEUM_PARK_DISPLAY,
    )

    out["placekey"] = out["join_key"]  #overwrite in place -- see comment above
    out = out[["placekey", "location_name", "top_category", "display_category", "sub_category",
               "borough", "lon", "lat", "income_known_devices", "income_mixing_score",
               "expected_mixing_score", "balance_score", "mixing_residual_z",
               "persistence_tier", "n_months", "months", "mu_hk", "q_bh_hk", "fdr_sig_05_hk"]]

    out_path = f"venues_{key}.csv"
    out.to_csv(out_path, index=False)
    print(f"{period} ({key}): n={len(out)}  CV R2(primary, GroupKFold)={cv_r2:.4f}  "
          f"-> {out_path}  tiers:", out["persistence_tier"].value_counts().to_dict(),
          " sig(FDR .05, HK):", int(out["fdr_sig_05_hk"].sum()))

    stats_rows.append({"period": period, "month": key, "n": len(out), "cv_r2": round(cv_r2, 4),
                        "cv_r2_fold_min": round(float(cv_full.min()), 4),
                        "cv_r2_fold_max": round(float(cv_full.max()), 4)})

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv("month_stats.csv", index=False)
print("\nmonth_stats.csv:")
print(stats_df.to_string(index=False))
