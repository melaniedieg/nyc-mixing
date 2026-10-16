"""
build_features_2026_06.py

Builds the model-ready, venue-level feature table for June 2026 from four inputs
(see README.md for exactly how to produce each one):

  - analysis_pois_2026_06_slim.csv   venue-level visit/income-mixing aggregates for the month
  - analysis_pois_2026_06.csv        the FULL (non-slim) export of the same month -- only used
                                      here for its q1_devices..q5_devices columns, which the slim
                                      export drops. Confirmed these sum exactly to
                                      income_known_devices and that normalized Shannon entropy of
                                      q1_pct..q5_pct reproduces income_mixing_score to <1e-10, so
                                      this is the real per-quintile breakdown behind the score,
                                      not a reconstruction (see fit_mixing_model_2026_06.py's
                                      visitor-sampling bootstrap, which resamples it directly).
  - cbg_geoid_2026_06.csv            placekey -> venue's own Census block group, from a spatial
                                      join of venue lat/long to the TIGER/Line block-group
                                      shapefile (see geocode_venues_to_cbg.py -- run that first)
  - census_new.pkl                   block-group income proxy + race composition, built from
                                      the raw ACS extract by the census-features section of
                                      income_mixing_model.ipynb
  - MTA_Subway_Stations.csv          public MTA subway station locations (data.ny.gov)

Set DATA_DIR / OUTPUT_DIR env vars to point at your own copies; defaults are ./data and
./output relative to wherever this script is run from.

Output: OUTPUT_DIR/model_ready_2026_06.pkl
"""
import os
import pandas as pd, numpy as np, time
from sklearn.neighbors import BallTree

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.width", 160)
t0 = time.time()
EARTH_R_M = 6_371_000.0

#---- venue-level aggregates for the month ----
poi = pd.read_csv(os.path.join(DATA_DIR, "analysis_pois_2026_06_slim.csv"), low_memory=False)
print("analysis_pois shape:", poi.shape)

#---- real per-quintile device counts, from the FULL export (the slim export drops these) --
#needed for the visitor-sampling bootstrap in fit_mixing_model_2026_06.py, which resamples
#each venue's own quintile composition rather than only resampling rows.
quintile_cols = ["q1_devices", "q2_devices", "q3_devices", "q4_devices", "q5_devices"]
poi_quintiles = pd.read_csv(
    os.path.join(DATA_DIR, "analysis_pois_2026_06.csv"),
    usecols=["placekey"] + quintile_cols, low_memory=False,
)
poi = poi.merge(poi_quintiles, on="placekey", how="left")
qsum = poi[quintile_cols].sum(axis=1)
mismatch = (qsum - poi["income_known_devices"]).abs() > 1
print(f"[{time.time()-t0:.0f}s] merged q1-q5 device counts, "
      f"{poi[quintile_cols].isna().any(axis=1).sum()} venues unmatched to the full export, "
      f"{mismatch.sum()} where q1..q5 sum disagrees with income_known_devices by >1 "
      f"(expected: 0 -- these would be excluded from the visitor-sampling bootstrap)")

#---- attach each venue's own geographic block group (from geocode_venues_to_cbg.py) ----
cbg = pd.read_csv(os.path.join(DATA_DIR, "cbg_geoid_2026_06.csv"), dtype={"cbg_geoid": "string"})
poi = poi.merge(cbg[["placekey", "cbg_geoid"]], on="placekey", how="left")
#Unmatched rate computed BEFORE stringification-with-zfill below -- astype(str) on a real NaN
#produces the literal string "nan" (then "000000000nan" after zfill), which .isna() no longer
#detects, so checking it after that step would always silently report 0% unmatched.
unmatched_rate = poi["cbg_geoid"].isna().mean()
poi["cbg_geoid"] = poi["cbg_geoid"].astype(str).str.zfill(12)
poi.loc[~poi["cbg_geoid"].str.fullmatch(r"\d{12}"), "cbg_geoid"] = np.nan
print(f"[{time.time()-t0:.0f}s] merged geographic cbg, unmatched: {unmatched_rate:.4f} "
      f"(venues with no valid 12-digit cbg_geoid are dropped by the dropna below, same as before, "
      f"but now for a diagnosable reason)")

#---- local commercial density, over the full mapped POI universe for the month (not just
#the eligible/modeled subset), so density reflects real physical surroundings ----
full = poi.dropna(subset=["latitude", "longitude"]).copy()
coords_rad = np.radians(full[["latitude", "longitude"]].values)
tree = BallTree(coords_rad, metric="haversine")

for radius_m in [250, 500]:
    counts = tree.query_radius(coords_rad, r=radius_m / EARTH_R_M, count_only=True)
    full[f"poi_density_{radius_m}m"] = counts - 1

cat = full["top_category"].values
ind250 = tree.query_radius(coords_rad, r=250 / EARTH_R_M)
same_cat = np.array([np.sum(cat[n] == cat[i]) - 1 for i, n in enumerate(ind250)])
full["same_category_density_250m"] = same_cat
print(f"[{time.time()-t0:.0f}s] commercial density computed")

#---- tourist-draw and chain density, same radius logic ----
TOURIST_CATEGORIES = {"Museums, Historical Sites, and Similar Institutions", "Traveler Accommodation"}
full["is_tourist_draw"] = full["top_category"].isin(TOURIST_CATEGORIES)
full["is_chain_poi"] = full["brands"].fillna("[]") != "[]"
tourist_flag = full["is_tourist_draw"].values
chain_flag = full["is_chain_poi"].values
for radius_m, label in [(500, "500m"), (800, "800m")]:
    ind = tree.query_radius(coords_rad, r=radius_m / EARTH_R_M)
    full[f"tourist_density_{label}"] = [tourist_flag[n].sum() - tourist_flag[i] for i, n in enumerate(ind)]
    full[f"chain_density_{label}"] = [chain_flag[n].sum() - chain_flag[i] for i, n in enumerate(ind)]
print(f"[{time.time()-t0:.0f}s] tourist/chain density computed")

#---- subway accessibility ----
stations = pd.read_csv(os.path.join(DATA_DIR, "MTA_Subway_Stations.csv"))
stations = stations.dropna(subset=["GTFS Latitude", "GTFS Longitude"])
s_tree = BallTree(np.radians(stations[["GTFS Latitude", "GTFS Longitude"]].values), metric="haversine")
dist, _ = s_tree.query(coords_rad, k=1)
full["dist_nearest_subway_m"] = dist[:, 0] * EARTH_R_M
for radius_m, label in [(500, "500m"), (800, "800m")]:
    full[f"subway_count_{label}"] = s_tree.query_radius(coords_rad, r=radius_m / EARTH_R_M, count_only=True)
print(f"[{time.time()-t0:.0f}s] transit computed")

#---- local income context: spatially-smoothed income proxy over nearby venues ----
#income_proxy_v2 (merged in below, on the full POI universe this time) is looked up via each
#venue's own geographic block group. That's a single administrative unit, and a real
#commercial corridor can straddle a block-group line, or draw from several adjacent block
#groups, none of which the venue's own single assigned block group captures. This adds, for
#every mapped POI, the average income_proxy_v2 of all OTHER mapped POIs within a radius (i.e.
#what the immediate few blocks actually look like income-wise, using everyone's own geographic
#block group) alongside the existing single-block-group value, the same way poi_density/
#tourist_density/chain_density already describe local context rather than relying on one
#administrative unit.
#
#Motivation: a venue's mixing_residual_z correlates with the average residual of its 5-10
#nearest other venues at r~0.61 (R^2~0.37), while top_category/sub_category explain under 1%
#of that same variance -- i.e. hyperlocal geography is a far stronger predictor of the
#"surprise" than what kind of business it is, which is what this feature is meant to capture.
census_new = pd.read_pickle(os.path.join(DATA_DIR, "census_new.pkl"))
full_econ = full.merge(
    census_new[["geoid", "income_proxy_v2"]],
    left_on="cbg_geoid", right_on="geoid", how="left"
)
income_vals = full_econ["income_proxy_v2"].to_numpy()
has_income = ~np.isnan(income_vals)

for radius_m in [250, 500]:
    ind = tree.query_radius(coords_rad, r=radius_m / EARTH_R_M)
    local_income = np.full(len(full_econ), np.nan)
    n_neighbors_known = np.zeros(len(full_econ), dtype=int)
    for i, neighbors in enumerate(ind):
        neighbors = neighbors[neighbors != i]  #exclude self
        valid = neighbors[has_income[neighbors]]
        n_neighbors_known[i] = len(valid)
        if len(valid) > 0:
            local_income[i] = income_vals[valid].mean()
    #Reliability fallback: a venue with no other income-known POI nearby (isolated location,
    #or a radius too small for this part of the city) falls back to its own single-CBG value
    #rather than becoming NaN and getting silently dropped from the model later.
    full_econ[f"local_income_proxy_{radius_m}m"] = np.where(
        n_neighbors_known > 0, local_income, income_vals
    )
    full_econ[f"local_income_n_neighbors_{radius_m}m"] = n_neighbors_known

full = full.merge(
    full_econ[["placekey", "local_income_proxy_250m", "local_income_proxy_500m",
               "local_income_n_neighbors_250m", "local_income_n_neighbors_500m"]],
    on="placekey", how="left"
)
print(f"[{time.time()-t0:.0f}s] local income context computed, "
      f"median neighbors within 500m: {full['local_income_n_neighbors_500m'].median():.0f}, "
      f"venues with 0 neighbors within 500m (fell back to own CBG): "
      f"{(full['local_income_n_neighbors_500m'] == 0).sum()}")

feat_cols = ["placekey", "poi_density_250m", "poi_density_500m", "same_category_density_250m",
             "tourist_density_500m", "tourist_density_800m", "chain_density_500m", "chain_density_800m",
             "dist_nearest_subway_m", "subway_count_500m", "subway_count_800m",
             "local_income_proxy_250m", "local_income_proxy_500m",
             "local_income_n_neighbors_250m", "local_income_n_neighbors_500m"]
poi = poi.merge(full[feat_cols], on="placekey", how="left")
for c in ["poi_density_250m", "poi_density_500m", "same_category_density_250m",
          "tourist_density_500m", "tourist_density_800m", "chain_density_500m", "chain_density_800m",
          "dist_nearest_subway_m", "subway_count_500m", "subway_count_800m"]:
    poi[f"log_{c}"] = np.log1p(poi[c])
#local_income_proxy_* is a level (like income_proxy_v2 itself), not a count, so it isn't
#log-transformed -- it goes into the model on the same footing as income_proxy_v2.

#---- encounter-type classification (libraries explicitly routed to public_civic) ----
public_civic_categories = [
    "Museums, Historical Sites, and Similar Institutions", "Other Amusement and Recreation Industries",
    "Civic and Social Organizations", "Promoters of Performing Arts, Sports, and Similar Events",
    "Colleges, Universities, and Professional Schools", "Performing Arts Companies",
    "Executive, Legislative, and Other General Government Support", "Arts, Entertainment, and Recreation",
    "Amusement Parks and Arcades", "Spectator Sports",
    "Religious, Grantmaking, Civic, Professional, and Similar Organizations",
]
commercial_social_categories = [
    "Restaurants and Other Eating Places", "Grocery Stores", "Specialty Food Stores",
    "Drinking Places (Alcoholic Beverages)", "Special Food Services", "Food Services and Drinking Places",
    "General Merchandise Stores, including Warehouse Clubs and Supercenters", "Department Stores",
    "Book Stores and News Dealers", "Used Merchandise Stores",
]
personal_social_categories = ["Personal Care Services", "Other Personal Services"]
mobility_categories = ["Transit and Ground Passenger Transportation", "Other Transit and Ground Passenger Transportation", "Taxi and Limousine Service"]
work_private_categories = [
    "Residential Building Construction", "Building Equipment Contractors", "Offices of Real Estate Agents and Brokers",
    "Automotive Repair and Maintenance", "Building Finishing Contractors", "Other Miscellaneous Manufacturing",
    "Foundation, Structure, and Building Exterior Contractors", "Printing and Related Support Activities",
    "Management of Companies and Enterprises", "Management, Scientific, and Technical Consulting Services",
    "Architectural, Engineering, and Related Services", "Nonresidential Building Construction", "Warehousing and Storage",
]

def classify(row):
    cat = row["top_category"]
    if row.get("sub_category") == "Libraries and Archives":
        return "public_civic"
    elif cat in public_civic_categories:
        return "public_civic"
    elif cat in commercial_social_categories:
        return "commercial_social"
    elif cat in personal_social_categories:
        return "personal_social"
    elif cat in mobility_categories:
        return "mobility"
    elif cat in work_private_categories:
        return "work_private"
    else:
        return "other"

poi["encounter_type"] = poi.apply(classify, axis=1)
encounter_df = poi[poi["encounter_type"].isin(["public_civic", "commercial_social", "personal_social"])].copy()
print(f"[{time.time()-t0:.0f}s] encounter classified, libraries included:",
      (poi.loc[poi["sub_category"] == "Libraries and Archives", "encounter_type"] == "public_civic").all())

INCOME_KNOWN_DEVICE_THRESHOLD = int(os.environ.get(
    "INCOME_KNOWN_DEVICE_THRESHOLD_OVERRIDE", "50"))  #sensitivity check override -- see
    #sensitivity_eligibility_floor.py / sensitivity_eligibility_floor_fdr.py
model_df = encounter_df[encounter_df["income_known_devices"] >= INCOME_KNOWN_DEVICE_THRESHOLD].copy()
model_df["log_devices"] = np.log1p(model_df["income_known_devices"])
print("model_df shape:", model_df.shape, "libraries:", (model_df["sub_category"] == "Libraries and Archives").sum())

#---- block-group income proxy + race composition (see README for how census_new.pkl is built) ----
census_new = pd.read_pickle(os.path.join(DATA_DIR, "census_new.pkl"))
model_df = model_df.merge(
    census_new[["geoid", "income_proxy_v2", "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_nh",
                "racial_diversity_index", "SE_A14001B_001"]].rename(columns={"SE_A14001B_001": "households"}),
    left_on="cbg_geoid", right_on="geoid", how="left"
)

borough_map = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan", "36081": "Queens", "36085": "Staten Island"}
model_df["borough"] = model_df["cbg_geoid"].str[:5].map(borough_map)

model_ready = model_df.dropna(subset=["income_mixing_score", "log_devices", "top_category", "income_proxy_v2"]).copy()
model_ready["log_population"] = np.log1p(model_ready["households"])
print(f"[{time.time()-t0:.0f}s] model_ready shape:", model_ready.shape)
print(model_ready["borough"].value_counts())
print("\nincome_mixing_score describe:")
print(model_ready["income_mixing_score"].describe())
print("\nbalance_score describe:")
print(model_ready["balance_score"].describe())

print("q1-q5 device-count columns present in model_ready:",
      all(c in model_ready.columns for c in quintile_cols))
model_ready.to_pickle(os.path.join(OUTPUT_DIR, "model_ready_2026_06.pkl"))
print(f"\n[{time.time()-t0:.0f}s] TOTAL DONE")
