"""
geocode_venues_to_cbg.py

Assigns each venue its own Census block group by spatially joining its coordinates
(latitude/longitude) to the 2025 TIGER/Line block-group polygons for New York State --
rather than inferring a venue's "neighborhood" from which block group its visitors happen
to live in. Run once per month, before build_features_2026_0X.py, which consumes its output.

Input:
  - analysis_pois_2026_0X_slim.csv       venue-level table with latitude/longitude (see README.md)
  - tl_2025_36_bg.shp (+ .shx/.dbf/.prj)  2025 TIGER/Line block-group shapefile for NY State,
                                          from the Census Bureau (see README.md)

Output: OUTPUT_DIR/cbg_geoid_2026_0X.csv   (placekey, cbg_geoid -- cbg_geoid is a nullable
                                            string; a venue that couldn't be matched gets a
                                            real null, not a placeholder digit string, so it
                                            can't silently pass as geocoded downstream)

Set MONTH (e.g. "2026_07"), DATA_DIR, OUTPUT_DIR, and TIGER_BG_SHP env vars as needed.
Requires geopandas and shapely (`pip install geopandas shapely`).
"""
import os
import re
import pandas as pd
import geopandas as gpd

MONTH = os.environ.get("MONTH", "2026_07")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
TIGER_BG_SHP = os.environ.get("TIGER_BG_SHP", os.path.join(DATA_DIR, "tl_2025_36_bg", "tl_2025_36_bg.shp"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bronx, Brooklyn (Kings), Manhattan (New York), Queens, Staten Island (Richmond)
NYC_COUNTY_FIPS = ["005", "047", "061", "081", "085"]

# A venue whose nearest block-group polygon (in the coastline-edge fallback below) is farther
# than this is left unmatched rather than silently assigned to an arbitrarily distant polygon --
# no real NYC venue should be this far from the nearest block group.
MAX_FALLBACK_DISTANCE_FT = 1000
GEOID_RE = re.compile(r"^\d{12}$")

bg = gpd.read_file(TIGER_BG_SHP)
bg_nyc = bg[bg["COUNTYFP"].isin(NYC_COUNTY_FIPS)].to_crs(epsg=4326)
print(f"NYC block groups: {len(bg_nyc)}")

poi = pd.read_csv(os.path.join(DATA_DIR, f"analysis_pois_{MONTH}_slim.csv"), low_memory=False)
poi_geo = poi.dropna(subset=["latitude", "longitude"]).copy()
gdf = gpd.GeoDataFrame(
    poi_geo, geometry=gpd.points_from_xy(poi_geo["longitude"], poi_geo["latitude"]), crs="EPSG:4326"
)

joined = gpd.sjoin(gdf, bg_nyc[["GEOID", "geometry"]], how="left", predicate="within")
joined = joined.drop_duplicates(subset="placekey", keep="first")

# A handful of venues (piers, marinas, beach-adjacent points) sit exactly on a coastline
# polygon edge and don't fall cleanly "within" any block group -- fall back to nearest polygon,
# in a projected CRS (NY State Plane, feet) so "nearest" is measured correctly.
missing_pks = joined.loc[joined["GEOID"].isna(), "placekey"]
if len(missing_pks):
    proj = gdf[gdf["placekey"].isin(missing_pks)].to_crs(epsg=2263)
    bg_nyc_proj = bg_nyc.to_crs(epsg=2263)
    nearest = gpd.sjoin_nearest(proj, bg_nyc_proj[["GEOID", "geometry"]], how="left", distance_col="_dist_ft")
    nearest = nearest.drop_duplicates(subset="placekey", keep="first")

    too_far = nearest["_dist_ft"] > MAX_FALLBACK_DISTANCE_FT
    if too_far.any():
        print(f"{too_far.sum()} of those venues' nearest block group is over "
              f"{MAX_FALLBACK_DISTANCE_FT}ft away (max {nearest.loc[too_far, '_dist_ft'].max():.0f}ft) "
              f"-- left unmatched rather than assigned to a distant polygon")
        nearest.loc[too_far, "GEOID"] = None

    fallback = dict(zip(nearest["placekey"], nearest["GEOID"]))
    joined["GEOID"] = joined.apply(
        lambda r: fallback.get(r["placekey"], r["GEOID"]) if pd.isna(r["GEOID"]) else r["GEOID"], axis=1
    )
    matched = nearest["GEOID"].notna().sum()
    dists = nearest.loc[nearest["GEOID"].notna(), "_dist_ft"]
    dist_note = f", median distance {dists.median():.0f}ft" if len(dists) else ""
    print(f"{len(missing_pks)} venues fell back to nearest-polygon match (coastline edge cases), "
          f"{matched} matched within {MAX_FALLBACK_DISTANCE_FT}ft{dist_note}")

# Unmatched count computed BEFORE stringification, on the real (still-nullable) GEOID column --
# astype(str) below would otherwise turn a missing value into the literal string "nan", which
# .isna() no longer detects, silently reporting 0 unmatched even when the join genuinely failed.
n_unmatched_join = joined["GEOID"].isna().sum()

out = joined[["placekey", "GEOID"]].rename(columns={"GEOID": "cbg_geoid"})
out["cbg_geoid"] = out["cbg_geoid"].astype("string").str.zfill(12)

# Validate: a real Census block-group GEOID is exactly 12 digits. Anything else -- including
# the "000000000nan" a missing value would otherwise silently become -- is set back to a true
# null (nullable StringDtype, not a placeholder) so it can't pass as geocoded downstream and
# accidentally match nothing in the Census extract without anyone noticing why.
valid = out["cbg_geoid"].str.fullmatch(GEOID_RE)
n_invalid_format = (~valid & out["cbg_geoid"].notna()).sum()
out.loc[~valid.fillna(False), "cbg_geoid"] = pd.NA

out_path = os.path.join(OUTPUT_DIR, f"cbg_geoid_{MONTH}.csv")
out.to_csv(out_path, index=False)
print(f"geocoded {len(out)} venues: {n_unmatched_join} unmatched by the spatial join, "
      f"{n_invalid_format} invalid GEOID format after cleanup, "
      f"{out['cbg_geoid'].isna().sum()} total null in output -> {out_path}")
