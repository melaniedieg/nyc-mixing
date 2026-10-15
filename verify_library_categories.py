"""
verify_library_categories.py

Spot-checks every venue the location-data vendor tagged as a library
(sub_category == "Libraries and Archives") against NYC's official public
library locations (NYPL + Brooklyn Public Library + Queens Public Library),
by nearest-neighbor distance -- the same BallTree pattern the feature-build
scripts already use for MTA subway stations.

This does NOT change any model input or output. It's a documentation/QA
script: it prints what it finds so the check is reproducible, rather than
silently editing categories. See "Category reliability" under Known
limitations in README.md for how the results were used.

Why distance alone isn't enough: NYC's block density means many unrelated
venues sit within a few dozen meters of any given building, so a loose
radius (tried: 50m) mostly returns false positives (a nearby restaurant or
grocery store, not a miscategorized library). A tight radius (~10m) cuts the
noise down to something worth reading by hand, but even then, the result
needs a human to look at it -- this script surfaces candidates, it doesn't
adjudicate them.

Set DATA_DIR to point at your own copy of NYC_Library_Locations.csv and the
venue-level export(s) you want to check (defaults to ./data).

Usage:
    python verify_library_categories.py 2026-05 2026-06 2026-07
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

DATA_DIR = os.environ.get("DATA_DIR", "./data")
EARTH_R_M = 6_371_000.0
MATCH_RADIUS_M = 10  #tight on purpose -- see module docstring


def load_library_tree(path: str) -> tuple[BallTree, pd.DataFrame]:
    libs = pd.read_csv(path)
    coords = libs["the_geom"].str.extract(r"POINT \(([-\d.]+) ([-\d.]+)\)").astype(float)
    libs["lib_lon"], libs["lib_lat"] = coords[0], coords[1]
    tree = BallTree(np.radians(libs[["lib_lat", "lib_lon"]].to_numpy()), metric="haversine")
    return tree, libs


def check_month(venues_path: str, tree: BallTree, libs: pd.DataFrame) -> None:
    df = pd.read_csv(venues_path)
    dist, idx = tree.query(np.radians(df[["lat", "lon"]].to_numpy()), k=1)
    df["dist_to_real_library_m"] = dist[:, 0] * EARTH_R_M
    df["nearest_lib_name"] = libs["NAME"].to_numpy()[idx[:, 0]]

    near = df["dist_to_real_library_m"] <= MATCH_RADIUS_M
    tagged = df["sub_category"] == "Libraries and Archives"
    #Catching Washington Square Library required this too: it's tagged
    #"Museums," not "Libraries and Archives," so it's invisible to the
    #category-tag checks below on its own -- the name is what gives it away.
    named_library = df["location_name"].str.contains("library", case=False, na=False)

    print(f"  within {MATCH_RADIUS_M}m of a real library: {near.sum()}")
    print(f"  tagged 'Libraries and Archives':            {tagged.sum()}")
    print(f"  named like a library:                       {named_library.sum()}")
    print(f"  agree (near AND tagged):                     {(near & tagged).sum()}")

    missed = df[near & ~tagged].sort_values("dist_to_real_library_m")
    if len(missed):
        print(f"\n  near a real library but NOT tagged as one (n={len(missed)}) -- likely miscategorized:")
        print(missed[["location_name", "sub_category", "dist_to_real_library_m"]].to_string(index=False))

    #tagged as a library but far from any of the 216 official NYPL/BPL/QPL
    #points is NOT automatically an error -- academic, law, and private
    #libraries are real libraries that just aren't in this reference file.
    #Anything under ~50m is almost certainly the same branch with a small
    #geocoding offset; only look hard at the larger distances.
    far_tagged = df[tagged & (df["dist_to_real_library_m"] > 200)].sort_values("dist_to_real_library_m")
    if len(far_tagged):
        print(f"\n  tagged as a library but >200m from any official branch (n={len(far_tagged)}) -- "
              f"could be a real non-public library, or could be miscategorized/mislocated, check by hand:")
        print(far_tagged[["location_name", "dist_to_real_library_m", "nearest_lib_name", "borough"]].to_string(index=False))

    #named like a library, NOT tagged as one, AND far from a real one -- the
    #riskiest bucket: could be a genuinely mislocated/fabricated POI entry.
    #This is how Washington Square Library turned up.
    suspect = df[named_library & ~tagged & (df["dist_to_real_library_m"] > 100)].sort_values("dist_to_real_library_m")
    if len(suspect):
        print(f"\n  named like a library, not tagged as one, and >100m from any official branch "
              f"(n={len(suspect)}) -- check identity by hand before using in any library-specific claim:")
        print(suspect[["location_name", "top_category", "sub_category", "dist_to_real_library_m", "borough"]].to_string(index=False))


if __name__ == "__main__":
    months = sys.argv[1:] or ["2026-05", "2026-06", "2026-07"]
    tree, libs = load_library_tree(os.path.join(DATA_DIR, "NYC_Library_Locations.csv"))
    for month in months:
        path = os.path.join(DATA_DIR, f"venues_{month}.csv")
        if not os.path.exists(path):
            print(f"skipping {month}: {path} not found")
            continue
        print(f"=== {month} ===")
        check_month(path, tree, libs)
        print()
