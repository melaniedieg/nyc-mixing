"""
sensitivity_fragments.py

Point 6 (sensitivity analyses), part 1: stress-tests the two consolidation thresholds
(DIST_THRESHOLD_FT=75, NAME_JACCARD_THRESHOLD=0.4) chosen somewhat by eye in
consolidate_fragments.py. Reruns just the clustering step (no RF refit / bootstrap -- that part
doesn't change the question being asked here) across a grid of nearby threshold values, and
checks two things for each combination: (1) how much the total number of fragment groups moves,
and (2) whether the Jacob Riis Rockaways cluster (the case this whole method was built around)
stays correctly grouped as its 4 real fragments, and the unrelated Brooklyn "Jacob Riis Beach"
stays correctly excluded, across the whole grid -- not just at the one combination that was
picked.

Input: model_ready_2026_0X.pkl for all four months (set MODEL_READY_DIR, default ./output).
Output: sensitivity_fragments_results.csv -- one row per (distance, Jaccard) grid combination.
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import networkx as nx

MODEL_READY_DIR = os.environ.get("MODEL_READY_DIR", "./output")
MODEL_READY_PATHS = {
    mm: os.path.join(MODEL_READY_DIR, f"model_ready_2026_{mm}.pkl") for mm in ("04", "05", "06", "07")
}
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
STOPWORDS = {
    "the", "a", "an", "at", "of", "and", "&", "inc", "llc", "corp", "corporation", "ny", "nyc",
    "street", "st", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive", "way",
    "pl", "place", "plaza", "court", "ct", "sq", "square", "pkwy", "parkway", "hwy", "highway",
    "ln", "lane",
}
JACOB_RIIS_ROCKAWAY_PKS = {"zzy-222@627-wgg-sbk", "zzy-228@627-tmh-49z", "zzy-22f@627-tmh-49z",
                           "zzy-22b@627-tmh-49z"}
JACOB_RIIS_BROOKLYN_PK = "zzy-223@627-s92-gc5"


def name_tokens(name):
    if not isinstance(name, str):
        return set()
    toks = re.findall(r"[a-z0-9']+", name.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 1}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


static_frames = []
for mm, path in MODEL_READY_PATHS.items():
    df = pd.read_pickle(path)
    static_frames.append(df[["placekey", "location_name", "top_category", "latitude", "longitude"]])
static_all = pd.concat(static_frames, ignore_index=True).drop_duplicates("placekey").reset_index(drop=True)
print(f"union of eligible placekeys: {len(static_all)}")

gdf = gpd.GeoDataFrame(
    static_all, geometry=gpd.points_from_xy(static_all["longitude"], static_all["latitude"]), crs="EPSG:4326"
).to_crs(epsg=2263)
xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
names_tok = static_all["location_name"].apply(name_tokens).to_numpy()
cats = static_all["top_category"].to_numpy()
pks = static_all["placekey"].to_numpy()
pk_idx = {pk: i for i, pk in enumerate(pks)}

DIST_GRID = [40.0, 50.0, 60.0, 75.0, 100.0, 125.0, 150.0]
JAC_GRID = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6]

#widest distance needed, build the KD-tree pairs once at the max radius, then filter down
tree = cKDTree(xy)
max_pairs = tree.query_pairs(r=max(DIST_GRID), output_type="ndarray")
dists = np.hypot(xy[max_pairs[:, 0], 0] - xy[max_pairs[:, 1], 0],
                  xy[max_pairs[:, 0], 1] - xy[max_pairs[:, 1], 1])

results = []
for dist_thresh in DIST_GRID:
    for jac_thresh in JAC_GRID:
        keep = dists <= dist_thresh
        pairs_d = max_pairs[keep]
        edges = []
        for i, j in pairs_d:
            if cats[i] != cats[j]:
                continue
            jac = jaccard(names_tok[i], names_tok[j])
            if jac >= jac_thresh:
                edges.append((pks[i], pks[j]))
        G = nx.Graph()
        G.add_nodes_from(pks)
        G.add_edges_from(edges)
        components = [c for c in nx.connected_components(G) if len(c) > 1]
        n_groups = len(components)
        n_merged_pks = sum(len(c) for c in components)

        #Jacob Riis sanity check: do all 4 real Rockaway fragments land in ONE group, and is
        #the Brooklyn entry excluded from it?
        pk_to_comp = {}
        for ci, c in enumerate(components):
            for pk in c:
                pk_to_comp[pk] = ci
        riis_groups = {pk_to_comp.get(pk, f"singleton:{pk}") for pk in JACOB_RIIS_ROCKAWAY_PKS}
        riis_correctly_grouped = len(riis_groups) == 1 and not str(list(riis_groups)[0]).startswith("singleton")
        brooklyn_group = pk_to_comp.get(JACOB_RIIS_BROOKLYN_PK, "singleton")
        brooklyn_correctly_excluded = brooklyn_group not in riis_groups

        results.append({
            "dist_threshold_ft": dist_thresh, "jaccard_threshold": jac_thresh,
            "n_groups": n_groups, "n_placekeys_merged": n_merged_pks,
            "jacob_riis_rockaway_correctly_grouped": riis_correctly_grouped,
            "jacob_riis_brooklyn_correctly_excluded": brooklyn_correctly_excluded,
        })

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_fragments_results.csv"), index=False)
pd.set_option("display.width", 160)
print(res_df.to_string(index=False))

print(f"\nchosen combination (75ft, 0.4): "
      f"{res_df[(res_df.dist_threshold_ft==75.0)&(res_df.jaccard_threshold==0.4)].to_string(index=False)}")
print(f"\nJacob Riis Rockaway group correct in {res_df['jacob_riis_rockaway_correctly_grouped'].sum()}/{len(res_df)} "
      f"grid combinations; Brooklyn correctly excluded in "
      f"{res_df['jacob_riis_brooklyn_correctly_excluded'].sum()}/{len(res_df)}")
print(f"group count range across grid: {res_df['n_groups'].min()}-{res_df['n_groups'].max()} "
      f"(chosen combination: {res_df[(res_df.dist_threshold_ft==75.0)&(res_df.jaccard_threshold==0.4)]['n_groups'].iloc[0]})")
