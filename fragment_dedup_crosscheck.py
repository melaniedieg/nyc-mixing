"""
fragment_dedup_crosscheck.py

Scoping check for the device-dedup/distinctness gap in consolidate_fragments.py (external
review point 6): that script merges fragments purely by spatial proximity (75ft) + name-token
Jaccard similarity, and sums each merged group's q1-q5_devices across members -- with no cross-
check against the vendor's OWN authoritative duplicate/overlap signals, which were never even
read into the pipeline. This script reads those signals (already present in model_ready, carried
through from analysis_pois by build_features) and cross-checks them against the CURRENT,
committed fragment_groups_report.csv to see whether (a) any existing merged group actually
contains a true parent/child pair -- which, per SafeGraph's own documented convention, means the
child's devices are already counted inside the parent's, so SUMMING them (as this project does)
double-counts -- and (b) whether any SHARED_POLYGON / vendor-flagged shared_candidate venues that
plausibly represent the same real place are being left OUT of consolidation entirely (the
opposite risk -- undercounting a fragmented venue because the name/distance heuristic missed it).

This is a scoping/diagnostic script, not a fix: it does not change any committed output. It
reports what it finds so the actual fix (if one is warranted) can be scoped precisely.

Input: OUTPUT_DIR/model_ready_<PERIOD>.pkl for all 8 periods (already has parent_placekey,
       shared_candidate, polygon_class, enclosed -- carried through from analysis_pois by
       build_features_<PERIOD>.py), and OUTPUT_DIR/fragment_groups_report.csv (consolidate_
       fragments.py's own committed output).
Output: prints findings; saves fragment_dedup_crosscheck_results.csv (one row per flagged issue).
"""
import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
DIST_THRESHOLD_FT = 75.0  # same as consolidate_fragments.py's own adopted threshold
WIDENED_DIST_FT = 150.0   # widened net for Check C, to see what the adopted 75ft might be missing

# ---- Step 1: union of eligible venues across all 8 periods, same as consolidate_fragments.py.
# NOTE: parent_placekey/polygon_class/enclosed are present in the RAW analysis_pois_<PERIOD>.csv
# for every period (confirmed), but build_features_<PERIOD>.py only carries them through to
# model_ready for 2026_04 -- an inconsistency across the per-period build scripts in its own
# right, worth fixing separately. Worked around here by re-joining them from the raw export. ----
flag_cols = ["placekey", "parent_placekey", "shared_candidate", "polygon_class", "enclosed"]
frames = []
for mm in PERIODS:
    df = pd.read_pickle(os.path.join(OUTPUT_DIR, f"model_ready_{mm}.pkl"))
    base_cols = ["placekey", "location_name", "top_category", "latitude", "longitude",
                 "income_known_devices"]
    f = df[base_cols].copy()
    raw = pd.read_csv(os.path.join(DATA_DIR, f"analysis_pois_{mm}.csv"), usecols=flag_cols,
                       low_memory=False)
    have_flags = [c for c in ["parent_placekey", "shared_candidate", "polygon_class", "enclosed"]
                  if c in df.columns]
    if len(have_flags) < 4:
        f = f.merge(raw, on="placekey", how="left")
    else:
        f = df[base_cols + have_flags].copy()
    f["month"] = mm
    frames.append(f)

all_months = pd.concat(frames, ignore_index=True)
static_all = all_months.sort_values("month").drop_duplicates("placekey", keep="first").copy()
print(f"union of eligible placekeys across all 8 periods: {len(static_all)}")
print(f"  non-null parent_placekey: {static_all['parent_placekey'].notna().sum()}")
print(f"  shared_candidate == True: {(static_all['shared_candidate'] == True).sum()}")
print(f"  polygon_class == SHARED_POLYGON: {(static_all['polygon_class'] == 'SHARED_POLYGON').sum()}")

pk_set = set(static_all["placekey"])
static_idx = static_all.set_index("placekey")

# ---- Step 2: load the CURRENT, committed fragment groups ----
groups = pd.read_csv(os.path.join(OUTPUT_DIR, "fragment_groups_report.csv"))
pk_to_group = {}
group_members = {}
for _, row in groups.iterrows():
    gid = row["group_id"]
    members = [p.strip() for p in row["placekeys"].split(",")]
    group_members[gid] = members
    for pk in members:
        pk_to_group[pk] = gid
print(f"\ncurrent committed fragment groups: {len(groups)}, "
      f"covering {sum(len(m) for m in group_members.values())} placekeys")

# ---- Check A: true parent/child pairs where BOTH are eligible -- does the current
# clustering (a) correctly keep them apart, or (b) accidentally merge them (-> double count)? ----
print("\n=== CHECK A: parent/child pairs (both eligible) ===")
pc_pairs = []
for pk, parent in static_all[["placekey", "parent_placekey"]].dropna().itertuples(index=False):
    if parent in pk_set:
        pc_pairs.append((pk, parent))
print(f"parent/child pairs where BOTH placekeys are eligible: {len(pc_pairs)}")

merged_pc_pairs = []
separate_pc_pairs = []
for child, parent in pc_pairs:
    g_child = pk_to_group.get(child)
    g_parent = pk_to_group.get(parent)
    if g_child is not None and g_child == g_parent:
        merged_pc_pairs.append((child, parent, g_child))
    else:
        separate_pc_pairs.append((child, parent))

print(f"  -> currently MERGED into the same fragment group (double-count risk): {len(merged_pc_pairs)}")
print(f"  -> currently kept separate (no double-count from consolidation): {len(separate_pc_pairs)}")
for child, parent, gid in merged_pc_pairs[:20]:
    cn = static_idx.loc[child, "location_name"] if child in static_idx.index else "?"
    pn = static_idx.loc[parent, "location_name"] if parent in static_idx.index else "?"
    print(f"     {gid}: child={child} ({cn!r})  parent={parent} ({pn!r})")

# ---- Check B: among SHARED_POLYGON / vendor-flagged shared_candidate venues, how many are
# left OUT of any fragment group entirely (possible missed fragmentation -> undercount)? ----
print("\n=== CHECK B: SHARED_POLYGON / shared_candidate venues NOT in any fragment group ===")
shared_poly = static_all[static_all["polygon_class"] == "SHARED_POLYGON"]
shared_poly_unmerged = shared_poly[~shared_poly["placekey"].isin(pk_to_group.keys())].copy()
print(f"SHARED_POLYGON eligible venues: {len(shared_poly)}; "
      f"NOT in any current fragment group: {len(shared_poly_unmerged)} "
      f"({100*len(shared_poly_unmerged)/max(len(shared_poly),1):.1f}%)")

shared_cand = static_all[static_all["shared_candidate"] == True]
shared_cand_unmerged = shared_cand[~shared_cand["placekey"].isin(pk_to_group.keys())].copy()
print(f"vendor-flagged shared_candidate venues: {len(shared_cand)}; "
      f"NOT in any current fragment group: {len(shared_cand_unmerged)} "
      f"({100*len(shared_cand_unmerged)/max(len(shared_cand),1):.1f}%)")

# ---- Check C: among SHARED_POLYGON venues not already in a group, find near-duplicate
# candidates the adopted 75ft/Jaccard>=0.4 clustering may have missed -- same category, within a
# WIDENED 150ft radius, any positive name-token overlap (not requiring the full 0.4 threshold) --
# to size whether the heuristic's miss rate on this vendor-flagged population is a real gap. ----
print(f"\n=== CHECK C: near-duplicate candidates among unmerged SHARED_POLYGON venues "
      f"(widened to {WIDENED_DIST_FT:.0f}ft, same category, ANY name overlap) ===")

def name_tokens(name):
    STOP = {"the", "a", "an", "of", "and", "at", "in", "on", "inc", "corp", "llc", "co"}
    toks = re.findall(r"[a-z0-9]+", str(name).lower())
    return set(t for t in toks if t not in STOP and len(t) > 1)

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

cand = shared_poly_unmerged.reset_index(drop=True)
near_dupe_rows = []
if len(cand) > 1:
    gdf = gpd.GeoDataFrame(
        cand, geometry=gpd.points_from_xy(cand["longitude"], cand["latitude"]), crs="EPSG:4326"
    ).to_crs(epsg=2263)
    xy = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=WIDENED_DIST_FT, output_type="ndarray")
    names_tok = cand["location_name"].apply(name_tokens).to_numpy()
    cats = cand["top_category"].to_numpy()
    pks = cand["placekey"].to_numpy()
    for i, j in pairs:
        if cats[i] != cats[j]:
            continue
        jac = jaccard(names_tok[i], names_tok[j])
        if jac > 0:  # ANY positive overlap -- deliberately looser than the adopted 0.4, to see
                      # what sits just below/outside the current threshold
            d = float(np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
            near_dupe_rows.append({
                "pk_a": pks[i], "name_a": cand.loc[i, "location_name"],
                "pk_b": pks[j], "name_b": cand.loc[j, "location_name"],
                "dist_ft": round(d, 1), "name_jaccard": round(jac, 3),
                "top_category": cats[i],
            })
near_dupes = pd.DataFrame(near_dupe_rows).sort_values("name_jaccard", ascending=False) \
    if near_dupe_rows else pd.DataFrame(columns=["pk_a","name_a","pk_b","name_b","dist_ft","name_jaccard","top_category"])
print(f"candidate near-duplicate pairs found: {len(near_dupes)}")
print(near_dupes.head(25).to_string(index=False))

# ---- Save everything for inspection ----
out_rows = []
for child, parent, gid in merged_pc_pairs:
    out_rows.append({"check": "A_merged_parent_child", "group_id": gid, "pk_a": child, "pk_b": parent,
                      "name_a": static_idx.loc[child, "location_name"] if child in static_idx.index else None,
                      "name_b": static_idx.loc[parent, "location_name"] if parent in static_idx.index else None})
for child, parent in separate_pc_pairs:
    out_rows.append({"check": "A_separate_parent_child", "group_id": None, "pk_a": child, "pk_b": parent,
                      "name_a": static_idx.loc[child, "location_name"] if child in static_idx.index else None,
                      "name_b": static_idx.loc[parent, "location_name"] if parent in static_idx.index else None})
out_df = pd.DataFrame(out_rows)
out_df.to_csv(os.path.join(OUTPUT_DIR, "fragment_dedup_crosscheck_parent_child.csv"), index=False)
near_dupes.to_csv(os.path.join(OUTPUT_DIR, "fragment_dedup_crosscheck_near_dupes.csv"), index=False)
print(f"\nsaved {OUTPUT_DIR}/fragment_dedup_crosscheck_parent_child.csv "
      f"and {OUTPUT_DIR}/fragment_dedup_crosscheck_near_dupes.csv")
