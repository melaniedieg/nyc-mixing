"""
presence_bookkeeping.py

Point 5: distinguish, per (consolidated) venue entity and month, between "absent",
"present but under the income-known eligibility floor", "present and eligible but not
actually in the pooled persistence table", and "included in the pooled estimate" --
so a venue missing from a month's results reads as what it actually is, not as
"not mixed."

Uses the FULL raw analysis_pois_2026_0X.csv (the vendor's venue-level export before
ANY eligibility filtering -- 88k-94k rows/month here, vs. ~27-30k in model_ready),
which has both `total_devices` (any observed visit, no income-matching required) and
`income_known_devices` (the devices that could be traced to a home CBG -- this is
what the 50-device MIN_INCOME_KNOWN_DEVICES floor applies to). Note a real, scoped
limitation up front: this raw export itself requires >=1 observed device to produce a
row at all (confirmed: min total_devices across all 4 months is 1, zero rows at 0) --
so "absent from analysis_pois" here means "zero devices observed," not necessarily
"not a real place that could have been visited." Distinguishing that from "a mapped
POI that NO ONE visited this month" would need the full places catalog
(global_places_poi_geometry_combined.parquet), which lives only on the LaCie drive,
not in this pipeline -- flagged as a known gap, not quietly elided.

A second thing this surfaces, worth treating as a finding in its own right: fragment
consolidation (point 7) currently runs on model_ready -- which is ALREADY filtered to
eligible venues -- so if two fragments of the same real venue each sit under 50
income-known devices individually but would clear 50 combined, NEITHER fragment ever
reaches model_ready, and the current pipeline can't rescue that case. This script
quantifies how often that actually happens, using the raw analysis_pois counts
directly (which don't have that problem, since they're unfiltered).

Input:  DATA_DIR/analysis_pois_2026_0X.csv (raw, per month) and DATA_DIR/cbg_geoid_2026_0X.csv
        plus DATA_DIR/census_new.pkl -- same DATA_DIR build_features_2026_0X.py reads from
        (default ./data). OUTPUT_DIR/fragment_groups_report.csv and
        OUTPUT_DIR/consolidated_pooled_persistence_2026.pkl -- same OUTPUT_DIR
        consolidate_fragments.py wrote to (default ./output).
Output: OUTPUT_DIR/presence_bookkeeping_2026.csv -- one row per consolidated entity, with a
        status per month in {absent, present_ineligible,
        present_eligible_excluded_by_category, present_eligible_no_income_data,
        present_eligible_fragment_rescue_not_fitted, present_eligible_unexplained, pooled},
        plus a summary printed to stdout.
"""
import os
import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
#Full "YYYY_MM" period labels -- see the matching comment in pool_persistence.py. Lets a
#period outside 2026 (e.g. "2025_12") be folded in without assuming every period is "2026_<mm>".
PERIODS = ("2025_12", "2026_01", "2026_02", "2026_03", "2026_04", "2026_05", "2026_06", "2026_07")
RAW_PATHS = {
    mm: os.path.join(DATA_DIR, f"analysis_pois_{mm}.csv") for mm in PERIODS
}
ELIGIBILITY_FLOOR = 50

#Tracing an initial, much larger "present_eligible_not_pooled" count (20k+/month) found it
#was almost entirely explained by TWO upstream mechanisms already in build_features_2026_0X.py
#that are real and intentional but were not previously surfaced in the README/pitch at the
#level of "this is why most observed venues never reach the model," not just the 50-device
#floor:
#1. encounter_type category filter (build_features line ~219): a venue only enters the
#model at all if its top_category falls into public_civic/commercial_social/
#personal_social -- categories like contractors, auto dealers, gas stations, and retail
#stores unrelated to public/social space ("mobility"/"work_private"/"other") are excluded
#regardless of device count. This accounts for ~96% of the April gap (20,732 of 21,608).
#2. CBG-income match failure: a venue's spatial geocode to a block group can succeed (0
#failures found) while that specific block group still has no income record in the ACS
#extract (census_new.pkl) at all -- consistent with mostly-unpopulated CBGs (parks,
#cemeteries, industrial land), which is why "Museums, Historical Sites" is the single
#largest category in this residual. Accounts for the remaining ~4% (876 of 21,608).
#A third, much smaller mechanism surfaced once the above two were pulled out: a handful of
#fragment-consolidation groups (8 venue-months total, across all 4 months) where the COMBINED
#income-known-device count clears the 50-device floor but NEITHER individual fragment did on
#its own. Consolidation (point 7) runs on model_ready, which is already filtered to
#individually-eligible venues -- so neither fragment ever got its own expected_mixing_cv fit
#that month, and the merge step (a weighted average of fragments' OWN fitted predictions, see
#consolidate_fragments.py) has nothing to combine. These are real, correctly-identified
#"rescued" cases (see rescued_months below) that the current pipeline still can't pool, not a
#new unexplained gap -- its own explicit status rather than folded into category/income-match.
#All three mechanisms together are now their own explicit statuses below, rather than being
#lumped into one undifferentiated "not pooled" bucket that would have looked like an
#unexplained gap. A true "present_eligible_unexplained" count would mean a FOURTH, as-yet-
#undiscovered mechanism and should be investigated by hand if it is ever nonzero.
MODELED_CATEGORIES = {
    "Museums, Historical Sites, and Similar Institutions", "Other Amusement and Recreation Industries",
    "Civic and Social Organizations", "Promoters of Performing Arts, Sports, and Similar Events",
    "Colleges, Universities, and Professional Schools", "Performing Arts Companies",
    "Executive, Legislative, and Other General Government Support", "Arts, Entertainment, and Recreation",
    "Amusement Parks and Arcades", "Spectator Sports",
    "Religious, Grantmaking, Civic, Professional, and Similar Organizations",
    "Restaurants and Other Eating Places", "Grocery Stores", "Specialty Food Stores",
    "Drinking Places (Alcoholic Beverages)", "Special Food Services", "Food Services and Drinking Places",
    "General Merchandise Stores, including Warehouse Clubs and Supercenters", "Department Stores",
    "Book Stores and News Dealers", "Used Merchandise Stores",
    "Personal Care Services", "Other Personal Services",
}

#---- rebuild placekey -> consolidated entity id from the saved fragment-groups report ----
groups = pd.read_csv(os.path.join(OUTPUT_DIR, "fragment_groups_report.csv"))
pk_to_group = {}
group_name = {}
for _, row in groups.iterrows():
    for pk in row["placekeys"].split(","):
        pk_to_group[pk] = row["group_id"]
    group_name[row["group_id"]] = row["names"].split(";")[0].strip()

pooled = pd.read_pickle(os.path.join(OUTPUT_DIR, "consolidated_pooled_persistence_2026.pkl"))
pooled_months = {row["placekey"]: set(row["months"].split(",")) for _, row in pooled.iterrows()}
entity_name = dict(zip(pooled["placekey"], pooled["location_name"]))

#---- per-month raw presence/eligibility, rolled up to consolidated entities ----
monthly_status = {}   #entity_id -> {month: status}
monthly_combined_elig = {}  #(entity_id, month) -> combined income_known_devices
rescued_months = []  #(entity_id, month) cases where combining fragments crosses the floor

census_new = pd.read_pickle(os.path.join(DATA_DIR, "census_new.pkl"))[["geoid", "income_proxy_v2"]]
cbg_paths = {
    mm: os.path.join(DATA_DIR, f"cbg_geoid_{mm}.csv") for mm in PERIODS
}

for mm, path in RAW_PATHS.items():
    raw = pd.read_csv(path, usecols=["placekey", "income_known_devices", "total_devices", "location_name",
                                      "top_category", "sub_category"])
    cbg = pd.read_csv(cbg_paths[mm], dtype={"cbg_geoid": "string"})
    raw = raw.merge(cbg, on="placekey", how="left")
    raw = raw.merge(census_new, left_on="cbg_geoid", right_on="geoid", how="left")
    raw["in_modeled_category"] = raw["top_category"].isin(MODELED_CATEGORIES) | (raw["sub_category"] == "Libraries and Archives")
    raw["has_income_match"] = raw["income_proxy_v2"].notna()
    raw["entity"] = raw["placekey"].map(pk_to_group).fillna(raw["placekey"])

    agg = raw.groupby("entity").agg(
        combined_income_known_devices=("income_known_devices", "sum"),
        n_fragments_present=("placekey", "size"),
        any_fragment_individually_eligible=("income_known_devices", lambda s: (s >= ELIGIBILITY_FLOOR).any()),
        any_in_modeled_category=("in_modeled_category", "any"),
        any_has_income_match=("has_income_match", "any"),
    ).reset_index()
    present_entities = set(agg["entity"])

    for _, r in agg.iterrows():
        eid = r["entity"]
        combined_elig = r["combined_income_known_devices"] >= ELIGIBILITY_FLOOR
        in_pool = mm in pooled_months.get(eid, set())
        if in_pool:
            status = "pooled"
        elif not combined_elig:
            status = "present_ineligible"
        elif not r["any_in_modeled_category"]:
            status = "present_eligible_excluded_by_category"
        elif not r["any_has_income_match"]:
            status = "present_eligible_no_income_data"
        elif not r["any_fragment_individually_eligible"] and r["n_fragments_present"] > 1:
            #Combined devices clear the floor and the group is in-category with income data,
            #but no single fragment individually cleared the floor in the ORIGINAL
            #(pre-consolidation) pipeline run -- so no fragment ever got its own
            #expected_mixing_cv fitted that month, and there is nothing for the merge step to
            #combine. This is the documented consolidation gap from the module docstring
            #(lines 23-29), not a new mechanism: verified every case landing here is exactly
            #one of the `rescued_months` entries (checked across all 4 months: 8 venue-months
            #total, all with n_fragments_present==2).
            status = "present_eligible_fragment_rescue_not_fitted"
        else:
            #Should not occur: every present+eligible+in-category+income-matched case is
            #accounted for by either "pooled" or the fragment-rescue gap above. Kept as its own
            #named bucket (not silently folded into another status) so a future change that
            #breaks this invariant shows up in the status-count printout instead of hiding.
            status = "present_eligible_unexplained"
        monthly_status.setdefault(eid, {})[mm] = status
        monthly_combined_elig[(eid, mm)] = r["combined_income_known_devices"]
        if combined_elig and not r["any_fragment_individually_eligible"] and r["n_fragments_present"] > 1:
            rescued_months.append((eid, mm, r["combined_income_known_devices"], r["n_fragments_present"]))

    #everything in pk_to_group (or a singleton that shows up in the pooled table at all)
    #but absent from this month's raw export
    all_known_entities = set(pk_to_group.values()) | set(pooled["placekey"])
    for eid in all_known_entities - present_entities:
        monthly_status.setdefault(eid, {})[mm] = "absent"

print(f"fragment-group-months rescued by consolidation (individually sub-floor, "
      f"combined >={ELIGIBILITY_FLOOR}): {len(rescued_months)}")
for eid, mm, combined, nfrag in sorted(rescued_months, key=lambda x: -x[2])[:15]:
    print(f"  {eid} ({group_name.get(eid, eid)}), month {mm}: {nfrag} fragments present, "
          f"combined={combined:.0f} income-known devices")

#---- build the final table ----
rows = []
for eid, by_month in monthly_status.items():
    row = {"entity": eid, "name": entity_name.get(eid, group_name.get(eid, eid))}
    for mm in RAW_PATHS:
        row[f"status_{mm}"] = by_month.get(mm, "absent")
    rows.append(row)
report = pd.DataFrame(rows)
report.to_csv(os.path.join(OUTPUT_DIR, "presence_bookkeeping_2026.csv"), index=False)
print(f"\nsaved {OUTPUT_DIR}/presence_bookkeeping_2026.csv ({len(report)} entities)")

status_cols = [f"status_{mm}" for mm in RAW_PATHS]
print("\nstatus counts per month:")
for c in status_cols:
    print(f"  {c}: {report[c].value_counts().to_dict()}")

print("\n--- named headline venues ---")
for name_substr in ["Amenazel", "Chens Fresh", "Jacob Riis", "Hunts Point", "Riis"]:
    hit = report[report["name"].str.contains(name_substr, case=False, na=False)]
    if len(hit):
        print(f"\n{name_substr}:")
        print(hit.to_string(index=False))
