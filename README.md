# NYC Income Mixing — Methodology & Analysis Code

This repo contains the analysis pipeline behind a data investigation into where New Yorkers
of different income levels show up in the same commercial and public space, and where they
don't. It's published for transparency and reproducibility alongside the reporting, not as a
general-purpose tool — see **What's not here** below before assuming any input file is included.

**Scope**: this is a study of commercial and secular public space — restaurants, bars, parks,
gyms, retail, and similar venues that show up in mobile-location visit data. Congregations
(churches, synagogues, mosques), K-12 schools, health care facilities, and shelters are outside
what the underlying data can see: the vendor filters these categories out entirely before the
data reaches this pipeline (see **Known limitations** below). That means this project can speak
to where income groups' visit patterns overlap in commercial and recreational life, but says
nothing one way or the other about that overlap (or its absence) in the institutions — schools,
congregations, clinics — that a lot of "third place"/social infrastructure theory treats as
central.

**What this data can and can't say.** Four claims nest inside each other here, and this project
only makes the first:

1. **Persistent compositional diversity** -- a venue's monthly visitor base draws from a wider
   (or narrower) range of incomes than its neighborhood alone would predict, and that pattern
   repeats across independently-rebuilt months. This is what `income_mixing_score` and every
   downstream number in this repo measures, and it's the only claim this project makes.
2. **Cross-income co-presence** (different income groups physically in the same place at the
   same time, not just the same month) would need visit-level time-of-day data this pipeline
   does not have. Not claimed anywhere in this repo or its reporting.
3. **Meaningful social interaction** isn't recoverable from mobile-location data under any
   analysis, however the model is built. Not claimed, and not implied by the word "mixing" as
   used here -- it is this project's technical shorthand for compositional diversity (the same
   way a "diversity index" doesn't imply the index itself socializes), not a description of
   visitor behavior.
4. **The venue itself causes the mixing it shows** would need a causal design (a natural
   experiment, a before/after, a matched comparison), not the cross-sectional model this repo
   fits. Not claimed.

Every score, residual, z-score, and pooled estimate produced by the scripts below is tier 1 only.
Any writeup built from this repo's output should be too.

## What the model does

For roughly 26,000-30,000 actively-visited NYC establishments per month, each venue gets an
`income_mixing_score` (and a related `balance_score`) built from anonymized, aggregated
mobile-location data: how evenly a venue's visitors are spread across income quintiles,
estimated from each visitor device's home Census block group. A venue in a naturally
income-diverse neighborhood will look "mixed" for purely geographic reasons, so the real
question isn't the raw score, it's the residual: given a venue's borough, neighborhood income,
race composition, local commercial density, subway access, and proximity to tourist/cultural
draws, is it *more or less* mixed than that context predicts?

A RandomForest model is fit against that context to produce an expected score per venue; the
residual (actual minus expected) is z-scored **within borough** (so venues are compared to
similar peers, not citywide) to flag statistical outliers.

That residual comes from a spatially-blocked, BUFFERED cross-validation, not a random one.
Nearby venues' residuals are correlated (documented below under **Known limitations**, r≈0.52-0.61
among near neighbors) — a random row-level K-fold split lets a venue's own physical neighbors leak
into its training fold, inflating validation R² relative to genuine generalization to an unseen
neighborhood. Venues are grouped into ~2000ft (610m) grid cells (projected to NY State Plane
feet) and 5-fold cross-validation is grouped by cell (`GroupKFold`); on top of that, a 1500ft
spatial BUFFER additionally excludes any training venue within 1500ft of a test-fold venue, not
just venues sharing its own cell (Note 7, a fifth round of external review -- see "No spatial
buffer between CV folds" under Known limitations for the full derivation and why grid-cell
grouping alone wasn't enough). This grid-cell, buffered R² (0.496-0.665 across the current 8
periods, Dec 2025-Jul 2026, device-count weighted) is the PRIMARY number and the one that feeds
the residuals, z-scores, and outlier flagging described below; the original random row-level
K-fold R² (0.76-0.77) is still fit and reported alongside it as a secondary comparison figure
only, as is the un-buffered grid-cell R² (0.55-0.73, the PRIMARY number before Note 7 -- see that
bullet). December's buffered R² (0.496) is the lowest of the 8 periods -- it's also by far the
smallest period (10,048 eligible venues, under half of July's 26,662), so a 1500ft buffer excludes
a larger share of an already-small, sparser training set; every data-quality check run on
December came back clean, so this reads as a real small-N effect of the buffer rather than a data
issue, but it's worth knowing about rather than rounding away.

Outliers are flagged as the top/bottom
0.5% of venues **within each borough** (not a flat citywide count): a fixed citywide number of
"slots" doesn't scale with borough size, and is sensitive to month-to-month noise near the cutoff
in a way a per-borough percentage isn't. This replaced an earlier flat top/bottom-30-citywide rule
after checking it directly against this data: two of the project's flagship venues (Jacob Riis
Beach, Chens Fresh Food Market) were genuinely extreme for their own borough but narrowly missed
the flat citywide cutoff some months — a concrete case of the exact failure mode a fixed-count
rule is prone to. A venue is only treated as a real finding, not noise, if it clears the following
bars:

1. **Bootstrap-stable, under TWO different kinds of resampling**: a venue must hold up under both
   of two complementary bootstraps, not just one. **Model-refit stability** (30 iterations)
   resamples which VENUES (rows) go into the training set and refits -- it tests whether the
   model's ranking is stable, not whether a given venue's own score is. **Visitor-sampling
   stability** (500 iterations) instead holds the model AND every venue's fixed, out-of-fold
   spatial-CV expected score completely fixed, and resamples only that one venue's own
   `income_mixing_score` from `Multinomial(income_known_devices, q1_pct..q5_pct)` -- i.e., "if
   this venue's visitors that month had been a different, equally likely sample of the same size,
   how far would its residual move against that SAME fixed expectation?" An earlier version of
   this bootstrap refit the model on every resampled draw and re-flagged against a newly-refit
   expectation each time, which mixed visitor-sampling noise together with model-refit noise and
   made the two bootstraps partially redundant with each other; comparing every draw against one
   fixed reference isolates pure visitor-sampling noise, and is also why this bootstrap now runs
   500 draws per venue instead of 30 -- there's no refit per draw anymore. The two catch different
   failure modes: a venue that's stable under model-refit but noisy under visitor-sampling is one
   whose home, device-count-limited sample is doing most of the work, not a genuinely improbable
   pattern.
2. **Persistent across months, not just repeatedly flagged**: the entire pipeline -- feature
   build, model fit, both bootstraps -- is rebuilt from scratch on each separate month of data.
   Rather than just checking whether a venue gets re-flagged, each venue's per-month OBSERVED
   spatial-CV residual (`improbable_copresence_cv` -- actual score minus the fixed, out-of-fold
   model expectation) is pooled across every month it clears the eligibility floor with a
   REML + Hartung-Knapp random-effects meta-analysis (the ADOPTED method; see **Known
   limitations** for why DerSimonian-Laird, kept only as a diagnostic/comparison here, is known
   to be anti-conservative at this sample size), weighted by the inverse of that month's
   visitor-sampling-bootstrap variance (see **Known limitations** for the real numbers and an
   important correction to how this used to be computed) -- separating genuine month-to-month
   movement in a venue's own mixing from visitor-sampling noise, producing a pooled estimate and
   a CONDITIONAL confidence interval (conditional on the fitted contextual model -- see below for
   what that does and doesn't cover) that widens honestly when a venue's months disagree with each
   other more than sampling noise alone would predict. The resulting p-values go through a
   Benjamini-Hochberg false-discovery-rate correction across every venue with 2+ eligible months
   (a venue tested only once has no independent replication to assess persistence against, so it's
   excluded from that correction rather than diluting or inflating it). Each venue also gets a
   plain-language `persistence_tier` label (`repeated_evidence` at 2 months, `provisional_persistence`
   at 3, `current_period_persistence` at all 4 of the current April-July window) -- see **Known
   limitations** for why none of this can yet earn the word "annual."

A second scoring approach, `balance_score`, is also fit and checked against flagged venues (see
`fit_balance_model_2026_0{4,5,6,7}.py` below), but it isn't treated as an independent third bar:
empirically it tracks `income_mixing_score` too closely (~0.98 correlated among flagged venues)
to count as separate confirmation. It functions as a robustness check that the primary score
isn't an artifact of one particular target definition, not as corroborating evidence from an
unrelated measure — see **Known limitations** for the full caveat.

## Validation checks already run

Four specific defensibility questions came up in review. Stated plainly here, with what was
actually checked, rather than just asserted:

**Is `income_mixing_score` reconstructable from published, non-proprietary numbers, not a black
box?** Yes. It is the normalized Shannon entropy of a venue's own visitor quintile shares:
`income_mixing_score = -sum(p_k * log(p_k) for k in 1..5) / log(5)`, where `p_k` is the share of
a venue's income-known devices in income quintile `k` that month (0 when a venue draws from
only one quintile, 1 when its visitors are split evenly across all five). `build_features_2026_0X.py`
confirms this reproduces the vendor-supplied `income_mixing_score` column to within 1e-10 from
the vendor's own `q1_devices..q5_devices` columns -- i.e. this is the real mechanism behind the
score, independently verified, not a black-box number taken on faith, and not a figure that
depends on anything in this repo being correct to reconstruct from the raw per-quintile counts.

**Does any context feature leak information from WHO visits a venue, rather than WHERE it sits?**
No -- checked directly against every feature the model uses. A venue's Census block group comes
from a spatial join of the venue's OWN coordinates to the TIGER/Line block-group shapefile
(`geocode_venues_to_cbg.py`), never from where its visitors live. Density, tourist/chain
proximity, and subway-access features are all built from other venues' and subway stations'
fixed locations, spatially blocked the same way. The one feature that smooths over multiple
venues (`local_income_proxy_250m/500m`) averages nearby venues' own already-venue-derived
`income_proxy_v2`, not anything about who visits any of them. No feature in `feats_num_full` /
`feats_cat` in `fit_mixing_model_2026_0X.py` is derived from visitor home locations.

**Is the finite-device-count problem actually handled, not just mentioned?** Yes, in three
places that compound rather than substitute for each other: (1) the 50-device eligibility floor
excludes the thinnest venues outright; (2) every model fit -- CV folds, the final fit, and every
bootstrap refit -- is weighted by `income_known_devices`, so a venue with 60 devices influences
the fitted relationship less than one with 600; (3) the visitor-sampling bootstrap's own
variance scales directly with device count (a smaller multinomial sample produces noisier
resampled draws), so a thin venue's pooled confidence interval is honestly wider, not just its
point estimate suppressed.

**Does the model generalize across TIME, not just across space within one month?** This is the
one real gap that needed a new analysis, not just a code check, and it surfaced something worth
knowing on the way to the answer. The existing grid-cell-blocked `GroupKFold` (see "What the
model does" above) only tests spatial generalization within a single month. `geo_temporal_holdout_validation.py`
tests the missing axis: train on three months pooled, predict the fourth, entirely out of time.
The direct version of that test came back with a mean weighted R²=0.78 -- HIGHER than the
within-month spatial CV's 0.69-0.73, which was a red flag rather than a win. The cause:
92.4% of April's venues also appear in at least one other month (64% of all 33,767 distinct
venues appear in all four), and every context feature except `log_devices` is derived from a
venue's fixed physical location, so it barely changes month to month. The model had effectively
already seen most "held-out" venues' context during training -- the same leakage mechanism that
justified grid-cell-blocking the within-month CV in the first place, just crossing months
instead of space, and just as able to inflate an R² if left unblocked.

Corrected for that -- holding out a venue's physical location from training in ALL months, not
just the target month, using the same ~2000ft grid-cell grouping applied to the pooled set of
all distinct venues -- the result is the reassuring one: mean weighted R²=0.7263 (range
0.698-0.743 across the four held-out months), close to and consistent with the within-month
spatial-CV figures for the corresponding months (April's doubly-held-out R²=0.698 against its
own within-month 0.69; the three summer months cluster the same way). No evidence the model is
overfit to a specific month once venue overlap is actually blocked rather than ignored. See
`geo_temporal_holdout_validation.py` and `geo_temporal_holdout_results.csv`.

**Known follow-up, stated plainly rather than silently left inconsistent:** the R² figures in
this subsection (mean weighted 0.78 unblocked / 0.7263 blocked, and the "April's doubly-held-out
0.698 against its own within-month 0.69" comparison) all predate Note 7's spatial-buffer fix to
the within-month CV below -- they were computed against the unbuffered `results_mixing_<MONTH>.pkl`
outputs. April's own within-month figure is now 0.620, not 0.69, under the buffered CV (see "No
spatial buffer between CV folds" under Known limitations), so the "close to and consistent with"
comparison above needs rechecking against the buffered numbers, and the cross-month holdout itself
arguably needs the same spatial buffer applied to its own train/predict split, not just the
within-month CV. Neither has been rebuilt yet; this is flagged as a follow-up, not yet scoped or
confirmed.

**How much do the two consolidation thresholds (75ft, name-token Jaccard≥0.4) and the 50-device
eligibility floor, each picked somewhat by eye, actually drive the results?** Checked both --
first for CV R²/grouping correctness, then (Note 8, a sixth round of external review) for the
thing that actually matters downstream: whether either choice moves the real FDR-survivor count
and the four named venues' own significance calls, not just an intermediate diagnostic.

*Fragment-consolidation thresholds.* `sensitivity_fragments.py` reruns the clustering step of
`consolidate_fragments.py` across a 42-combination grid (distance ∈ {40,50,60,75,100,125,150}ft ×
name-token Jaccard ∈ {0.25,0.3,0.35,0.4,0.5,0.6}), built on one shared KD-tree query at the widest
radius rather than rebuilding the tree per combination, and (Note 8) at every grid point also
re-merges each combination's fragment groups and reruns the full REML+Hartung-Knapp pooling/BH-FDR
across the resulting ~31,650-entity family -- not just the clustering, so the grid speaks to the
same conclusions the README actually reports. Three things were checked at every grid point: (1)
whether the unrelated Brooklyn "Jacob Riis Beach" stays excluded, (2) whether the Rockaway Jacob
Riis cluster's 4 real fragments stay correctly grouped together, and (3, Note 8) how much the
grid point moves the FDR survivor count and the four named venues' significance calls. Brooklyn
is correctly excluded in all 42/42 combinations -- that part of the method is robust across the
whole grid. The Rockaway grouping is correct in 28/42, breaking down specifically once the
Jaccard threshold reaches 0.5 or higher (both 0.5 and 0.6 fail it at every distance tested);
distance threshold on its own never breaks it within the range tested. That means 0.4 is closer
to necessary than merely a reasonable-looking choice -- the method depends on not setting that
threshold too strict, which is worth knowing plainly rather than presenting 0.4 as an
arbitrary-but-fine pick. The total number of groups found ranges from 106 (strictest) to 770
(loosest) across the grid, driven mostly by the Jaccard threshold, not the distance threshold --
consistent with the earlier finding that distance alone rarely breaks the Rockaway case. The FDR
survivor count stays in a tight 4,845-4,909 band across all 42 combinations (versus the published
4,871), and none of the four named venues ever flips significance anywhere on the grid: Chens
Fresh Food Market and Hunts Point Floating Pier stay non-significant in all 42, Amenazel Organic
stays significant in all 42, and the merged Jacob Riis group's own call tracks its 28/42
correctly-grouped pattern (it's only a coherent single entity to test in those 28) without ever
flipping sign or crossing the FDR line. So this choice is robust on both axes checked: the
clustering mechanics the earlier check already covered, and the actual reported conclusions.
See `sensitivity_fragments.py` and `sensitivity_fragments_results.csv` for the full 42-row table.

UPDATE (Note 10): this entire grid -- the 4,845-4,909 survivor band, the "versus the published
4,871" comparison, and "Amenazel Organic stays significant in all 42" -- predates the GLS
temporal-dependence fix adopted below (see "The independence assumption -- FIXED (Note 10...)" in
Known limitations) and has NOT been rerun under it: `sensitivity_fragments.py` still carries its
own pre-Note-10 duplicated `reml_tau2`/`hartung_knapp`, unlike `sensitivity_eligibility_floor_fdr.py`
below, which inherits fixes automatically via its `pool_persistence.py` subprocess call. Given that
Amenazel flips to non-significant under the now-ADOPTED method, this grid's "stays significant in
all 42" claim should be read as a pre-Note-10 snapshot, not a current robustness confirmation --
re-running it under the GLS method is the open follow-up flagged there.

*Eligibility floor.* `sensitivity_eligibility_floor.py` reruns the exact fit_mixing_model_2026_04.py
model spec and CV scheme (same features, same RandomForest hyperparameters, same
income_known_devices sample weighting, same buffered 2000ft grid-cell-blocked GroupKFold(5) --
see "No spatial buffer between CV folds" below) at four other floors for April, so every number
below is directly comparable to the published, buffered floor=50 baseline (mean CV R²=0.6201,
fold range 0.573-0.681). Raising the floor (75, 100, 150) needed no rebuild -- every
spatial/context feature in `build_features_2026_0X.py` is computed on the full POI universe
before the eligibility filter is applied as a final subset, so raising the floor is just
filtering more rows out of the existing floor=50 dataframe. Lowering it to 30 needed an actual
rebuild (venues with 30-49 devices were never in the floor=50 dataframe at all), run via a
patched copy of `build_features_2026_04.py` with the threshold changed -- cheap for the same
reason.

| floor | n venues | CV R² (mean) | fold range |
|---|---|---|---|
| 30  | 31,982 | 0.6440 | 0.620–0.678 |
| 50 (published) | 27,481 | 0.6201 | 0.573–0.681 |
| 75  | 22,832 | 0.6411 | 0.566–0.704 |
| 100 | 18,937 | 0.6627 | 0.645–0.684 |
| 150 | 13,336 | 0.6540 | 0.621–0.678 |

Read this against the buffered baseline, not the pre-Note-7 one this table used to show (an
earlier version of this table reported 0.6927/0.6948/0.6971/0.7093/0.7190 -- the unbuffered
numbers, superseded the same way the rest of the CV figures in this README were by Note 7's
spatial-buffer fix). The honest read of the buffered table is non-monotonic rather than a clean
climb: R² dips at the published floor=50 (0.6201) and at floor=75 (0.6411), relative to floor=30
(0.6440), then climbs to a high at floor=100 (0.6627) before easing slightly at floor=150
(0.6540). There's no cliff and the full range is narrow (0.62-0.66), so this still doesn't read as
50 propping up an otherwise fragile number -- but it no longer supports the earlier, tidier
"flat to a slight steady climb" framing, which was itself an artifact of the unbuffered CV's
optimism. See `sensitivity_eligibility_floor.py` and `sensitivity_eligibility_floor_results.csv`.

*Eligibility floor, FDR-conclusion stability (Note 8).* CV R² only tests whether the floor
changes how well the model fits -- it says nothing about whether the floor changes what the
project actually concludes. `sensitivity_eligibility_floor_fdr.py` reruns the FULL downstream
pipeline at each floor and for all 8 periods -- the same buffered PRIMARY CV above, Note 5's
Dirichlet-multinomial visitor-sampling-variance bootstrap, the 15-replicate model-refit/prediction-
variance bootstrap, and `pool_persistence.py`'s own REML+Hartung-Knapp pooling and
Benjamini-Hochberg FDR control -- at floors 30, 75, 100, 150, compared against the published
floor=50's own RAW (pre-fragment-consolidation) output, so this isolates the floor's own effect
from fragment-consolidation (already stress-tested separately, above). This is a materially bigger
undertaking than the CV-only check (real RandomForest refits at every floor and period, not just a
clustering or CV pass), so it's its own companion script rather than folded into
`sensitivity_eligibility_floor.py` -- same reasoning that split `estimate_model_prediction_variance.py`
and `estimate_visitor_sampling_variance_dm.py` out of `fit_mixing_model_2026_0X.py` as their own
required pipeline steps.

| floor | n venues (≥1 month) | n tested (≥2 months) | FDR survivors | survivor rate |
|---|---|---|---|---|
| 30  | 40,050 | 37,227 | 5,761 | 15.48% |
| 50 (published) | 34,943 | 31,864 | 4,832 | 15.16% |
| 75  | 29,661 | 26,448 | 3,995 | 15.11% |
| 100 | 25,114 | 21,949 | 3,037 | 13.84% |
| 150 | 18,202 | 15,383 | 1,983 | 12.89% |

The survivor rate declines close to monotonically as the floor rises, from 15.48% at floor=30 to
12.89% at floor=150 -- an intuitive pattern (a stricter floor shrinks the tested family down to
fewer, individually larger venues, and does not by itself make the underlying effects easier to
detect) and not a sign that 50 was an unusually favorable or unfavorable cut point; it sits in the
middle of the range, not at either extreme. The three non-grouped named venues move more
individually than the aggregate rate does: **Chens Fresh Food Market**, never significant at the
published floor=50 (`q_bh_hk`=0.1146) or above, IS significant at floor=30 (`q_bh_hk`=0.0078,
`mu_hk`=-0.289, 8 eligible months instead of 5) -- lowering the floor recovers two additional
borderline-eligible months for this specific venue, and with them a real, if less certain,
negative effect becomes detectable. **Amenazel Organic** survives at floors 30 and 50
(`q_bh_hk`=0.0075 and 0.0321) but loses testability as the floor rises: non-significant at 75
(`q_bh_hk`=0.078, still only 3 eligible months), and untestable at all (fewer than 2 eligible
months) at 100 and 150 -- a power loss from a shrinking eligible-month count, not a reversal of
the effect's sign or direction. **Hunts Point Floating Pier** stays non-significant at every floor
tested, consistent with its published read (a real but not-yet-FDR-significant effect, 2 eligible
months at most floors). So the floor choice does move individual conclusions for specific
venues -- most notably, it determines whether Chens clears the bar at all -- but it does so
through the mechanical, explainable channel of how many eligible months a venue has, not through
any instability in the pooling method itself. See `sensitivity_eligibility_floor_fdr.py` and
`sensitivity_eligibility_floor_fdr_results.csv`.

**A staging bug this check surfaced and fixed, with no effect on the published headline number:**
building the table above meant rerunning `pool_persistence.py` end to end against the buffered
Note 7 outputs, which surfaced a defect in the already-committed Note 7 delivery: the repo's
working copy of `pool_persistence.py` had been silently overwritten, during Note 7's own staging,
by a pre-Note-5 copy missing the Dirichlet-multinomial overdispersion correction entirely (no
`VS_VAR_PATHS` merge at all) -- a staging-cache mixup, not a change anyone made on purpose. That
stale script is what actually produced the committed `pooled_persistence_2026.{pkl,csv}` (the RAW,
pre-consolidation file): 5,593 survivors under the broken script versus the correct 4,832 once
fixed and rerun, a real difference. The published HEADLINE number this README reports throughout
-- the **4,871**-survivor CONSOLIDATED figure from `consolidate_fragments.py`, and every
named-venue number quoted under "Persistence is now a formal pooled estimate" and the four NAMED
venue bullets below -- was never affected: `consolidate_fragments.py`'s own copy already had the
Note 5 fix, and a full independent rerun against freshly-regenerated, buffer-consistent
visitor-sampling-variance inputs reproduced the existing 4,871 and every individual venue's
`mu_hk`/`q_bh_hk` to floating-point precision, confirmed directly rather than assumed. Both
`pool_persistence.py` and the RAW `pooled_persistence_2026.{pkl,csv}` have been corrected and
recommitted alongside this Note 8 extension.

*Fragment-consolidation device dedup, cross-checked against the vendor's own duplicate/overlap
flags (Note 9, a seventh round of external review).* `consolidate_fragments.py` sums each merged
group's q1-q5_devices across its member fragments (see Known limitations, POI fragmentation) --
a real concern raised in external review is that this could double-count if the vendor's own
visit-to-POI attribution isn't fully exclusive, or under-count if genuinely duplicate catalog
entries are missed by the 75ft/name-Jaccard>=0.4 thresholds. The vendor's raw
`analysis_pois_<PERIOD>.csv` already carries three authoritative signals for exactly this --
`parent_placekey` (explicit parent/child POI relationships), `shared_candidate`, and
`polygon_class` (`OWNED_POLYGON`/`SHARED_POLYGON`) -- present in all 8 periods' raw exports but
previously unused anywhere in the pipeline. `fragment_dedup_crosscheck.py` re-joins these onto the
union of eligible venues and checks both directions. **Double-count risk:** of 535 parent/child
pairs where both placekeys are eligible, only 2 are currently merged into the same fragment group
(both benign-looking -- "Torah Temimah Talmudical Seminary" x2 and "Dyckman Farmhouse/House
Museum" -- not a double-count in any real sense), and the other 533 are correctly kept separate;
double-counting via parent/child POIs is not a material problem in this pipeline. **Under-count
risk:** of 10,477 SHARED_POLYGON-flagged eligible venues, 97.3% are not in any fragment group
(expected -- most shared-polygon venues are genuinely distinct businesses sharing a building, not
duplicates); widening the search to 150ft/same-category/any name-token overlap among the unmerged
ones surfaced 291 near-duplicate candidates, none involving any of the four named venues or the
Jacob Riis cluster. Recomputing the borderline candidates with the production `name_tokens`
function (not a simplified stand-in) found a real, if modest, gap: `name_tokens` kept a trailing
possessive `'s` as part of its token (ASCII apostrophe only -- the regex `[a-z0-9']+` doesn't
split on it), so "Lorenzo's Pizza" tokenized to `{"lorenzo's", "pizza"}` while "Lorenzo Pizza Of
New York" tokenized to `{"lorenzo", "pizza", "new", "york"}` -- `lorenzo's` and `lorenzo` never
matched, holding their Jaccard below the 0.4 threshold (0.25) despite being the same place. Three
such pairs were found this way (the other two: "Alex's Deli Of Ny Corp"/"Alex Deli Grocery" and
"Alex's Tacos"/"Alex Tacos Deli"). ADOPTED FIX: `name_tokens` now strips a trailing possessive `'s`
(both the ASCII apostrophe and the curly `'` a handful of vendor records use) before tokenizing,
so "Lorenzo's" tokenizes the same as "Lorenzo." Rerunning the full citywide pass with this fix
merges those 3 pairs (265 groups/534 placekeys -> 275 groups/554 placekeys) and drops the
cross-check's own near-dupe candidate count from 291 to 288 -- exactly the 3 pairs it closed, no
side effects elsewhere. Effect on the full consolidated table, checked directly rather than
assumed: the ADOPTED REML+Hartung-Knapp FDR survivor count is unchanged at **4,871** (same
31,645-venue tested family, down from 31,652 only because 10 more fragments merged into existing
venues rather than counting separately); around 30 individual venues' `q_bh_hk` move across the
q<=0.05 boundary in either direction, but nearly all of those are the SAME entity shifting
`fragment_groups_report.csv` group-ID numbers (new groups inserted mid-sequence renumber later
ones) with a third- or fourth-decimal q-value change from the BH procedure's family-size shift, not
a real change in that venue's own evidence. None of the four named venues (Chens Fresh Food
Market, Amenazel Organic, Hunts Point Floating Pier, or the Jacob Riis cluster/Jacob Riis Beach)
move at all -- confirmed by direct before/after comparison, not inferred. See
`fragment_dedup_crosscheck.py`, `fragment_dedup_crosscheck_parent_child.csv`, and
`fragment_dedup_crosscheck_near_dupes.csv`.

*Neighborhood income diversity (omitted-variable check).* `sensitivity_neighborhood_diversity.py`
tests a specific concern raised in external review: the contextual baseline conditions on a block
group's mean income (`income_proxy_v2`) and racial composition, but not on how spread out that
income distribution is -- two neighborhoods can share a mean and differ sharply in income
diversity, and if that spread itself predicts mixing, a venue's residual could partly reflect the
neighborhood's omitted diversity rather than anything about the venue itself. The predictor
(`neighborhood_income_diversity_v2`, normalized entropy of a block group's 16 ACS income
brackets) had already been computed in `census_new.pkl` -- explored early in
`income_mixing_model.ipynb` but never carried into `build_features_2026_0X.py`'s feature
selection, so no production model has ever used it. Adding it back in and rerunning the exact
fit_mixing_model spec/CV for July 2026: CV R² moves from 0.7191 to 0.7193 (smaller than the 0.032
spread between folds), the new feature ranks 19th of 21 numeric features by RF importance (behind
even the plain `income_proxy_v2` it refines), and none of the four named venues' residual
z-scores move by more than 0.03 -- Amenazel +0.011, Chens +0.008, Hunts Point Floating Pier
-0.0002, both Jacob Riis fragments -0.034. It's moderately correlated with the existing
local-income features (-0.38 to -0.45, not extreme enough that collinearity alone would explain
the low importance), consistent with those finer-grained, venue-centered local-income measures
already absorbing most of what neighborhood-level income spread would add. Tested on July 2026
only so far -- reassuring, not exhaustive; see `sensitivity_neighborhood_diversity.py` and its
two output CSVs for the full numbers, and rerun with a different `MONTH` env var to extend the
check to more periods.

## Repo structure

```
income_mixing_model.ipynb       Main annotated notebook: feature engineering (including the
                                 ACS income-proxy construction), model fit, bootstrap, and the
                                 full methodology writeup in markdown cells alongside the code.

geocode_venues_to_cbg.py        Assigns each venue its own Census block group by spatially
                                 joining its lat/long to the TIGER/Line block-group shapefile.
                                 Run once per month, before build_features_2026_0X.py. Produces
                                 cbg_geoid_2026_0X.csv. Requires geopandas and shapely.

build_features_2026_0{4,5,6,7}.py Standalone script version of the feature pipeline, one per
                                 replication month, used for the cross-month check described
                                 in the notebook. Also merges in q1_devices..q5_devices from
                                 the FULL (non-slim) analysis_pois_2026_0X.csv export, needed
                                 for fit_mixing_model_2026_0X.py's visitor-sampling bootstrap.
                                 Produces model_ready_2026_0X.pkl.

fit_mixing_model_2026_0{4,5,6,7}.py   Fits the final model against income_mixing_score, weighted
                                     by device count, CV'd with a ~2000ft grid-cell-blocked
                                     GroupKFold (primary) plus a random KFold reported only as
                                     a secondary comparison, runs TWO 30-iteration bootstraps
                                     (model-refit resampling and visitor-sampling resampling of
                                     each venue's own q1-q5 device counts -- see Known
                                     limitations), saves stable top/bottom-0.5%-per-borough
                                     lists with both stability figures attached. FIXED (Note 7,
                                     a fifth round of external review -- see Known limitations,
                                     No spatial buffer between CV folds): the PRIMARY CV now
                                     excludes, from each fold's training set, any point within
                                     BUFFER_FT=1500ft of that fold's test points -- not just
                                     points sharing a test point's own grid cell -- to close the
                                     near-boundary leakage the plain grid-cell blocking left open.

fit_balance_model_2026_0{4,5,6,7}.py  Fits the same feature set against balance_score as a
                                     robustness check (see Known limitations -- this score
                                     is not independent of income_mixing_score), also weighted
                                     by device count and CV'd the same grid-cell-blocked way.

verify_library_categories.py    Spot-checks vendor-tagged "library" venues against NYC's
                                 official public library locations by nearest-neighbor distance
                                 -- see "Category reliability" under Known limitations. Reads
                                 NYC_Library_Locations.csv plus a venues_2026_0X_slim.csv-style
                                 export; prints matches/mismatches, doesn't change any file.

estimate_overdispersion.py      REVISED after a third round of external review (Note 5 -- see
                                 Known limitations): cross-period check, now over all 8 periods
                                 instead of the original 3, of the independence assumption behind
                                 the visitor-sampling bootstrap. Finds real evidence FOR
                                 extra-multinomial clustering (corrected VIF~1.48, reversing the old
                                 3-month 0.91 reading) and now saves an ADOPTED S_hat/rho_hat (plus
                                 its honest sensitivity range) to overdispersion_estimate.pkl for
                                 estimate_visitor_sampling_variance_dm.py to use -- no longer just a
                                 print-and-stop diagnostic. Run once all 8 periods'
                                 model_ready_<PERIOD>.pkl files exist (needs the full balanced panel
                                 to measure serial correlation honestly).

estimate_visitor_sampling_variance_dm.py  REQUIRED pipeline step (Note 5 fix): re-estimates every
                                 eligible venue's visitor-sampling variance using a Dirichlet-
                                 multinomial-corrected bootstrap instead of fit_mixing_model_2026_0X.
                                 py's plain-multinomial one, using estimate_overdispersion.py's
                                 adopted S_hat. Cheap (~1-2 minutes per period -- no model refit,
                                 just resampling). Run once per period via MONTH env var, AFTER
                                 estimate_overdispersion.py and BEFORE pool_persistence.py/
                                 consolidate_fragments.py, which now read resid_vs_mean/resid_vs_var
                                 from here instead of from fit_mixing_model_2026_0X.py's output.
                                 Reads model_ready_<MONTH>.pkl, results_mixing_<MONTH>.pkl, and
                                 overdispersion_estimate.pkl; produces
                                 visitor_sampling_variance_dm_<MONTH>.csv.

estimate_model_prediction_variance.py  REQUIRED pipeline step (promoted from a one-off
                                 sensitivity check after external review -- see Known limitations,
                                 Model prediction variance): spatial-block bootstrap (same design as
                                 sensitivity_spatial_block_bootstrap.py) that estimates every eligible
                                 venue's model-refit/prediction variance for one period, so
                                 pool_persistence.py/consolidate_fragments.py can fold it into their
                                 pooling weight instead of using visitor-sampling variance alone.
                                 ~10-15 minutes per period (reruns the full grid-cell-blocked
                                 GroupKFold(5) 15 times). Run once per period via MONTH env var,
                                 BEFORE pool_persistence.py/consolidate_fragments.py for that period.
                                 Reads model_ready_<MONTH>.pkl; produces
                                 model_prediction_variance_<MONTH>.csv. FIXED (Note 7): each
                                 replicate's GroupKFold(5) now applies the same BUFFER_FT=1500
                                 spatial buffer as fit_mixing_model_2026_0X.py's PRIMARY CV,
                                 recomputed fresh per replicate since the block bootstrap can
                                 duplicate a cell's rows.

estimate_temporal_dependence.py  REQUIRED pipeline step (Note 10 fix -- an eighth round of
                                 external review): direct, all-lag test of the pooling's
                                 implicit independent-months assumption on the real pooled
                                 residual r_vm (not the related-but-different quintile-share
                                 check behind estimate_overdispersion.py), on the same 7,838-venue
                                 balanced panel. Finds a real, one-lag-only excess correlation
                                 (+0.156 raw / +0.180 standardized at lag 1, fading to at-or-below
                                 the demeaning-artifact benchmark by lag 2+) and saves the ADOPTED
                                 single-point rho (+0.1683, mean of the two lag-1 estimates) to
                                 temporal_dependence_estimate.pkl for pool_persistence.py/
                                 consolidate_fragments.py's GLS covariance to use. Run once all 8
                                 periods' results_mixing_<PERIOD>.pkl,
                                 visitor_sampling_variance_dm_<PERIOD>.csv, and
                                 model_prediction_variance_<PERIOD>.csv files exist (needs the full
                                 balanced panel to measure serial correlation honestly, same
                                 requirement as estimate_overdispersion.py). Run BEFORE
                                 pool_persistence.py/consolidate_fragments.py. See Known
                                 limitations, "The independence assumption -- FIXED (Note 10...)".

pool_persistence.py             Pools each venue's per-month residual (the fixed-reference point
                                 estimate from fit_mixing_model_2026_0X.py above, pooled against an
                                 overdispersion-corrected sampling variance -- Note 5) across however
                                 many months it cleared the eligibility floor, via a REML +
                                 Hartung-Knapp random-effects meta-analysis over var_combined =
                                 visitor-sampling + model-refit variance, now with an explicit GLS
                                 covariance structure for calendar-adjacent months' measured lag-1
                                 serial dependence (rho=+0.1683, fixed -- Note 10, the ADOPTED
                                 method and weight; DerSimonian-Laird is kept only as a
                                 diagnostic/comparison -- see Known limitations), then applies a
                                 Benjamini-Hochberg FDR correction across venues with 2+ eligible
                                 months -- see Known limitations. Reads every month's
                                 results_mixing_2026_0X.pkl, model_prediction_variance_<MM>.csv,
                                 (Note 5) visitor_sampling_variance_dm_<MM>.csv, and (Note 10)
                                 temporal_dependence_estimate.pkl; produces
                                 pooled_persistence_2026.{pkl,csv}. Superseded for most purposes by
                                 consolidate_fragments.py below, which reruns this same pooling on
                                 fragment-consolidated entities -- kept because it's also the
                                 pre-consolidation sanity check referenced in Known limitations.

consolidate_fragments.py        Citywide pass that groups POI-catalog fragments representing the
                                 same real place (spatial proximity within 75ft + name-token
                                 Jaccard similarity + same top_category; see Known limitations --
                                 POI fragmentation) and reruns the persistence pooling above
                                 (including Note 10's GLS temporal-dependence correction) on the
                                 merged entities, including Note 5's overdispersion correction --
                                 singletons pick it up via the same visitor_sampling_variance_dm_
                                 <MM>.csv merge as pool_persistence.py, while merged fragment groups
                                 get their OWN Dirichlet-multinomial bootstrap re-run on their
                                 combined quintile counts (not covered by that per-venue CSV), using
                                 the same adopted S_hat from overdispersion_estimate.pkl. A merged
                                 group's observed score (Note 6 -- see Known limitations, "The
                                 merged-group score's Jensen's-inequality asymmetry") is now a
                                 device-weighted average of each fragment's OWN income_mixing_score,
                                 matching merged_expected_cv's own combination rule, instead of the
                                 entropy of the fragments' pooled/summed counts; the merged group's
                                 own bootstrap above was updated to match (resamples each fragment's
                                 own counts independently and averages their scores, rather than
                                 pooling counts before scoring). Produces fragment_groups_report.csv
                                 (275 groups, 49 flagged as exact-name matches worth a manual glance
                                 -- grouping unaffected by Note 5 or Note 6; UPDATED by Note 9's
                                 possessive-apostrophe normalization in name_tokens, see Validation
                                 checks already run above) and
                                 consolidated_pooled_persistence_2026.{pkl,csv} -- this is the
                                 canonical, current persistence table; prefer it over
                                 pooled_persistence_2026.{pkl,csv} above.

presence_bookkeeping.py         Per consolidated entity and month, distinguishes "absent,"
                                 "present but under the eligibility floor," "present and eligible
                                 but excluded by the encounter_type category filter," "...but no
                                 CBG income match," "...but only rescued over the floor by
                                 fragment-combining after the fact, so never individually fitted,"
                                 and "pooled" -- so a venue missing from a month's pooled table
                                 reads as what it actually is rather than "not mixed." See Known
                                 limitations. Produces presence_bookkeeping_2026.csv.

geo_temporal_holdout_validation.py   Tests whether the model generalizes across TIME, not just
                                 across space within one month -- see Validation checks already
                                 run above. Trains on three months pooled, predicts the fourth,
                                 with venues grid-cell-blocked out of training in ALL months (not
                                 just the held-out one) so a test venue was never seen anywhere.
                                 Reads all four months' model_ready_2026_0X.pkl; produces
                                 geo_temporal_holdout_results.csv.

sensitivity_fragments.py        Stress-tests the two consolidation thresholds in
                                 consolidate_fragments.py (75ft, name-token Jaccard≥0.4) across a
                                 42-combination grid -- see Validation checks already run above.
                                 Reruns the clustering step, then (Note 8) re-merges and reruns
                                 the full REML+Hartung-Knapp pooling/BH-FDR at every grid point --
                                 not just a clustering/CV-correctness check, but whether the
                                 threshold choice moves the actual FDR survivor count and the four
                                 named venues' significance calls. Reads all 8 periods'
                                 model_ready_2026_0X.pkl plus each period's
                                 visitor_sampling_variance_dm_<PERIOD>.csv and
                                 model_prediction_variance_<PERIOD>.csv; produces
                                 sensitivity_fragments_results.csv.

fragment_dedup_crosscheck.py     Cross-checks consolidate_fragments.py's clustering against the
                                 vendor's OWN authoritative duplicate/overlap signals -- Note 9, a
                                 seventh round of external review, on the concern that summing
                                 q1-q5_devices across fragment members (see Known limitations, POI
                                 fragmentation) could double- or under-count if the vendor's own
                                 visit attribution isn't as exclusive as assumed. Re-joins
                                 parent_placekey/shared_candidate/polygon_class from the raw
                                 analysis_pois_<PERIOD>.csv (present in all 8 periods' raw exports,
                                 but only build_features_2026_04.py currently carries them into
                                 model_ready -- a minor, separate inconsistency, not yet fixed) onto
                                 the union of eligible venues, then checks (a) parent/child pairs
                                 that are BOTH eligible against fragment_groups_report.csv's actual
                                 grouping (double-count risk), and (b) SHARED_POLYGON/shared_candidate
                                 venues NOT in any group, widened to 150ft/any name-token overlap, for
                                 plausible same-place duplicates the 75ft/Jaccard>=0.4 production
                                 thresholds missed (under-count risk). See Validation checks already
                                 run above for the findings and the apostrophe-normalization fix this
                                 surfaced. Reads model_ready_<PERIOD>.pkl, analysis_pois_<PERIOD>.csv,
                                 and fragment_groups_report.csv; produces
                                 fragment_dedup_crosscheck_parent_child.csv and
                                 fragment_dedup_crosscheck_near_dupes.csv.

sensitivity_eligibility_floor.py  Stress-tests the 50-device eligibility floor in
                                 build_features_2026_0X.py at four other floors (30, 75, 100,
                                 150) for April -- see Validation checks already run above.
                                 Reruns the exact fit_mixing_model_2026_04.py model spec and
                                 buffered grid-cell-blocked CV at each floor so results are
                                 directly comparable to the published, buffered baseline. Reads
                                 the floor=50 model_ready_2026_04.pkl plus a separately-rebuilt
                                 floor=30 version; produces
                                 sensitivity_eligibility_floor_results.csv.

sensitivity_eligibility_floor_fdr.py  Companion to sensitivity_eligibility_floor.py (Note 8, a
                                 sixth round of external review): reruns the FULL downstream
                                 pipeline -- buffered PRIMARY CV, Note 5's DM visitor-sampling
                                 bootstrap, the 15-replicate model-refit/prediction-variance
                                 bootstrap, and pool_persistence.py's REML+Hartung-Knapp/BH-FDR
                                 pooling -- at floors 30, 75, 100, 150 across all 8 periods, since
                                 CV R² alone doesn't say whether the floor moves the actual FDR
                                 conclusions. Compares against the published floor=50's own RAW
                                 (pre-consolidation) output. Real RandomForest refits at every
                                 floor/period, materially more expensive than the CV-only check
                                 above -- see Validation checks already run above for the full
                                 numbers. Reads model_ready_<PERIOD>.pkl/overdispersion_estimate.pkl
                                 (published floor=50 outputs) plus build_features_<PERIOD>.py (for
                                 the floor=30 rebuild); produces
                                 sensitivity_eligibility_floor_fdr_results.csv.

sensitivity_neighborhood_diversity.py  Tests whether adding a block group's income-distribution
                                 diversity (computed in census_new.pkl, but never carried into
                                 the production feature set) changes CV R² or any named venue's
                                 residual -- see Validation checks already run above. Reruns the
                                 exact fit_mixing_model_2026_0X.py model spec/CV with the one
                                 feature added. Reads model_ready_2026_0X.pkl plus census_new.pkl;
                                 produces sensitivity_neighborhood_diversity_results.csv and
                                 sensitivity_neighborhood_diversity_named_venues.csv.

sensitivity_spatial_block_bootstrap.py  Corrects the model-refit stability bootstrap in
                                 fit_mixing_model_2026_0X.py, which resamples individual venues
                                 (rows) and predicts in-sample rather than respecting the
                                 project's own spatial blocking -- see Known limitations above.
                                 Resamples grid CELLS with replacement and reruns the real
                                 grid-cell-blocked GroupKFold(5) inside each replicate, so every
                                 re-flagging prediction stays genuinely held out. Reads
                                 model_ready_2026_07.pkl and results_mixing_2026_07.pkl (for the
                                 real reference flags); produces
                                 sensitivity_spatial_block_bootstrap_results.csv.
```

Run order per period: `geocode_venues_to_cbg.py` -> `build_features_2026_0X.py` ->
`fit_mixing_model_2026_0X.py` and `fit_balance_model_2026_0X.py` -> `estimate_model_prediction_
variance.py` (REQUIRED since the model-refit-variance fix -- see Known limitations; reads that
same period's `model_ready_<MONTH>.pkl`, ~10-15 minutes per period). The notebook is
self-contained for a single period end to end. `estimate_overdispersion.py` is now ALSO REQUIRED
(Note 5 -- see Known limitations): run it once all 8 periods' `model_ready_<PERIOD>.pkl` files
exist (it needs the full balanced panel to measure serial correlation honestly), producing
`overdispersion_estimate.pkl`'s adopted S_hat. Then run `estimate_visitor_sampling_variance_dm.py`
per period (~1-2 minutes per period) -- also REQUIRED, and must come after
`estimate_overdispersion.py` since it reads that S_hat. Once every period's
`results_mixing_<MONTH>.pkl`, `model_prediction_variance_<MONTH>.csv`, AND
`visitor_sampling_variance_dm_<MONTH>.csv` exist (i.e. `fit_mixing_model_2026_0X.py`,
`estimate_model_prediction_variance.py`, and `estimate_visitor_sampling_variance_dm.py` have all
been rerun for each period in `pool_persistence.py`'s `PERIODS` tuple), run
`estimate_temporal_dependence.py` (Note 10 -- also REQUIRED, and needs the same full 8-period
balanced panel as `estimate_overdispersion.py`), producing `temporal_dependence_estimate.pkl`'s
ADOPTED rho. Then run `consolidate_fragments.py` (which calls the same pooling logic as
`pool_persistence.py` internally, so running `pool_persistence.py` separately first is optional,
not a prerequisite), then `presence_bookkeeping.py`, which reads `consolidate_fragments.py`'s
output.

Regression is weighted by `income_known_devices` (device count) when fitting the RandomForest,
since `income_mixing_score` has a documented finite-sample bias below ~150 devices (see the
diagnostic cell in the notebook) that the 50-device eligibility floor alone doesn't correct.

## Data sources (not included in this repo)

- **Mobile location / visit data**: an anonymized, aggregated mobility panel (visit-level
  records matched to POIs via Placekey, with each device's home location generalized to a
  Census block group). The panel is Veraset's mobile-location product, accessed through a
  Dewey Data subscription. Dewey's subscriber terms permit publishing "summary insights
  derived from the Licensed Data" while restricting redistribution of the raw data itself,
  with a required acknowledgment for any derived publication -- credited here to Dewey Data
  and Veraset accordingly. POI category/name/Placekey metadata is SafeGraph's, joined in by
  Veraset, and warrants its own separate attribution. *[Note for repo maintainer: this
  reflects Dewey's and Veraset's public terms pages, not a lawyer's review of the actual
  signed agreement -- confirm against your own order terms (or ask Dewey directly) before
  this goes public.]*
- **POI metadata**: Placekey-linked venue attributes (name, category, brand, lat/long).
- **Census block-group boundaries**: `tl_2025_36_bg` (2025 TIGER/Line block-group shapefile for
  NY State), public data from the Census Bureau. Each venue's own block group is assigned by
  spatially joining its lat/long to this shapefile (`geocode_venues_to_cbg.py`) — a venue's
  borough, neighborhood income, and race context all come from where the venue itself sits,
  not from where its visitors happen to live.
- **Census ACS estimates**: block-group-level income and race/ethnicity tables (variable codes
  like `SE_A14001B_*` and `SE_A04001_*` come from a Social Explorer ACS extract). The
  cumulative household-income-bracket table (`SE_A14001B_*`) is used to build `income_proxy_v2`
  instead of the single median-income variable (`SE_A14006_001`), which is suppressed for
  ~20% of NYC block groups, non-randomly by borough — see the notebook for the full rationale.
- **MTA subway stations**: `MTA_Subway_Stations.csv`, public data from data.ny.gov / the MTA.
- **NYC public library locations**: `NYC_Library_Locations.csv`, public data from NYC Open Data
  (the combined NYPL/Brooklyn Public Library/Queens Public Library point file), used only for
  the category spot-check in `verify_library_categories.py` -- not a model input.

None of the raw or lightly-transformed vendor mobility data is included or should be added to
this repo. Before publishing, confirm the vendor's license actually permits sharing the
derived, venue-level outputs (placekey, scores, residuals) that these scripts produce — some
data licenses restrict redistribution even of aggregated results.

## What's not here

- Raw or per-device visit records. `build_features_*.py` and the notebook expect an
  already-aggregated `analysis_pois_2026_0X_slim.csv` (venue-level, including each venue's own
  lat/long), the FULL (non-slim) `analysis_pois_2026_0X.csv` export of the same month (used only
  for its venue-level `q1_devices`..`q5_devices` columns, which the slim export drops -- this is
  still a venue-level aggregate, not a device-level table, just a wider one), and a
  `cbg_geoid_2026_0X.csv` (venue -> its own geographic Census block group, produced by
  `geocode_venues_to_cbg.py` from a spatial join against the TIGER/Line block-group shapefile --
  never derived from visitors' home locations; an earlier `cbg_lookup_2026_0X.csv`, the
  visitor-derived version this replaced, is obsolete and should not be reintroduced -- see Known
  limitations).
- The raw ACS extract and MTA station file — public data, but not redistributed here; fetch
  your own copies (see Data sources) and point `DATA_DIR` at them.
- Any file path specific to one machine. Every script reads its paths from environment
  variables with sensible defaults, never a hardcoded absolute path -- most read `DATA_DIR`/
  `OUTPUT_DIR` (defaulting to `./data`/`./output`); the scripts that need every month's output at
  once (`pool_persistence.py`, `consolidate_fragments.py`, `presence_bookkeeping.py`,
  `geo_temporal_holdout_validation.py`, `sensitivity_fragments.py`,
  `sensitivity_eligibility_floor.py`) read `OUTPUT_DIR`/`DATA_DIR` too, on the assumption that
  each month's `build_features_2026_0X.py`/`fit_mixing_model_2026_0X.py` run was pointed at the
  SAME shared `DATA_DIR`/`OUTPUT_DIR` each time (producing differently-month-named files into one
  shared pair of folders, not one folder per month) -- set them to wherever your inputs live. (An
  earlier version of this repo had three scripts -- `pool_persistence.py`, `consolidate_fragments.py`,
  `presence_bookkeeping.py` -- with absolute `/home/claude/work/...`-style paths hardcoded despite
  this claim; caught by external review and fixed. If you find another one, it's a bug, not an
  exception to this rule.)

## Known limitations

- The model explains roughly 73-75% of the variance in venue-level mixing across the three
  replicated months; the rest is unexplained, which is normal for social/behavioral data but
  means "expected" is a statistical baseline, not a claim about what should happen at a given
  venue.
- The "less mixed than expected" outliers are consistently somewhat more statistically stable
  than the "more mixed" ones across all four months.
- Three of the four replication months (May-July 2026) are summer; a fourth, April 2026, was
  added specifically to test whether one of the headline venues (a public beach) is a
  warm-weather-specific draw rather than a year-round effect -- see the persistence bullet below
  for the answer (it is: `I2`=96.3% for that venue, meaning most of its month-to-month movement is
  real seasonal variation, not sampling noise). No fall/winter month has been tested yet.
- **Device counts vs. real foot traffic**: `income_known_devices` measures how many visitors the
  mobile-location panel captured and could confidently trace to a home neighborhood -- not how
  many people actually visited. A venue showing 89 income-known devices in a month plausibly saw
  far more real foot traffic; the panel only ever samples a fraction of it. That distinction
  matters because a modest device count means the income-mix score rests on a thinner slice of a
  venue's true visitor base, which makes it more sensitive to a handful of unusual devices --
  regardless of how busy the place actually is. Many flagged venues sit just above the 50-device
  eligibility floor, and that's why we treat individually-named small businesses as leads worth
  on-the-ground verification rather than settled findings; the neighborhood- or corridor-level
  pattern is the safer unit of analysis in most cases.
- **Panel representativeness**: neither Dewey nor Veraset's own documentation discloses what
  share of the population their location panel captures or how representative it is by
  income, race, or age -- that isn't something this project can audit directly. (Veraset does
  disclose panel *size* and rough geographic coverage -- roughly 70% of census block groups,
  ~90% of populated ones, varying month to month, with the US panel averaging 24.1 million
  devices so far in 2026, down from 50.1 million in 2024 -- but not its demographic skew.) The
  closest available evidence on that skew is an independent, peer-reviewed analysis of
  SafeGraph data (the same general category of commercial POI-visit data Veraset's product
  descends from, though not an audit of this exact panel): Li, Z., Ning, H., Jing, F., & Lessani, M.N. (2024). "Understanding
  the Bias of Mobile Location Data Across Spatial Scales and Over Time: A Comprehensive Analysis
  of SafeGraph Data in the United States." *PLOS ONE*. It found households earning under $50k are
  significantly underrepresented (most pronounced in the South and Midwest), the $50k-$100k band
  is the best-represented income group, Hispanic populations are notably underrepresented in
  several states, and representation of both low-income and minority populations is worse in
  rural counties than urban ones. This doesn't tell us the size of the effect specifically in
  NYC, and the direction of the resulting distortion is not uniform across venues -- it depends
  on a venue's true composition, not a blanket "everything reads low." At a venue that is
  genuinely used across income levels, undercounting the low end mostly shaves visibility off an
  already-mixed crowd, so the observed score likely *understates* true mixing. At a venue whose
  true visitor base is dominated by one lower-income group with only a thin overlay of other
  visitors, the same undercounting can make that thin overlay look proportionally larger than it
  is, manufacturing an appearance of mixing that may not be real. Practically: the "more mixed
  than expected" finding for a place with independent, non-dataset evidence of broad, cross-class
  use (Jacob Riis Beach's century of documented public, cross-class history) sits in the first,
  lower-risk case; an individually-named small business flagged as surprisingly mixed with no
  such outside corroboration sits in the second, higher-risk case, and is exactly the kind of
  claim the eligibility-floor limitation above already says needs on-the-ground verification
  before naming. The same mechanism cuts the other way for "less mixed than expected" findings:
  undercounting low-income visitors tends to inflate apparent mixing at low-income-dominated
  venues, which would make it *harder*, not easier, for a genuinely segregated venue to clear the
  bar as an outlier -- consistent with the pattern already noted above that "less mixed" outliers
  have been more statistically stable than "more mixed" ones across all three months.
- **Cross-validation is now spatially blocked, which changed the reported numbers**: an earlier
  version of this pipeline validated with a random row-level 5-fold split. Because nearby venues'
  residuals are correlated (see "Location vs. category" below, r≈0.52-0.61), a random split let a
  venue's own physical neighbors leak across the train/test boundary, inflating apparent
  out-of-sample R². Switching to a ~2000ft grid-cell-blocked `GroupKFold` (see "What the model
  does" above) is a stricter, more honest validation, and it changed the headline numbers: primary
  R² fell from roughly 0.75-0.77 (random-fold) to roughly 0.72-0.73 (grid-cell) across the three
  months, and the 30-iteration bootstrap-stability check -- computed downstream of these residuals
  -- runs lower too. In June, for example, mean stability among TOP ("more mixed than expected")
  outliers fell from 68.9% to 56.7%, and among BOTTOM ("less mixed than expected") outliers from
  88.7% to 81.0%; the same direction of change held in May and July. The project's specific named
  findings were checked directly against this stricter validation and hold up well under it: Jacob
  Riis Beach/Park (as a fragment complex -- see POI fragmentation below), Chens Fresh Food Market,
  Amenazel Organic, and Hunts Point Floating Pier all remain flagged at or near 100% stability in
  every month tested under the corrected, spatially-blocked residuals and this resampling scheme.
  (A second, different resampling scheme -- visitor-sampling stability, below -- tells a less
  uniform story for a couple of these venues; see that bullet for the real numbers. A THIRD issue,
  caught later in external review, affects this specific 100%-stability number for Amenazel
  Organic -- the "30-iteration bootstrap-stability check" itself, not just the residuals it's
  computed from, turned out to have its own spatial-dependence problem; see the
  spatial-block-bootstrap bullet in **Known limitations** below for the corrected number.)
- **Persistence is now a formal pooled estimate, not a percentage-of-30-iterations count, and it
  resolves the earlier July/Jacob-Riis ambiguity**: an earlier version of this section reported
  "visitor-sampling stability" as a raw percentage of bootstrap iterations in which a venue
  re-cleared an arbitrary top/bottom-0.5%-per-borough percentile cutoff, computed separately per
  month with no principled way to combine months into one honest estimate -- which is why a venue
  could look like "93.3% stable in May, a coin flip in July" with no way to say what its *actual*
  persistent effect was. That's been replaced by the pooling in `pool_persistence.py` /
  `consolidate_fragments.py` (see "What the model does" above): each venue's per-month OBSERVED
  spatial-CV residual, measured against one FIXED out-of-fold expectation, is pooled across its
  eligible months with a REML + Hartung-Knapp random-effects meta-analysis (the ADOPTED method;
  DerSimonian-Laird's `mu_re`/CI/`tau2`/`I2` are kept only as a diagnostic/comparison -- see the
  DerSimonian-Laird caveat below for why) into one pooled estimate (`mu_hk`), confidence interval,
  and a direct measure of how much of its apparent month-to-month movement is genuine
  (`tau2_reml`) rather than visitor-sampling noise, then run through a Benjamini-Hochberg FDR
  correction across every venue tested in 2+ months (31,645 such venues citywide, current
  8-period pooling Dec 2025-Jul 2026; **3,233** survive the ADOPTED REML+Hartung-Knapp FDR control
  at q<=0.05 over `var_combined` (overdispersion-corrected sampling variance + model-refit variance,
  both now also corrected for the spatial-buffer fix below -- see "Model prediction variance", "The
  overdispersion check's independence assumption", and "No spatial buffer between CV folds" in
  Known limitations), versus 14,122 under the DerSimonian-Laird diagnostic (also lower than before,
  for the same reason -- see below). This breaks down as SEVEN separate, independently-measured
  effects, not one: switching DL->Hartung-Knapp alone (still over the original plain-multinomial
  sampling variance) dropped survivors to 6,961 (the DerSimonian-Laird-optimism gap the caveat below
  warns about); folding in model-refit variance on top of that dropped it further to 5,498 (a real,
  separately-quantified source of previously-omitted uncertainty -- see "Model prediction variance"
  below); correcting the sampling variance itself for real extra-multinomial clustering (Note 5 --
  see "The overdispersion check's independence assumption" below) dropped it further still, to
  4,502 (an 18% reduction on top of the already-adopted 5,498, concentrated almost entirely in
  large, high-traffic venues -- see that bullet for why); fixing a separate, unrelated bias in how
  `consolidate_fragments.py` combines MERGED fragment groups' observed scores (Note 6 -- see "The
  merged-group score's Jensen's-inequality asymmetry" below) left that 4,502 total unchanged, but
  flipped two individual merged groups' own FDR call right at the boundary at the time (Osteria
  Brooklyn and John & Joe's Pizzeria -- neither previously named or cited in this README);
  removing the CV's missing spatial buffer (Note 7, a fifth round of external review -- see "No
  spatial buffer between CV folds" below) moved it to 4,871 (a real increase this time, +8.2% --
  see that bullet for the full per-period R² impact and why it goes up rather than down); closing
  an apostrophe-driven fragment-dedup gap (Note 9 -- see "POI fragmentation" below) left the
  survivor count unchanged at 4,871 (10 more fragments merged, net tested family down 7, but the
  ADOPTED count itself didn't move); and finally, replacing the one-way random-effects model's
  implicit independent-months assumption with an explicit GLS correlation structure for the real,
  measured lag-1 serial dependence (Note 10, an eighth round of external review -- see "The
  independence assumption" below) moved it to the current **3,233** (a real decrease this time,
  -33.6% -- driven almost entirely by a BH-FDR cutoff shift under a pooling-wide variance
  inflation, not a bug; see that bullet for the full mechanism and why the drop is so much bigger
  than any single venue's own CI widening). UPDATE: Note 7's rebuild also supersedes Note 6's two
  boundary flips above -- under the buffered numbers, both Osteria Brooklyn and John & Joe's
  Pizzeria were already non-significant, and Note 10 leaves them that way too (now even less
  significant, like every other borderline finding); Note 5's and Note 6's own isolated-effect
  numbers above are still correct as measured at the time, just superseded by later rebuilds, same
  convention throughout this section. Of the four NAMED venue bullets below, **none now clear the
  ADOPTED FDR bar** -- Amenazel Organic was the last one standing after Note 7, and Note 10 is
  what overturns it -- see those bullets for the real, current numbers.

  **Point-estimate correction (external review caught this):** an earlier version of this script
  used the MEAN of the 500 visitor-sampling bootstrap draws (`resid_vs_mean`) as each month's point
  estimate, reserving the observed residual for nothing. That's backwards, and not just
  stylistically: normalized entropy is concave, so the mean of resampled-entropy draws is
  systematically a little LOWER than the entropy of what was actually observed (a finite-sample/
  Jensen's-inequality bias), meaning every pooled estimate below was quietly biased, worst for
  low-device venues where the bootstrap distribution skews most. The fix is the standard one:
  the OBSERVED spatial-CV residual (`improbable_copresence_cv`) is the point estimate; the 500
  bootstrap draws are still exactly the right tool for estimating ITS VARIANCE (`resid_vs_var`,
  used as the inverse-variance pooling weight) -- they just never should have stood in for the
  point estimate itself. The numbers below reflect that fix; see `pool_persistence.py`'s docstring
  for the full explanation and the (deferred) bias-correction alternative if one is ever wanted.

  **What the resulting confidence interval is, and isn't:** `resid_vs_var` captures
  VISITOR-SAMPLING uncertainty only -- "if this venue had drawn a different, equally-likely sample
  of its own visitors, how much would its residual move" -- while holding the fitted RandomForest's
  prediction completely fixed. It does NOT capture uncertainty in that prediction itself (how much
  it would move under different training rows, different spatial folds, different hyperparameters,
  or a different month's panel composition). UPDATE: this gap is now closed -- see the "Model
  prediction variance" bullet immediately below for the fix that was adopted and the real numbers
  behind it. So every CI in this section is a CONDITIONAL interval -- conditional on the fitted
  contextual model being exactly right -- not a full confidence interval for the venue's true
  residual in every respect (see the separate, still-open independence-assumption caveat in that
  same bullet).

  **Model prediction variance -- ADOPTED FIX (external review caught this, second round):**
  quantified directly rather than left as a forward-reference. Ran a 15-replicate spatial-block
  bootstrap (same design as `sensitivity_spatial_block_bootstrap.py`, which fixed the OTHER
  model-refit bootstrap defect below) against a reference sample of flagged venues: model-refit/
  prediction variance was BIGGER than the sampling variance already in the published CI for 31% of
  them, and more than 3x bigger for 18% -- not a rounding-error-sized gap. `estimate_model_
  prediction_variance.py` now estimates this for every eligible venue in every period (not just a
  reference sample -- tracking all of them costs nothing extra per bootstrap replicate, since the
  bottleneck is the model refits, not how many venues' predictions get read off after), and
  `pool_persistence.py`/`consolidate_fragments.py` now pool over `var_combined = resid_vs_var +
  model_pred_var` instead of `resid_vs_var` alone -- this is the ADOPTED pooling variance; see
  those scripts' docstrings. Effect on the full 8-period pool, measured at the time (before the
  separate Note 5 fix below): median variance inflation is +14%, and confirmed findings (ADOPTED
  REML+Hartung-Knapp survivors, consolidated entity set) dropped from 6,961 to 5,498 -- see
  "Persistent across months" above for where that number was used at the time. UPDATE: superseded
  again by Note 5 below, which corrects `resid_vs_var` itself -- the current adopted survivor count
  is **4,502**, not 5,498; this bullet's +14%/5,498 numbers describe only this fix's own, isolated
  effect, measured against the sampling variance as it stood before Note 5. Compute cost, stated
  plainly: this is now a
  REQUIRED monthly pipeline step, not an optional check -- `estimate_model_prediction_variance.py`
  reruns the full grid-cell-blocked `GroupKFold(5)` 15 times per period (~10-15 minutes per period
  on the data sizes seen so far), and must be run for a period before `pool_persistence.py`/
  `consolidate_fragments.py` can pool it.

  **The independence assumption -- FIXED (Note 10, an eighth round of external review):**
  separately, the pooling here used to treat a venue's eligible periods as independent replicates
  of `r_vm`. The number this section used to cite for that ("+0.057 raw / +0.20 excess at lag 1,
  fading by lag 2+") turned out, on tracing it, to describe a DIFFERENT check -- the quintile-share
  serial-correlation test behind Note 5's Dirichlet-multinomial correction (see "The overdispersion
  check's independence assumption" below), not a direct test of `r_vm` itself; no committed script
  ever tested `r_vm` directly before now, a gap caught the same way the Note 7 staging bug was --
  by refusing to cite a number without rerunning the check that's supposed to produce it.
  `estimate_temporal_dependence.py` does, across all 7 available lags, on the same 7,838-venue
  balanced panel: demeaning by each venue's own period mean still mechanically induces a
  correlation of exactly -1/(T-1) = -0.143 between any two periods under pure independence (T=8),
  regardless of lag, so any deviation from -0.143 is real structure. The real result: lag 1 comes
  in at +0.156 excess (raw) / +0.180 excess (variance-standardized), fading to at-or-below the
  benchmark by lag 2 and every lag beyond -- a genuinely one-lag-only pattern, confirming the
  qualitative shape of the old (uncertainly-sourced) claim while correcting its exact number and,
  more importantly, giving it a real, reproducible basis for the first time.

  ADOPTED FIX: `pool_persistence.py`/`consolidate_fragments.py`'s `reml_tau2`/`hartung_knapp` now
  build an explicit GLS covariance Sigma_v = diag(tau2 + var_combined) + rho * sqrt(var_combined_i
  * var_combined_j) on entries for a pair of a venue's eligible months that are CALENDAR-adjacent
  (one month apart on the real Dec 2025-Jul 2026 timeline, not just "next in this venue's own
  eligible-months list"), zero elsewhere. `tau2` keeps its usual meaning (occasion-level
  heterogeneity, independent across months, the same convention the old diagonal method already
  used) -- only the sampling/model-error component (`var_combined`) carries the measured lag-1
  correlation, matching the fact that it fades by lag 2+ rather than acting like a persistent
  between-month signal. rho is FIXED at **+0.1683** (mean of the raw/standardized lag-1 excess
  estimates, same single-point-from-an-honestly-reported-range convention as `S_hat`'s adoption in
  Note 5), not re-estimated per venue -- there isn't remotely enough per-venue data to fit rho and
  tau2 both. At rho=0 this is EXACTLY the old diagonal method (checked to 1e-15, not assumed --
  see `note10/prototype_ar1_pooling.py`, prototyped on a 2,100-venue stratified sample and the four
  named venues before being adopted here, where the regression-check caught a real bug in the
  first attempt at this generalization: modeling the shared random effect as a single draw across
  all of a venue's months rather than keeping it occasion-level, which inflated variance 150x-4000x
  before the fix).

  Effect on the full pipeline, measured after adopting this (not merely prototyped): median
  variance inflation across the 31,645-venue tested family is 1.247x (CIs widen ~11-15%), in line
  with the prototype's 1.235x. But the ADOPTED REML+Hartung-Knapp FDR survivor count drops much
  more sharply than that modest per-venue widening would suggest -- from **4,871 to 3,233**
  (-33.6%) -- because Benjamini-Hochberg's step-up procedure compares every p-value against the
  WHOLE ranked family, and this project's p-values sit densely enough near the historical cutoff
  that a broad, modest upward shift (unadjusted p<0.05 count dropped only ~12%, from 10,138 to
  8,937) cascades into a much bigger drop in the FDR-ADJUSTED count. Checked, not assumed, that
  this isn't a bug: the BH function itself is untouched, the rho=0 sanity check still passes to
  float precision on the real data, and the full-pipeline variance-inflation distribution matches
  the prototype closely. Of the four named venues, **Amenazel Organic -- the only one still
  clearing the bar after Note 7 -- flips to non-significant** (`q_bh_hk` 0.0317->0.0509, the
  narrowest possible margin, driven by all 5 of its eligible months being mutually adjacent so
  every one of its pairs picks up the correlation); Chens Fresh Food Market, the Jacob Riis
  complex, and Hunts Point Floating Pier were already non-significant and stay that way, with
  q-values moving up further. **No named venue survives the ADOPTED FDR bar as of this fix** --
  see each bullet under "Persistent across months" above for the exact numbers. This was a
  deliberate decision, not an oversight: the alternative (reporting a different, still-significant
  venue as the new headline finding, or declining to adopt this correction because the one
  surviving finding didn't survive it) was considered and rejected as a form of the same
  selection bias this project has spent its whole review process correcting for -- see the
  project discussion log for the reasoning. Compute cost, stated plainly: this makes
  `reml_tau2`/`hartung_knapp` do a small (<=8x8) matrix inversion per optimizer step instead of a
  closed-form scalar sum, which is still cheap per venue but adds up across ~35,000 venues --
  `pool_persistence.py` now takes several minutes rather than well under one.

  Known open item from adopting this: `sensitivity_fragments.py` (Note 8's 42-combination
  threshold grid) has its own duplicated `reml_tau2`/`hartung_knapp` and has NOT yet been updated
  to this GLS method, so its own survivor-count/named-venue numbers in "Fragment-consolidation
  thresholds" above still reflect the pre-Note-10 diagonal method -- `sensitivity_eligibility_
  floor_fdr.py` is unaffected since it shells out to `pool_persistence.py` as a subprocess and so
  inherits this fix automatically. Re-running the fragment-threshold grid under the new method is
  a real, not-yet-done follow-up.

  **Model-refit stability bootstrap doesn't reproduce the published scoring procedure (external
  review caught this):** the "30-iteration bootstrap-stability check" described above resamples
  individual VENUES (rows) with replacement, refits one RandomForest, and predicts on every venue
  including the ones whose own rows were just resampled into that same model's training set -- an
  in-sample prediction for those venues, not the out-of-fold prediction the published number
  actually is, and it never re-applies the grid-cell spatial blocking the real CV depends on
  (nearby venues' residuals are correlated ~0.52-0.61; see "Location vs. category" below). The fix
  (`sensitivity_spatial_block_bootstrap.py`): resample grid CELLS, not venues, with replacement,
  and rerun the actual grid-cell-blocked `GroupKFold(5)` inside each replicate (grouped by each
  cell's original id, so a cell drawn more than once still lands entirely in one fold), keeping
  every re-flagging prediction genuinely held out. Run for July 2026 against the real 266
  reference-flagged venues (133 top + 133 bottom): mean stability barely moves (67.3% original vs.
  66.0% corrected), but that average hides a real asymmetry -- 57 venues drop from >=80% stability
  under the flawed procedure to below 80% once corrected, against only 10 that move the other way,
  so the original check was overstating how many venues clear the stability bar, specifically for
  venues sitting near that line. Of the four headline venues, three are essentially unchanged or
  firm up (Chens Fresh Food Market 100%->100%, Hunts Point Floating Pier 100%->100%, the two Jacob
  Riis fragments 86.7%->100% and 60%->100%), but **Amenazel Organic drops from 100% to 61.5%** --
  still flagged most of the time under the corrected procedure, but nowhere near as uniformly
  robust as the flawed bootstrap reported, and the one named venue where this fix changes the
  picture. Caveat on the corrected numbers themselves: because grid cells are resampled with
  replacement rather than appearing every iteration, each venue's own cell was only actually drawn
  in a median of 13 of 20 replicates, and N_BOOT was reduced from 30 to 20 because each replicate
  now requires a full 5-fold refit (~5x the model fits per draw) rather than one -- so the
  corrected per-venue percentages rest on fewer effective draws than the flawed version's, and are
  themselves a first pass (July 2026 only) rather than an exhaustive multi-month recheck. See
  `sensitivity_spatial_block_bootstrap.py` and `sensitivity_spatial_block_bootstrap_results.csv`
  for the full 266-venue table.

  **DerSimonian-Laird with 2-4 months, read honestly (UPDATE: superseded as the adopted
  method):** `tau2` and `I2` below use the standard moment-based DerSimonian-Laird estimator,
  which is known to be optimistic (too-narrow intervals, too-eager to call `tau2=0`) with only a
  handful of observations per group -- a `tau2=0` reading for a venue with 2-4 months usually
  means the heterogeneity test lacked the power to detect real between-month variation, not that
  none exists. REML for `tau2`, Hartung-Knapp intervals (which widen the CI to reflect how little
  data `tau2` itself was estimated from), and a prediction interval for a hypothetical future
  month were flagged here as the documented, better-behaved alternative -- that alternative has
  now been IMPLEMENTED AND ADOPTED (`tau2_reml`/`mu_hk`/`ci_hk_lo`/`ci_hk_hi`/`q_bh_hk`/
  `fdr_sig_05_hk`, in `pool_persistence.py` and `consolidate_fragments.py`); `tau2`/`I2`/`mu_re`/
  `ci_re`/`q_bh`/`fdr_sig_05` are retained only as the DerSimonian-Laird diagnostic/comparison,
  not the reported result. The pooling weight itself was ALSO updated since (see "Model prediction
  variance" in Known limitations) -- the specific counts and named-venue CIs below reflect both the
  HK method switch and the var_combined weight, i.e. the current, final state of both changes.
  Read every `tau2`/`I2`/CI pairing below as "no heterogeneity detected under a method known to
  underpower small samples," not as "heterogeneity ruled out."

  The four named venues, under this corrected point estimate and the ADOPTED REML+Hartung-Knapp
  pooling, each span however many of the current 8 periods (Dec 2025-Jul 2026) they individually
  cleared the eligibility floor in -- not uniformly "four months" now that the pool extends past
  April-July. Every CI below is the conditional, visitor-sampling-only interval described above,
  and every `persistence_tier` reflects only how many periods a venue cleared, not a seasonal or
  annual claim. DerSimonian-Laird's `mu_re`/`q_bh` are given alongside for comparison, but
  `mu_hk`/`q_bh_hk` are what's reported:
  - **Amenazel Organic** (Bronx specialty food store), `persistence_tier`=`current_period_persistence`
    (5/5 eligible periods, Mar-Jul 2026): `mu_hk`=+0.2025, 95% CI [0.100, 0.304], `tau2_reml`=0.0024,
    `q_bh_hk`=0.0509 (was 0.0317 before Note 10's temporal-dependence correction, 0.0106 before
    Note 7's spatial-buffer fix, 0.0081 before Note 5's overdispersion correction, 0.0023 before
    folding model-refit variance into the pooling weight -- see Known limitations). **This is now
    the one named venue where Note 10 changes the finding**: Amenazel no longer clears the adopted
    bar, by the narrowest of margins (q crosses from 0.0317 to 0.0509 -- Note 10's GLS pooling
    widens its CI by about 26% because all 5 of its eligible months are mutually adjacent,
    meaning every one of its 4 within-venue pairs picks up the measured lag-1 correlation), even
    though the point estimate itself barely moves (`mu_hk` 0.2025→0.2025) and DL's diagnostic
    still would call it significant (`mu_re`=+0.203, `q_bh`<0.0001). Read this the same way Jacob
    Riis and Chens have been read since Note 5/7: a real, consistent effect that no longer clears
    a bar that keeps getting more honest about what it isn't accounting for, not a finding that
    turned out to be fake. It remains the cleanest-looking of the four by p-value, just not clean
    enough to survive this correction.
  - **Chens Fresh Food Market** (Brooklyn grocery), `persistence_tier`=`current_period_persistence`
    (5/5 eligible periods, Mar-Jul 2026): `mu_hk`=-0.2153, 95% CI [-0.401, -0.029], `tau2_reml`=0.0099,
    `q_bh_hk`=0.1391 (was 0.1138 before Note 10's temporal-dependence correction, 0.0189 before
    Note 7's spatial-buffer fix, 0.0141 before Note 5, 0.0087 before model-refit variance).
    Already non-significant since Note 7, and Note 10 pushes it further that direction (CI widens
    modestly -- this venue's 5 eligible months aren't all mutually adjacent, so fewer of its pairs
    pick up the lag-1 correlation than Amenazel's did), even though DL's diagnostic still would
    call it significant (`mu_re`=-0.219, `q_bh`=0.0020) and the pooled effect itself barely moves
    (`mu_hk` -0.217→-0.215, still clearly negative and directionally identical). Read this the
    same way Jacob Riis has been read since Note 5: a real, consistent effect that no longer
    clears a bar that keeps getting more honest, not a finding that turned out to be fake -- see
    "No spatial buffer between CV folds" and "The independence assumption" below for why these
    corrections move some venues' residuals/variance enough to matter even when the effect itself
    barely changes.
  - **Jacob Riis Park / beach complex** (the 3-fragment Rockaways cluster, now consolidated into
    one entity -- see POI fragmentation below), `persistence_tier`=`current_period_persistence`
    (5/5 eligible periods, Mar-Jul 2026): `mu_hk`=+0.0709, 95% CI [-0.0028, 0.1447], `tau2_reml`=0.0012
    (DL's diagnostic `I2`=87.2% -- real period-to-period movement, not sampling noise, accounts
    for almost all the apparent variation here, consistent with a warm-weather-linked pattern
    rather than a flat one across the year). **This was the first named venue where the method
    switch changed the finding**, and has stayed non-significant through every correction since:
    `q_bh_hk`=0.1886 (was 0.1446 before Note 10's temporal-dependence correction, 0.1424 before
    Note 7, 0.1168 before Note 6, 0.1094 before Note 5, 0.0665 before folding in model-refit
    variance, itself already above the bar under the HK method switch alone), versus `q_bh`=0.00002
    under the DL diagnostic. Jacob Riis is both the kind of large, high-traffic venue Note 5's
    correction hits hardest (see that bullet below -- a venue this size sees its visitor-sampling
    variance roughly double) AND the only named venue that's a merged fragment group, so it was
    also the only one of the four directly affected by Note 6's fix -- its old, pooled-entropy
    `merged_score` was mechanically inflated relative to the device-weighted average of its 3
    fragments' own scores. The pooled effect is still clearly positive and directionally identical
    under every method tried -- what's changed, cumulatively, is whether it clears a strict
    multiple-testing bar once the between-period variance is estimated by REML instead of DL's more
    optimistic closed-form estimator, the CI widened by Hartung-Knapp, the pooling weight widened by
    real model-refit uncertainty, the sampling variance itself corrected for real extra-multinomial
    clustering, the merged score corrected for the pooling-combination bias, and now the CV itself
    corrected for the missing spatial buffer. Read this as a real, persistent, but
    no-longer-formally-confirmed effect under the stricter standard -- not as a venue that stopped
    being unusual.
  - **Hunts Point Floating Pier** only clears the eligibility floor in 2 of the current 8 periods
    (June and July 2026) -- see `presence_bookkeeping_2026.csv` for exactly why it's
    absent/ineligible the others. `persistence_tier`=`repeated_evidence`: pooled across those 2
    adjacent periods, `mu_hk`=+0.268, 95% CI [-0.180, 0.717] (wide, since Hartung-Knapp's
    t-distribution has only 1 degree of freedom at n=2, widened further by model-refit variance,
    Note 5's overdispersion correction, Note 7's spatial-buffer fix, and now Note 10's
    temporal-dependence correction -- its only adjacent month-pair picks up the full measured
    lag-1 correlation -- Hunts Point is another high-traffic venue, hit by the same size-dependent
    effect as Jacob Riis), `q_bh_hk`=0.2368 (was 0.2037 before Note 10, 0.1816 before Note 7,
    0.1699 before Note 5, 0.117 before folding in model-refit variance), versus `q_bh`<0.0001
    under the DL diagnostic (whose normal-approximation
    CI is much narrower at this same sample size -- exactly the DL optimism problem this switch was
    meant to correct for). A real, precisely-estimated-looking effect that, assessed honestly under
    a method built for few observations and now a properly corrected sampling variance and CV,
    isn't yet distinguishable from chance at only 2 eligible periods -- not evidence it's wrong,
    but a reminder that 2-period pooling is the thinnest evidence this project produces.

  Separately -- and this is the correction that prompted building the citywide consolidation pass
  in the first place -- the Brooklyn "Jacob Riis Beach" catalog entry, 14.6km from the real
  Rockaways complex and confirmed to be a different, unrelated venue that merely shares part of
  its name (see POI fragmentation below), pools to `mu_hk`=+0.0247 across its 3 eligible periods
  (`persistence_tier`=`provisional_persistence`), 95% CI [-0.004, 0.054], `q_bh_hk`=0.1859 (was
  0.2017 before Note 7, 0.1730 before Note 5, 0.117 before model-refit variance) -- same as Hunts
  Point, this does not clear the adopted bar (DL's diagnostic `q_bh`=0.00001 would have called it
  significant), a much smaller effect than the real Jacob Riis complex regardless of method, and
  should never be cited as evidence for, or against, the Jacob Riis finding. Unlike its Rockaways
  namesake, this is a small, low-traffic venue, so neither Note 5's nor Note 7's correction moved
  it by much -- the small shifts here are mostly ordinary noise from a fresh bootstrap/refit, not a
  real overdispersion or buffer
  effect.

  **The overdispersion check's independence assumption -- FIXED (Note 5, a third round of
  external review):** the visitor-sampling bootstrap assumes each income-known device's
  home-quintile draw is independent given the venue's observed rate. If visits cluster -- multiple
  devices from the same household, workplace, or tour group arriving together -- true sampling
  variance runs higher than a plain multinomial implies. This was checked directly early on
  (`estimate_overdispersion.py`, originally against May/June/July 2026 only -- the first three real
  months that existed at the time) via a chi-square test of homogeneity across each venue's 2-3
  months, with the right degrees of freedom for the fact that each venue's reference rate has to be
  *estimated* from those same few months. That version found an observed-to-expected variance ratio
  of 0.91 and read it as real evidence AGAINST clustering -- but it rested on an assumption it never
  tested: that a venue's different months are independent of each other once you've accounted for
  its own fitted mean. With only 3 months (at most 2 lag-pairs per venue), there wasn't enough data
  to test that directly.

  Now that all 8 periods exist, `estimate_overdispersion.py` tests it directly, on the same
  7,838-venue balanced panel used for the serial-correlation check two bullets up: it measures the
  actual lag-1 correlation between adjacent periods' quintile-share deviations from each venue's
  own period mean, against the same -1/(T-1) demeaning-artifact benchmark. All 7 available
  period-pairs show real, consistently positive excess correlation (+0.03 to +0.29, averaging
  +0.19) -- not an artifact, and not a result of too little data this time. The raw VIF on the full
  8-period panel is already 1.21 (not 0.91 -- that was an artifact of only having 3 months to check
  against), and correcting the reference distribution for the measured serial correlation pushes it
  to roughly 1.48 at the mean estimate (honest range 1.24-1.70 across the 7 pairs' own estimates).
  **This reverses the original conclusion**: there is real evidence FOR extra-multinomial
  clustering in how devices visit venues, not against it.

  ADOPTED FIX: `estimate_visitor_sampling_variance_dm.py` now draws each bootstrap replicate from a
  Dirichlet-multinomial instead of a plain multinomial, using the Dirichlet-multinomial
  concentration `S_hat`=732 this correction implies (honest range 504-1,452 across the 7-pair
  estimates -- see `estimate_overdispersion.py`'s docstring for the full derivation).
  `pool_persistence.py`/`consolidate_fragments.py` now read `resid_vs_mean`/`resid_vs_var` from
  this corrected bootstrap instead of `fit_mixing_model_2026_0X.py`'s plain-multinomial one (left
  unmodified and documented as superseded, rather than touched directly, since nothing downstream
  still reads its visitor-sampling columns).

  This correction is NOT a flat percentage bump -- it's strongly size-dependent, because the
  Dirichlet-multinomial's variance-inflation factor VIF(n)=(S+n)/(S+1) scales roughly linearly with
  a venue's own device count. Across the full 180,367 eligible venue-periods: venues near the
  50-device eligibility floor see almost no change (median +10% variance for n<100, 37% of all
  venue-periods); mid-size venues (100-499 devices, 56% of all venue-periods) see +27%; venues with
  500-1,999 devices see their sampling variance roughly double (+106%); and the 1,037 venue-periods
  (0.6%) above 2,000 devices see it roughly quintuple (+406%). In practice that means this
  correction barely touches most small, individually-named businesses, but hits large, high-traffic
  venues -- exactly the Jacob Riis complex and Hunts Point Floating Pier -- hardest; see those two
  venues' bullets above for the actual effect on their numbers.

  Caveat on the correction itself, stated plainly: `S_hat` is estimated from only 7 lag-pairs (each
  itself pooled across all 7,838 panel venues, so 7 is the real sample size for "how many
  independent looks at the serial-correlation structure," not 7,838). The 7 estimates are
  reasonably consistent -- all positive, no sign flips -- but span a real range, so `S_hat`=732 is
  the best available point estimate, not a precisely known constant; a few more periods of data
  would narrow this meaningfully. The test still can't fully separate genuine device-level
  clustering (the thing it's meant to measure) from the same kind of real month-to-month behavioral
  persistence the serial-correlation bullet above documents elsewhere in this pipeline -- both would
  show up as "more cross-period movement than plain sampling predicts," and this check doesn't
  cleanly distinguish them. Read the correction as a real, data-driven improvement over assuming
  independence outright, not as a precisely-calibrated final answer.

  **The merged-group score's Jensen's-inequality asymmetry -- FIXED (Note 6, a fourth round of
  external review):** this one applies only to the 265 merged fragment groups in
  `consolidate_fragments.py` (see POI fragmentation below) -- singleton venues (the other 34,409
  consolidated entities) were never affected. A merged group's observed score, `merged_score`, used
  to be computed as the entropy of its fragments' SUMMED quintile counts -- equivalent to H(the
  device-weighted average of each fragment's own quintile shares). Normalized entropy H is concave,
  so by Jensen's inequality H(weighted avg of shares) >= weighted avg of H(shares), ALWAYS, as a
  mathematical fact, not an empirical one -- confirmed computationally across all 1,706 real
  (group, period) observations in this project's own data, zero exceptions, median gap 0, mean
  0.0013, max 0.027. `merged_expected_cv` (the baseline/"expected" side of the same residual) got no
  equivalent treatment: it was already a plain device-weighted AVERAGE of each fragment's own
  already-fitted `expected_mixing_cv`, not an entropy-of-pooled quantity. So the two sides of every
  merged group's residual were being combined by two different, inconsistent rules -- mechanically
  inflating `merged_score` (and therefore the residual) relative to a method that matches
  `merged_expected_cv`'s own combination rule, with no necessary connection to whether the merged
  venue is actually more mixed in any real sense. The effect is small in absolute terms and
  concentrated in multi-fragment groups, and before fixing it, quantifying it first showed it
  changed no reported/cited finding's sign or survival call.

  ADOPTED FIX: `merged_score` is now a device-weighted average of each fragment's OWN
  `income_mixing_score` (same vendor entropy formula, applied to each fragment's own shares, THEN
  averaged) -- exactly mirroring `merged_expected_cv`'s own combination rule, so both sides of the
  residual are now built the same way. The merged group's own Dirichlet-multinomial bootstrap (see
  Note 5 above) was updated to match: each of its 500 draws now resamples every fragment's OWN
  quintile counts independently (own `alpha_i = S_hat * own quintile shares`), scores each
  fragment's own draw, and takes the same device-weighted average across fragments -- rather than
  pooling fragments' counts into one draw and scoring the pool -- so the bootstrap variance stays
  consistent with what the point estimate is actually averaging.

  Effect on the full consolidated table: the total ADOPTED REML+Hartung-Knapp FDR survivor count is
  completely unchanged at **4,502** (see "Persistent across months" above), but two individual
  merged groups' own FDR call flip right at the q<=0.05 boundary -- Osteria Brooklyn (`GRP00207`,
  Brooklyn) moves from surviving (`q_bh_hk`=0.0384) to not (`q_bh_hk`=0.0597), and John & Joe's
  Pizzeria (`GRP00172`, Bronx) moves from not surviving (`q_bh_hk`=0.0550) to surviving
  (`q_bh_hk`=0.0499); neither was previously named or cited anywhere else in this README. Of the
  four named venues discussed above, only Jacob Riis is a merged group, so it's the only one this
  fix touches directly -- see its bullet above for the updated numbers (its already non-significant
  `q_bh_hk` moved from 0.1168 to 0.1424, since its old `merged_score` was inflated by this same
  mechanism). `fragment_groups_report.csv`'s 265 groups themselves are unaffected -- this fix
  changes how a merged group's score is computed, not which fragments get grouped together.

  **No spatial buffer between CV folds -- FIXED (Note 7, a fifth round of external review):**
  grid-cell `GroupKFold` keeps a whole ~2000ft cell on one side of a fold, but says nothing about
  cells on OPPOSITE sides of a fold boundary -- they sit directly adjacent with a 0ft gap, even
  though residuals are measurably autocorrelated well past one cell width (see "Location vs.
  category" below, r≈0.52-0.61 among near neighbors). A correlogram on real residuals -- checked
  on 4 periods spanning the full window, not just one, after this exact question was raised --
  confirms it's not an artifact of any single month:

  | distance | Dec 2025 | Feb 2026 | Apr 2026 | Jul 2026 |
  |---|---|---|---|---|
  | 0-250ft | 0.50 | 0.50 | 0.56 | 0.56 |
  | 250-500ft | 0.39 | 0.40 | 0.43 | 0.48 |
  | 500-1000ft | 0.27 | 0.28 | 0.34 | 0.41 |
  | 1000-1500ft | 0.18 | 0.20 | 0.20 | 0.31 |
  | 1500-2000ft | 0.14 | 0.16 | 0.16 | 0.21 |
  | 2000-3000ft | 0.12 | 0.10 | 0.11 | 0.18 |
  | 3000-5000ft | 0.05 | 0.05 | 0.08 | 0.13 |

  The shape is consistent across periods (July runs a bit "stickier" at every distance, consistent
  with the warm-weather clustering already documented under "Serial correlation" above) and decays
  smoothly rather than hitting zero at any clean cutoff -- there is no single obviously-correct
  buffer distance, only a bias/variance tradeoff, so this was checked directly rather than guessed.

  ADOPTED FIX: `fit_mixing_model_2026_0X.py`'s PRIMARY CV and `estimate_model_prediction_
  variance.py`'s own resampled refits now additionally exclude any training-candidate venue within
  `BUFFER_FT`=1500 of ANY test-fold venue (standard "buffered CV" -- Roberts et al. 2017), not just
  venues sharing its own cell. 1500ft was picked over a larger buffer for a concrete, checked
  reason: at this venue density, a buffer wide enough to clear the correlogram's tail (3000ft+)
  excludes 85-90%+ of each fold's training venues and starts starving individual boroughs
  entirely -- checked directly on April (min per-borough training count collapses to 122-337
  venues at 3000ft, against 772+ at 1500ft). 1500ft keeps a workable training set while still
  cutting the near-boundary correlation roughly in half relative to no buffer at all.

  Effect, measured across all 8 periods, not estimated from one: the PRIMARY (buffered) CV R²
  drops in every period -- 0.552→0.496 (Dec), 0.643→0.585 (Jan), 0.680→0.641 (Feb), 0.662→0.604
  (Mar), 0.693→0.620 (Apr), 0.730→0.665 (May), 0.720→0.650 (Jun), 0.719→0.660 (Jul) -- confirming
  the un-buffered design was genuinely optimistic, not just in the one month first checked.
  `model_pred_var` (see that bullet above) rises correspondingly (medians roughly 1.3-1.8x higher
  per period), since a buffered refit is a noisier estimate of the same prediction. Propagated
  through the full pipeline -- `pool_persistence.py`/`consolidate_fragments.py` rerun on the
  buffered `expected_mixing_cv` and `model_pred_var`, for real, not approximated -- the ADOPTED
  REML+Hartung-Knapp FDR survivor count moves from **4,502 to 4,871** (+8.2%). Of the four named
  venues: Amenazel Organic still survives, but its margin thins (`mu_hk` 0.274→0.202, `q_bh_hk`
  0.0106→0.0317); Hunts Point Floating Pier and the Jacob Riis complex both stay non-significant,
  essentially unmoved (`q_bh_hk` 0.182→0.204 and 0.142→0.141 respectively); and **Chens Fresh Food
  Market flips from surviving (`q_bh_hk`=0.0190) to not surviving (`q_bh_hk`=0.114)** -- the one
  previously "clean" named finding this fix actually overturns. Read Chens the same way Jacob Riis
  has been read since Note 5: a real, directionally consistent effect (`mu_hk` still -0.217) that
  no longer clears the stricter, now-more-honest bar, not a finding that turned out to be fake.

  Known follow-ups, stated plainly rather than silently left inconsistent: (1)
  `sensitivity_spatial_block_bootstrap.py`'s July reference-stability table (the "Amenazel 100%→
  61.5%" finding under "Model-refit stability bootstrap" above) was computed against the
  PRE-Note-7 flagged-venue set and has not yet been rerun under the buffer -- its specific
  per-venue percentages predate this fix. (2) `geo_temporal_holdout_validation.py`'s cross-month
  comparison (under "Does the model generalize across TIME" above, April's "0.698 against its own
  within-month 0.69") also predates this fix and compares against the un-buffered April R² --
  a matching buffered rebuild hasn't been done. Neither changes the ADOPTED persistence/FDR
  numbers above, which already reflect the full buffered pipeline; both are secondary/diagnostic
  artifacts that would need their own rerun for internal consistency, not yet done as of this fix.
- **`balance_score` is a robustness check, not independent confirmation**: an earlier draft of
  this methodology treated a venue's also being extreme under `balance_score` as a third,
  independent bar alongside bootstrap stability and cross-month replication. Directly checking
  this: among venues flagged by `income_mixing_score`, the two metrics' residual z-scores
  correlate at roughly 0.98. That's not independent evidence -- the two scores are overwhelmingly
  tracking the same underlying signal, not catching different things. `balance_score` still has
  value as a check that the primary score isn't an artifact of one particular target definition,
  but it should not be described as a second, unrelated confirmation of a finding.
- **Lingering/dwell measurement is not used in the headline scores**: `income_mixing_score`
  and `balance_score` -- and every feature in the RandomForest model -- are built entirely
  from visitor counts by income quintile; no dwell-time or "lingering" measure feeds into
  either. A separate score family (`typical_dwell_mixing_score`, `dwell_balance_score`,
  `shared_balance_score`, and the `shared_candidate` gate) was explored to check whether
  headline venues also cleared a stricter, dwell-based bar, and almost none did (1.6% of
  eligible venues clear `shared_candidate`; 74% of eligible venues have zero devices meeting
  the lingering criterion in at least one income quintile). That's largely a measurement
  artifact, not a behavioral finding: Veraset changed how a "visit" gets attributed in
  mid-2024 (a single ping now qualifies, versus two pings ~4 minutes apart previously), and
  dwell time now reads as zero for the large majority of visits nationally (74-80% per
  Veraset's own documentation) -- not because visits got shorter, but because a single ping
  can't establish a duration in the first place. The near-total failure of the dwell-based
  family to clear its own bar reflects that measurement gap, not evidence that few venues
  sustain an income-diverse crowd. Since these scores were never used to produce any reported
  finding, this doesn't affect anything above -- it's documented here so the near-empty
  `shared_candidate` column isn't mistaken for a result.
- **Sensitive-location categories are absent, not just underrepresented**: Veraset has
  filtered visits to health care facilities, places of worship, and shelters from its data
  since August 2022 for privacy reasons -- and, confirmed directly against the June 2026 file,
  K-12 schools as well: searching all 92,168 venues for NAICS 6111 (Elementary and Secondary
  Schools) returned zero matches, against dozens of colleges, tutoring centers, and
  specialty-instruction venues that do appear. (One public school, PS 128M, slipped through
  mislabeled under the college/university NAICS code -- a rare leak, not evidence K-12 schools
  are generally present.) This isn't a sampling gap that undercounts an otherwise-present
  category; these venue types cannot appear in this analysis at all, regardless of how popular
  or income-mixed they might be. That matters for how this project frames itself: Oldenburg's
  original "third place" concept and Klinenberg's "social infrastructure" both treat
  congregations and schools as central examples of places that foster cross-class contact, and
  this dataset structurally cannot speak to either category. A sensitive location nested inside
  a larger one (a cafeteria inside a school, a gift shop inside a hospital) is removed along
  with it too, which can make an otherwise-ordinary venue disappear for reasons unrelated to
  its own category.
- **Category reliability**: `top_category`/`sub_category` come from the location-data vendor and
  aren't fully reliable -- see the "libraries under NAICS 712" note above for the general
  pattern. We spot-checked every vendor-tagged "library" against NYC's official public library
  locations (`NYC_Library_Locations.csv`, from NYC Open Data, combining the NYPL/BPL/Queens
  Public Library systems -- see `verify_library_categories.py`) by nearest-neighbor distance.
  One branch, Castle Hill, was mistagged as a grocery store and is now treated as a library in
  any library-level figure. One vendor-tagged "library," Washington Square Library, sits ~260m
  from the nearest real library point -- inside Washington Square Park itself rather than at a
  library address -- and could not be confirmed as an actual library; it's excluded from the
  libraries analysis rather than reported as a finding. Plain proximity matching is noisy in
  dense blocks (lots of unrelated venues sit within a few dozen meters of any given building),
  so this was a targeted check, not a wholesale recategorization, and shouldn't be read as
  having validated every other category in the data.
- **Location vs. category**: location predicts how mixed a venue reads far more than what kind
  of business it is does. Nearest-neighbor spatial autocorrelation of the model's residuals (how
  much more or less mixed a venue is than the model expects, after accounting for its category,
  income, race, population density, and transit/tourist/chain-business context) comes out to
  r≈0.52-0.61 depending on how many neighbors are averaged (R²≈0.27-0.37) -- a venue's "surprise"
  score is strongly predicted just by its physical neighbors' surprise scores, independent of
  category. Category itself explains far less: `top_category` (24 groups) accounts for under 1%
  of the variance left in that residual once the model has already used it as an input, and only
  about 2.9% of the raw, pre-model mixing score. In Cobble Hill specifically, an 11-venue cluster
  of flagged businesses spanning several unrelated categories all cleared the surprise threshold
  together, in the same direction -- a pattern that fits something about that stretch of blocks
  much better than it fits anything the businesses have in common. This is also the motivation
  for the spatially-smoothed local-income feature described above: averaging the income context
  of nearby venues, not just a venue's own census block, is a direct test of whether some of this
  spatial clustering is really just uncaptured neighborhood income context. Adding it modestly
  reduced, but didn't eliminate, both the spatial autocorrelation and the Cobble Hill cluster's
  average surprise score (about 9% smaller in absolute z-score terms). The practical takeaway
  this project draws from all of this: a corridor-level or neighborhood-level pattern is real and
  worth flagging, but pinning it on an individual named business is a much weaker claim than the
  data supports, and isn't one this project intends to make.

  Two checks were run to make sure this wasn't an artifact of the wrong level of categorization
  or of the vendor's category labels themselves. First, granularity: category's explanatory power
  was tested at three resolutions -- the pipeline's own coarse `encounter_type` grouping (3
  buckets), the vendor's `top_category` (24 groups), and its `sub_category` (62 groups) -- and it
  moves almost exactly the way you'd expect if the category signal is genuinely weak rather than
  mismeasured: it collapses going coarser (`encounter_type` explains under 0.3% of raw variance)
  and is essentially flat going finer (`top_category` and `sub_category` both sit around 2.9%).
  If a more predictive category boundary existed at some level between these, finer categories
  should have explained meaningfully more than coarser ones; instead the relationship saturates
  almost immediately, which argues against "we haven't sliced business type finely enough" as the
  explanation. Second, mislabeling specifically between public institutions and commercial
  venues -- the two categories furthest apart in character, and the pairing most likely to
  distort this finding if the vendor's tagging confused them. Cross-checking venue names against
  category assignments found the crossover rate is low in both directions (0.7% of civic-tagged
  venues have food/drink-sounding names, 0.08% of commercial-tagged venues have civic-sounding
  names), and most of even those turned out to be correctly tagged businesses whose names are
  simply misleading (Fraunces Tavern is genuinely a museum; Washington Market Park is genuinely a
  park). Two real errors did surface -- Castle Hill Branch Library tagged as a grocery store
  (already noted above) and Bar-Ilan University tagged as a drinking establishment, apparently
  from automated tagging catching the substring "Bar" in the name -- both worth correcting, but
  at roughly two confirmed errors across the full eligible universe, mislabeling isn't a
  plausible explanation for category's weak showing overall. Taken together: business category,
  checked at multiple resolutions and cross-referenced against name-based sanity checks,
  consistently explains only a small share of why a venue reads as more or less mixed than
  expected -- far less than location does. This isn't a measurement artifact; it's a real feature
  of the data.
- **POI fragmentation, now corrected with a general citywide pass rather than a one-off fix**: a
  handful of physical locations are represented by more than one catalog entry sharing the same
  footprint but different names -- most notably the Jacob Riis Park/beach complex in the
  Rockaways. An earlier version of this section fixed that one case by hand; `consolidate_fragments.py`
  now does this citywide and systematically, so the same correction applies everywhere it occurs,
  not just where we happened to look. It groups venues by THREE conditions together: within 75ft
  (~23m) of each other spatially, name-token Jaccard similarity >=0.4 after stripping stopwords
  (including street-suffix words like "street"/"ave"/"blvd" -- added after an early version of
  this check nearly merged "Knapp Street Pizza" with the unrelated "Knapp Street Bagel Cafe" on
  shared street-name tokens alone) and normalizing a trailing possessive `'s` so "Lorenzo's"
  tokenizes the same as "Lorenzo" (UPDATE, Note 9 -- see "Fragment-consolidation device dedup"
  under Validation checks already run above: the vendor's own parent/child and shared-polygon
  flags surfaced this as a real, if modest, apostrophe-driven near-miss gap), AND the same vendor
  `top_category`. Run across the union of all 8 periods' eligible venues (34,943 placekeys -- this
  grows each time a new period is folded in, since it's a union across periods, not a fixed catalog
  size), this finds 275 fragment groups (554 placekeys merged down to 275 entities); 49 of those
  groups are flagged separately as exact-name matches (e.g. two "Starbucks" entries, two "The
  Salvation Army" entries) -- a real chain/franchise sitting twice in the same spot is plausible,
  but so is a genuine chain having two nearby-but-distinct locations, so these 49 are listed in
  `fragment_groups_report.csv` for a manual glance rather than auto-merged with full confidence.

  Merging a confirmed fragment group sets its merged `income_mixing_score` as a device-count-
  weighted average of each PRESENT fragment's OWN `income_mixing_score` that month (UPDATE, Note 6
  -- see "The merged-group score's Jensen's-inequality asymmetry" above: an earlier version of this
  pass instead took the entropy of the fragments' summed q1-q5 device counts, which is mechanically
  biased upward relative to this by Jensen's inequality, for reasons unrelated to real mixing), and
  sets its merged "expected" reference as the same device-count-weighted average of each fragment's
  OWN already-fitted `expected_mixing_cv` -- a combination rule over existing fitted predictions,
  explicitly NOT a full model refit on merged covariates, stated plainly as a real limit of the
  quick-turnaround approach rather than overclaimed rigor. The merged entity then gets its own
  visitor-sampling bootstrap (also updated for Note 6 to match this combination rule -- see above)
  and is pooled across months exactly like any other venue (see the persistence bullet above).

  For the Jacob Riis complex specifically: the correct group is 4 fragments -- "Riis Park Visitor
  Center," "Jacob Riis Park," "The People's Beach at Jacob Riis Park," and a "Jacob Riis Beach"
  entry that IS part of the real Rockaways cluster -- all within the 75ft threshold, same
  category, name-Jaccard above 0.4. (The current pooled estimate -- `mu_hk`, CI, `q_bh_hk` -- is
  reported in exactly one place, the "Jacob Riis Park / beach complex" bullet under "Persistent
  across months" above, rather than repeated here where an earlier draft's copy had drifted out of
  sync with it; see that bullet for the real numbers and every method-change caveat.) Critically,
  this is NOT the same "Jacob Riis Beach" originally flagged in passing conversation as "tagged
  Brooklyn instead of Queens, clearly a mis-assignment" -- that was a misreading of the data, not a
  data bug. There is a genuinely separate, different "Jacob Riis Beach" catalog entry, 14.6km away
  in Brooklyn (confirmed by haversine distance, a different `cbg_geoid`, a much smaller footprint,
  and no spatial or name-similarity edge to the Rockaways group under this consolidation's rule),
  which correctly stays its own unmerged singleton entity, pooling to a small positive effect that
  does NOT clear the adopted FDR bar (see the "Separately..." paragraph right after that same
  bullet above for its current numbers) -- entirely on its own, unrelated merits, and must not be
  cited as part of, or against, the real Jacob Riis finding.
- **Most observed venues never reach the model, and now there's a precise, per-venue-month
  accounting of why**: `presence_bookkeeping.py` compares the FULL raw vendor export (88k-94k
  venues/month, before ANY filtering) against what actually reaches the pooled persistence table,
  for every consolidated entity and month, and assigns one of six statuses: `pooled`,
  `present_ineligible` (under the 50-device floor), `present_eligible_excluded_by_category`,
  `present_eligible_no_income_data`, `present_eligible_fragment_rescue_not_fitted`, or `absent`
  (zero devices observed that month -- see the caveat below). Investigating an initially huge,
  undifferentiated "present and eligible but not pooled" count (20,000+ venues/month) found it's
  fully explained by three specific, now-documented mechanisms, not a mystery gap:
  1. **The `encounter_type` category filter** (`build_features_2026_0X.py`, previously
     undocumented at this level of detail): a venue only enters the model at all if its
     `top_category` falls into `public_civic`/`commercial_social`/`personal_social` --
     categories like auto dealers, gas stations, contractors, and most retail unrelated to
     public/social space (`mobility`/`work_private`/`other`) are excluded regardless of device
     count, before the 50-device floor is even applied. This is a real, intentional design
     choice -- not a bug -- but it means the ~27,000-30,000-establishment universe this project
     reports is restricted to specific encounter categories by construction, not simply "every
     actively-visited NYC establishment," and that should be stated plainly wherever that
     establishment count is cited (see the pitch document). Accounts for ~96% of the gap
     (20,730-21,908 venues/month, varying slightly by month).
  2. **CBG-income match failure**: a venue's spatial geocode to a Census block group can succeed
     while that specific block group still has no income record at all in the ACS extract --
     consistent with mostly-unpopulated block groups (parks, cemeteries, industrial land).
     Accounts for the remaining ~4% (878-1,017 venues/month).
  3. **Fragment-rescue groups that clear the combined floor but were never individually fitted**:
     a handful of consolidation groups (8 venue-months total, across all four months, all 2-fragment
     groups) where the COMBINED income-known-device count clears the 50-device floor but neither
     individual fragment did on its own in the ORIGINAL (pre-consolidation) pipeline run -- so
     neither fragment ever got its own `expected_mixing_cv` fit that month, and the merge step
     (a weighted average of fragments' own fitted predictions, see POI fragmentation above) has
     nothing to combine. These are correctly-identified "rescued" cases the pipeline still can't
     pool, not a new unexplained mechanism; after separating out all three mechanisms above, the
     residual `present_eligible_unexplained` status is confirmed at exactly zero in every month --
     i.e., nothing is left over that these three don't account for.

  Stated caveat, carried over from the module's own docstring: the raw vendor export itself
  requires >=1 observed device to produce a row at all, so "absent" here means "zero devices
  observed that month," not necessarily "not a real place that could have been visited" --
  distinguishing those two would need the full places catalog (not part of this pipeline), and
  that gap is flagged rather than quietly elided. Separately, this same bookkeeping surfaced a
  related, structural limitation worth stating plainly: fragment consolidation (see POI
  fragmentation above) runs on `model_ready`, which is already filtered to individually-eligible
  venues, so case 3 above (fragment-rescue) is a genuine, unresolved gap in the current pipeline,
  not fully fixed by consolidation alone -- quantified here rather than left as an assumption.

## Pre-specified criteria for adding a new month (locked 2026-10-03, before March)

This project runs discovery and confirmation as two separate steps, and this section exists so
the confirmation step's rule is written down and dated BEFORE the next month of data (March
2026) is added -- not adjusted afterward to fit whatever March happens to show. **Discovery**
(which venues get a first look at all) is the top/bottom 0.5%-per-borough percentile rule
described in "What the model does." **Confirmation** (which of those are reported as a real,
persistent finding, not noise) is the pooling rule below. A venue is only named as a persistent
finding if it clears the confirmation rule -- clearing the discovery screen alone is not
sufficient, and is not how any venue in this writeup is described.

The confirmation rule, exactly as currently implemented and not to be changed retroactively once
March's residuals exist:

1. **Eligibility**: a venue-month enters the pool only if it clears `income_known_devices >= 50`
   (the eligibility floor already used throughout) AND falls in an `encounter_type` of
   `public_civic`/`commercial_social`/`personal_social` AND has a CBG income match -- see
   `presence_bookkeeping.py` in Known limitations for exactly which venue-months this excludes
   and why.
2. **Per-venue-month residual**: `improbable_copresence_cv` (the OBSERVED spatial-CV residual --
   actual score minus the FIXED out-of-fold expectation) as the point estimate, with
   `resid_vs_var` (the 500-draw visitor-sampling bootstrap's variance against that same fixed
   expectation) as its pooling weight -- see "What the model does" and the point-estimate-correction
   note under Known limitations for why the point estimate is the observed residual, not the
   bootstrap mean.
3. **Pooling**: REML + Hartung-Knapp random-effects meta-analysis across a venue's eligible
   months (the ADOPTED method), producing `mu_hk`, a 95% CI, `tau2_reml`, and a prediction
   interval (`pool_persistence.py` / `consolidate_fragments.py`); the same script also reports
   DerSimonian-Laird's `mu_re`/`tau2`/`I2`/CI as a diagnostic/comparison, not the reported result.
4. **Tested family**: only venues with `n_months >= 2` are tested for persistence at all (a
   single month has no replication to test against, and is never reported as "confirmed" no
   matter how large its one-month residual).
5. **Significance bar**: Benjamini-Hochberg FDR correction across that entire tested family,
   threshold `q_bh_hk <= 0.05` (the adopted HK-based q-value; `q_bh` is the DL-based diagnostic).
   This is applied citywide, across every eligible venue, not just the
   handful already named in the pitch -- so adding March and re-running this step can only ever
   ADD or REMOVE venues from the confirmed set by this same rule, never by hand-picking a
   different threshold because a particular venue's March number looks interesting or disappointing.
6. **Fragment consolidation**: the 75ft / name-Jaccard>=0.4 / same-`top_category` grouping rule
   in `consolidate_fragments.py` runs before pooling, on the union of eligible placekeys across
   ALL months including March once it's added -- so a fragment group's membership is decided by
   geometry and naming, not by which specific months happen to need a merge.

Any change to these specific numbers (50-device floor, 500 draws, 75ft, 0.4 Jaccard, `n_months>=2`,
`q_bh<=0.05`) after March's data exists should be logged here with a date and a stated reason
unrelated to March's own results (e.g. "raising the floor to reduce finite-sample bias," not
"because Amenazel's March number came in weaker than expected") -- the whole point of writing
this down now is so that kind of change is visible rather than quietly absorbed into a rerun.

**Change log:**
- 2026-10-03, same day as the lock above, before March exists and for a reason that has nothing
  to do with any month's results: an external methodological review caught that item 2's point
  estimate had been implemented backwards (the bootstrap mean standing in for the observed
  residual, which is a biased estimator of it -- see Known limitations). Corrected to the version
  recorded above. Flagged here rather than quietly fixed, exactly per this section's own rule.

## Environment

Python 3 with `pandas`, `numpy`, `scikit-learn`, `duckdb`, and `pyarrow` (for the raw-parquet
steps in the notebook only — the standalone `.py` scripts only need CSV/pickle I/O plus
`pandas`/`numpy`/`scikit-learn`).
