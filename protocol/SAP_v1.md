# Statistical Analysis Plan v1

**Frozen date:** 2026-08-23  
**Status:** pre-outcome design freeze  
**Study:** Coverage Is Not Access: National Diffusion, Referral Networks, and Equity of Therapeutic ERCP for Choledocholithiasis in Brazil's Unified Health System, 2021–2025

## 1. Primary question and evidence contract

The study asks whether formal national incorporation of therapeutic ERCP translated into sustained, equitable, and network-resilient real access within SUS. Aim 1–3 form the minimum publishable core. Aim 4 strengthens clinical relevance but remains associational. A staggered-adoption quasi-causal analysis is optional and enters the main text only if every gate in `config/qc_gates.yaml` passes.

All outputs must use one of three labels: descriptive, associational, or causal-eligible. Failure of a gate can only downgrade the claim; it cannot be repaired by changing wording after inspecting favorable results.

## 2. Study period, populations, and counting unit

The primary window is January 2021 through December 2025. Years 2016–2020 are historical context only because the dedicated code is not a homogeneous pre-period measure. Provisional 2026 files are excluded from the main paper.

The implementation cohort includes any SIH-SP record containing procedure code `0407030255`, collapsed to one AIH by competence month, CNES, and NAIH. SP detail rows and `SP_QTD_ATO` are not case counts.

The primary clinical and equity cohort is the adult subset whose valid RD diagnostic fields include K80.3, K80.4, or K80.5, excluding principal C23/C24 indications. The all-indication cohort supports implementation analyses; it is never represented as synonymous with choledocholithiasis.

## 3. Aims and primary estimands

### Aim 1: diffusion and maintenance

Estimate monthly adoption among eligible hospital-months and 12-month retention among adopters. Adoption is the first month with at least one unique ERCP AIH. Maintenance is activity in at least six months of a rolling 12-month window; cessation is six consecutive inactive months while the hospital remains eligible. Thresholds of three or five AIHs and a three-consecutive-month definition are sensitivity analyses.

The adoption risk set excludes ineligible hospitals and previously adopted hospitals. Hospital eligibility is derived from same-month CNES inpatient status and pre-specified capability concepts. Official CNES semantic codes must be reviewed and frozen before the risk set is materialized.

### Aim 2: equity and geographic access

For the strict adult cohort, estimate standardized treated utilization, observed road travel time, and the share of the adult population within 120 road-minutes of an active provider. The primary structural exposure is continuous IVS rank and population-weighted IVS quintile. Municipality attributes remain contextual; no individual deprivation claim is permitted.

Primary inequality measures are absolute and relative contrasts plus SII and RII. The three primary equity endpoints form one Holm-adjusted family. The 180-minute threshold, cross-boundary care, and nearest-provider distance are secondary.

### Aim 3: referral networks and resilience

Build annual directed weighted residence-municipality to provider-CNES networks from unique AIHs. Primary metrics are weighted strength, HHI, betweenness, community structure, and cross-boundary flows. Resilience is the incremental adult population lacking an alternative provider within 180 minutes after removal of pre-specified high-in-strength hubs, benchmarked against random-node removal and repeated at 120 minutes.

### Aim 4: hospital organization and in-hospital outcomes

The primary outcome is in-hospital death in the strict adult cohort. The primary exposure is trailing-12-month all-indication hospital ERCP volume, modeled with restricted cubic splines. The model targets a standardized marginal risk difference with hospital-clustered uncertainty; marginal risk ratios and clearly labeled conditional odds ratios are supportive.

Covariates are fixed before outcome modeling and reflect patient case mix, hospital capacity, context, state, and calendar month. ICU use, length of stay, and other post-exposure mediators cannot adjust the mortality model. Stepwise selection and univariable screening are prohibited.

## 4. Data engineering and temporal alignment

SIH-SP is filtered before expansion to minimize data. Matching rows are collapsed to unique AIHs and linked to RD using competence month, CNES, and NAIH. Every partition records input rows, code hits, unique AIHs, duplicate patterns, RD matches, and output hash. CNES attributes are joined by the same competence month; future capacity cannot explain past adoption.

Municipality codes are normalized through a versioned IBGE crosswalk. Geography, Census, IVS, ANS, IPCA, COVID pressure, and health-region files require source URLs, access dates, licenses, sizes, and SHA-256 hashes.

## 5. Missing data and measurement error

Identifiers, procedure presence, adoption month, primary outcomes, and linkage keys are never imputed. Missingness is profiled by state, year, hospital, and cohort. Race/color uses an explicit missing category in the primary model because recording is likely institutionally patterned; multiple imputation is only a sensitivity analysis after semantic and MAR diagnostics.

Contextual exposure analyses require at least 95% target-population linkage. Road-time claims require at least 99% valid active-provider geocoding and 95% routing success for primary patient flows. Failure invokes the title and claim downgrades in `config/analysis_freeze.yaml`.

## 6. Model diagnostics and uncertainty

All models must converge, use explicit reference categories, report effect sizes and 95% confidence intervals, and account for hospital clustering or hierarchy. Binary-outcome models require separation checks and calibration plots. Sparse or separated models use penalized or simpler pre-specified alternatives; favorable Wald p-values cannot rescue a failed model.

Nonlinear exposure terms are assessed jointly. No hospital ranking is produced from unstable estimates. Subgroups are estimated through interaction terms, not by comparing significance across strata, and are labeled exploratory unless explicitly pre-specified.

## 7. Sensitivity and negative-control plan

Required sensitivity analyses include cohort A versus B, alternative diagnostic sources, adult/emergency restrictions, alternative adoption thresholds, broad/primary/strict eligible-hospital risk sets, 120/180-minute travel thresholds, route versus great-circle measures, provider CNES versus municipality networks, missingness approaches, and exclusion or explicit adjustment of the most disrupted COVID period.

Negative controls must be selected before outcome inspection and must be plausibly affected by coding or hospital reporting but not by the hypothesized access pathway. Candidate controls are not accepted until ontology and data-density review; a convenient null result cannot be selected post hoc.

## 8. Quasi-causal stopping rule

Staggered adoption is endogenous and does not create causal identification by itself. The module is causal-eligible only if the eligible risk set, adoption definition, overlap, event-time support, pre-trends, coding stability, COVID robustness, and negative controls all pass. One failed gate removes the module from the main paper; results may remain in a clearly exploratory supplement only when informative.

## 9. Reproducibility, privacy, and reporting

Every number, table, figure, abstract sentence, and manuscript claim traces to a frozen analysis table and model object. Plot scripts cannot reconstruct cohorts. Public outputs suppress cells below five and never release patient-level records. Reporting follows RECORD/STROBE and includes data provenance, codebooks, schema matrices, missingness heatmaps, model diagnostics, and every pre-specified sensitivity result.

The machine-readable contracts in `config/` are normative. Any amendment after hash freeze requires a dated explanation, disclosure of whether outcomes had been viewed, the affected estimand, and a new hash.

