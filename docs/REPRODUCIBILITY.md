# Reproducibility guide

## Level A: frozen source-data verification

Run from the repository root:

```bash
python scripts/validate_release.py
python scripts/make_data_dictionary.py --check
```

These commands verify that all 22 frozen figure/table CSV files are present, match the hashes in `source_data_manifest.json`, retain the declared privacy-suppression boundary, and correspond to the final display numbering.

## Level B: display regeneration

The R renderer regenerates the seven frozen Stage 7 figures and four result tables from aggregate source data only. It performs no model fitting and never reads raw records. R 4.5.3 was recorded in the accepted render manifest, but exact package versions were not preserved.

The Python editor renderer regenerates the enhanced versions of Figures 1, 3, 4, and 6. Figures 2, 5, and Supplementary Figure 1 were not scientifically or visually changed during the editor-package revision; the script copies the hash-verified reference artifacts supplied in `results/figures/`.

Generated vector or PDF files may not be byte-identical across operating systems because fonts, PDF metadata, and graphics backends differ. Validate the source-data hashes, panel content, dimensions, labels, and numeric values rather than requiring cross-platform PDF/TIFF byte identity.

## Level C: raw-data-to-result rerun

This level is conditional and is not self-contained in GitHub. Required external inputs include SIH/SUS, CNES, IBGE population/municipality data, Ipea/IVS, ANS beneficiary data, OpenStreetMap, policy/SIGTAP sources, and the COVID-pressure series used by the frozen protocol. Populate `data/acquisition_manifest_template.csv`, check every file hash, place local paths outside Git, and follow the SAP and configuration gates.

The full analysis order is:

1. download/probe and convert SIH/SUS and CNES sources;
2. build and freeze Cohorts A and B;
3. construct hospital-month risk sets and Aim 1 outputs;
4. prepare ecological equity inputs and GraphHopper routes/coverage for Aim 2;
5. build Aim 3 patient-flow, service-area, potential-access, and structural-resilience outputs;
6. build Aim 4 complete-context analytic data and prespecified bias-reduced point estimates; retain the `DOWNGRADE`/`NOT_EVALUATED` bootstrap gate;
7. build the result registry and frozen figure/table source-data package;
8. run tests and release validation.

Stage 6 quasi-causal scripts and outputs are deliberately excluded because they are not reportable in the manuscript.

## Thread limit

Load `config/threading.env`. The total process envelope must not exceed eight threads; nested BLAS/OpenMP libraries remain at one thread.
