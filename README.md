# DATASUS ERCP study reproducibility package

This repository accompanies the nationwide study of therapeutic ERCP for choledocholithiasis in Brazil's Unified Health System, 2021–2025. It contains the frozen analysis code, scientific contracts, privacy-screened aggregate source data, final display files, and the manuscript materials needed to inspect and reproduce the reported figures and tables.

## Reproducibility scope

Two different claims are kept separate:

1. **Public source-data reproduction — available now.** The CSV files in `data/source_data/` support all seven frozen study figures and four frozen result tables. The final manuscript uses Figures 1–6 and Tables 1–2; former Figure 7 is Supplementary Figure 1, and former Tables 3–4 are Supplementary Tables 5–6. `python scripts/validate_release.py` verifies file presence, frozen source-data hashes, privacy boundaries, repository structure, and parallel-execution settings.
2. **Raw-data-to-result reproduction — conditional.** Record-level SIH/SUS and CNES files are not redistributed. A complete rerun requires reacquisition of the dated public source databases, the specified OpenStreetMap snapshot, GraphHopper 9.1, and an environment compatible with the original analysis. The exact remote Python/R package lock was not preserved; this limitation is documented in `environment/ENVIRONMENT_STATUS.md`.

The repository does not include `data_raw`, record-level administrative data, credentials, non-public working files, mutable intermediate registries, caches, or analyses outside the descriptive and associational scope reported in the manuscript.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python scripts/validate_release.py
python scripts/make_data_dictionary.py --check
python scripts/verify_checksums.py
```

GitHub Actions runs the same three release checks from
`.github/workflows/validate.yml` on every push and pull request. Use Python
3.12 (or another supported Python 3 runtime); the unqualified `python` command
must not resolve to Python 2.

With R 4.5.3 and the packages listed in `environment/install_r_packages.R`, the frozen publication renderer can be run from the repository root:

```bash
Rscript src/stage07_render_figures_tables_rc_v1_2.R \
  --project-root . \
  --source-dir data/source_data \
  --contract config/stage07_figures_tables_contract_rc_v1_2.yaml \
  --output-dir reproduced/stage07_displays
```

The enhanced editor versions of Figures 1, 3, 4, and 6 can be rebuilt with:

```bash
python scripts/reproduce_editor_figures.py --output-dir reproduced/editor_figures
```

Figures 2, 5, and Supplementary Figure 1 are unchanged frozen renderings and are copied from the verified reference artifacts by that command. See `docs/REPRODUCIBILITY.md` for the exact verification boundary.

## Repository map

- `src/`: public analysis and rendering scripts; legacy recovered code and scripts outside the reported analysis scope are excluded.
- `tests/`: frozen contract and analysis tests for the results reported in the manuscript.
- `config/`: scientific contracts and safe, relative-path configuration.
- `protocol/`: statistical analysis plan and the two prespecified Aim 4 amendments.
- `data/source_data/`: aggregate, privacy-screened figure/table source data and frozen manifests.
- `data/acquisition_manifests/`: dated source acquisition records that can be made public.
- `results/`: final editor figures and frozen result-table CSVs.
- `manuscript/`: final English and Chinese manuscripts, supplementary material, and table compendium.
- `references/`: field-level reference-verification records used for the manuscript.
- `manifests/`: checksums and release evidence.
- `docs/`: availability, provenance, data-source, display-crosswalk, and limitation notes.
- `.github/workflows/validate.yml`: public-package integrity checks.
- `GITHUB_UPLOAD_CHECKLIST.md`: upload boundary and author actions required before a public release.

## Scientific boundaries

- Results are descriptive or associational; no causal policy-effect claim is supported.
- `first observed coded use` is not a true adoption date.
- `observed uptake` is based on administrative coding.
- The patient-flow network describes observed treated flows.
- Reimbursement is not interpreted as cost.
- Resilience is a structural stress test, not an intervention counterfactual.
- Aim 2 primary-family results are supporting because prespecified quality criteria were not met. Aim 4 bootstrap uncertainty and formal confidence intervals are not reported for the same reason.
- Analyses outside the manuscript's descriptive and associational scope are not part of the paper or this public package.

## Licence and archival identifier

This package is deposited in a private GitHub repository at https://github.com/FENG1567/datasus-ercp-reproducibility. No software or aggregate-data licence has yet been authorised, and no archival DOI has been assigned. Do not add a licence by inference. Before making the repository public, the authors/institution must choose compatible code and aggregate-data licences and archive a tagged release in a DOI-bearing repository. See `LICENSE_SELECTION_REQUIRED.md` and `docs/CODE_AVAILABILITY.md`.

## Citation

Use `CITATION.cff` when citing this repository and add the archival DOI when assigned. The corresponding author is Xianzhi Meng (`mengxianzhi@hrbmu.edu.cn`).
