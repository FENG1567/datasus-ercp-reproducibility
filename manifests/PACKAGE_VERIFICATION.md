# Public-package verification summary

Date: 2026-09-03

## Passed checks

- Frozen source-data manifest: 22/22 CSV hashes matched.
- Frozen source-data verification: all recorded contract, input-hash, privacy, registry, row-count, and unavailable-result boundary checks passed.
- Python static compilation: all copied `src/` and `scripts/` files compiled.
- Public test suite in the available Python 3.12 environment: 85 passed and 4 skipped; all test modules, including the geospatial modules, were collected.
- Aim 4 v2 runner/contract regression: 7 passed, 1 skipped.
- Aim 4 v3 runner/contract regression: 9 passed, 2 skipped.
- Enhanced editor-figure reproduction: 29 artifacts built; all seven PNG outputs were byte-identical and pixel-identical to the final reference figures.
- Final manuscript, supplement, table compendium, and figure files were copied byte-for-byte from the final editor package.
- Public-package content scan: no credential value, absolute server/workstation path, restricted raw/intermediate-data directory, or non-public working file was detected.
- Eight-thread configuration is present and enforced by the copied contracts/runners.

## Remaining reproducibility constraints

- One optional test requiring `statsmodels` and three R integration/parse tests were skipped because those dependencies were unavailable in the selected local validation runtime.
- The R renderer was not executed in this packaging environment because `Rscript` was unavailable. The included render manifest records R 4.5.3 and successful source parsing and rendering in the original analysis environment.
- A full raw-data-to-result run is not possible from this folder alone because record-level inputs and GraphHopper/OSM assets are deliberately excluded.

## Author/institution decisions still required

- archive a tagged release in a DOI-bearing repository and add the DOI;
- complete missing IVS/SRAG source URLs and access dates;
- capture a clean full environment lock (`pip freeze`, R `sessionInfo()`, Java/GraphHopper versions) during a clean rerun.
