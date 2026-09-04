# Known limitations of the public package

1. Record-level DATASUS/CNES data are not redistributed, so a raw-data-to-result rerun requires reacquisition.
2. The exact remote Python/R dependency lock and Java build were not preserved. R 4.5.3 and GraphHopper 9.1 are documented; other dependency files are compatibility specifications.
3. The copied IVS and SRAG manifests contain hashes but not original source URLs/access dates; these fields should be completed from authoritative sources before archival deposition.
4. The GitHub repository is currently private; no archival DOI or software/data licence has yet been authorised.
5. Aim 2 primary-family and Aim 4 bootstrap evidence remain `DOWNGRADE`; Aim 4 formal confidence intervals remain `NOT_EVALUATED`.
6. Regional Figure 4 coverage and the vulnerability-gap component remain `NOT_EVALUATED` because the accepted frozen municipality-to-region and municipality-level IVS joins were unavailable.
7. The variable-level missingness decomposition for 2,597 Aim 4-excluded records is not available from an accepted frozen audit.
