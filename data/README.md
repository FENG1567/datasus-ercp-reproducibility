# Data directory

`source_data/` contains aggregate CSV files underlying the frozen figures and tables. `source_data_manifest.json` records their SHA-256 hashes and upstream provenance; `source_data_audit.json` records row-count, contract, privacy, and boundary checks.

`acquisition_manifests/` contains safe, public provenance records for external source files. It does not contain the files themselves.

No record-level or directly identifying data belong in this repository. Do not add `data_raw`, `data_stage`, `data_analytic`, unsuppressed counts of 1–4, routing caches, or local data paths.
