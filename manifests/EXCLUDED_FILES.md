# Files not distributed in the public package

The public package excludes:

- all `data_raw`, `data_stage`, `data_analytic`, row-level SIH/SUS/CNES data, and routing caches;
- credentials, SSH helpers, `.env` files, host-specific secrets, and absolute workstation/server paths;
- writable copies of the result registry;
- temporary execution logs, working checkpoints, internal process records, correspondence drafts, and other files that are not needed to reproduce the reported results;
- superseded recovery scripts with hard-coded machine paths;
- scripts and outputs from analyses outside the descriptive and associational scope reported in the manuscript;
- superseded submission-file versions;
- large external OSM, IVS, ANS, SRAG, IBGE, and DATASUS source files; only safe acquisition manifests are included.

These exclusions prevent credential leakage, disclosure of restricted or record-level data, and accidental publication of non-reportable analyses. They also mean that the public repository is not a self-contained raw-data archive.
