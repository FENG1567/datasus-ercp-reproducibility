# GitHub upload checklist

Use the contents of this folder as the repository root. Upload every tracked
file and directory, including the hidden `.github/` directory. Do not upload
the parent folder or any adjacent project folders.

## Ready to upload

- analysis and rendering code in `src/` and `scripts/`;
- public tests and frozen scientific contracts;
- 22 privacy-screened aggregate source-data CSV files and their hashes;
- final figures, result tables, manuscripts, supplement, and table compendium;
- acquisition manifests, data dictionary, data/code availability statements,
  reproducibility guide, environment specifications, and citation metadata;
- deterministic release manifest and checksum list;
- GitHub Actions validation workflow.

## Validate before each upload or tagged release

Run from the repository root with Python 3.12:

```bash
python scripts/validate_release.py
python scripts/make_data_dictionary.py --check
python scripts/verify_checksums.py
python -m pytest tests -q
```

If an authorised file is changed, first regenerate the inventory and checksums:

```bash
python scripts/generate_release_manifest.py
```

Then rerun all validation commands. Do not manually edit
`manifests/checksums.sha256` or `manifests/public_release_manifest_v1.json`.

## Required author or institutional decisions before public release

- approve software and aggregate-data licences and add the resulting licence
  files; do not infer a licence from this package;
- make the current private repository public only after the software and
  aggregate-data licences have been approved;
- after archiving a tagged release, add the assigned DOI rather than a
  placeholder DOI;
- complete the missing IVS and SRAG source URLs/access dates in the acquisition
  records when authoritative values are available;
- during a clean full rerun, capture exact Python, R, Java, GraphHopper, and OS
  versions to replace the current compatibility-only environment record.

## Never upload

Do not add record-level DATASUS/CNES files, `data_raw`, `data_stage`,
`data_analytic`, routing caches, credentials, SSH helpers, non-public working
files, temporary logs, mutable intermediate registries, or analyses outside
the descriptive and associational scope reported in the manuscript. See `manifests/EXCLUDED_FILES.md` for the complete boundary.

## Reproducibility claim supported by this package

The package supports immediate hash verification of the frozen aggregate
source data and regeneration of the published displays. A full
raw-data-to-result rerun remains conditional on reacquiring the dated public
inputs and GraphHopper/OpenStreetMap assets; the repository is not a raw-data
archive and does not preserve an exact historical package lock.
