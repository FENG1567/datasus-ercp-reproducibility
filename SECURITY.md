# Security and data-governance notice

Do not commit credentials, `.env` files, private keys, database cookies, SSH configuration, local/server absolute paths, record-level DATASUS files, or unsuppressed small-cell extracts.

The public package is limited to aggregate source data that passed the frozen privacy audit. If a suspected credential or sensitive record is found, keep the repository private, remove the item from history before publication, rotate any affected credential, and re-run `python scripts/validate_release.py`.
