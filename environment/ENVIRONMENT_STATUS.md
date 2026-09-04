# Environment status

- Frozen display manifest: R 4.5.3.
- Frozen routing service: GraphHopper 9.1 with profile `ercp_car`.
- Release-packaging validation environment: Python 3.12.13; the verified subset is in `requirements-recorded-local.txt`.
- The complete remote Python and R package lock was not preserved. `requirements.txt`, `environment.yml`, and `install_r_packages.R` are compatibility specifications, not proof of byte-identical historical environments.

This prevents a defensible claim of fully locked raw-to-result computational reproduction. The aggregate source-data package and its hashes remain independently verifiable. Before DOI deposition, run the pipeline in a clean environment, save `python -m pip freeze`, R `sessionInfo()`, Java version, GraphHopper checksum/version, OS information, and the generated output hashes.
