# Data sources and acquisition boundary

The repository redistributes only aggregate, privacy-screened source-data CSVs. Record-level SIH/SUS and CNES files are excluded and must be reacquired under the custodians' current terms.

| Source | Role | Reproduction record |
|---|---|---|
| DATASUS SIH/SUS | Hospital admissions and therapeutic ERCP cohorts | Acquisition scripts in `src/`; record-level files excluded |
| DATASUS CNES | Hospital characteristics and risk sets | Acquisition/audit scripts in `src/`; record-level files excluded |
| IBGE | Adult population, municipalities, and city-seat anchors | Reacquire the matching releases; record URL/date/hash in the template |
| Ipea/IVS | Municipality-level vulnerability context | File names and hashes are in `data/acquisition_manifests/ivs_official_download_manifest.json`; original URL/access date were not preserved in that manifest |
| ANS | Supplementary-insurance context | Dated URLs, access times, sizes, and hashes are in `ans_icb_manifest.csv` |
| OpenStreetMap/Geofabrik | Road network for routing | `brazil-260822.osm.pbf`, 2,074,198,072 bytes, SHA-256 `b3997da728ca224c6ed062e3adc203cd1e76be1afd02d017af12a075ee52f281`, URL recorded in the OSM manifest, ODbL 1.0 noted by the original manifest |
| GraphHopper | Route and isochrone engine | Version 9.1, profile `ercp_car`; see `config/graphhopper.example.yml` |
| SRAG/COVID series | Hospital-pressure sensitivity input | File names/hashes are recorded; original URLs/access dates were not preserved in the copied manifest |
| CONITEC, ordinances, SIGTAP | Policy and coding chronology | Public official sources; the frozen timeline is represented in figure source data and the SAP |

Before a raw-data rerun, complete every missing `source_url`, `version_or_release`, `accessed_at`, licence/terms field, file size, and SHA-256 value in `data/acquisition_manifest_template.csv`. Do not commit local paths or credentials.
