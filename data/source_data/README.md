# Frozen figure and table source data

These are aggregate, privacy-screened files from the accepted Stage 7 `rc_v1_2` release. Their hashes are recorded in `source_data_manifest.json`; the release checks are recorded in `source_data_audit.json`.

| Files | Display supported |
|---|---|
| `figure_1_policy_timeline_source_data.csv`, `figure_1_cohort_funnel_source_data.csv` | Main Figure 1 |
| `figure_2_monthly_uptake_source_data.csv`, `figure_2_first_observed_source_data.csv`, `figure_2_maintenance_source_data.csv` | Main Figure 2 |
| `figure_3_equity_source_data.csv`, `figure_3_travel_source_data.csv` | Main Figure 3 |
| `figure_4_municipality_coverage_source_data.csv`, `figure_4_national_regional_coverage_source_data.csv`, `figure_4_vulnerability_gap_source_data.csv` | Main Figure 4 |
| `figure_5_suppressed_network_edges_source_data.csv`, `figure_5_service_area_source_data.csv`, `figure_5_centrality_source_data.csv` | Main Figure 5 |
| `figure_6_adjusted_point_estimates_source_data.csv`, `figure_6_bootstrap_validity_source_data.csv`, `figure_6_sensitivity_status_source_data.csv` | Main Figure 6 |
| `figure_7_targeted_removal_source_data.csv`, `figure_7_random_benchmark_source_data.csv` | Supplementary Figure 1 (frozen name remains Figure 7) |
| `table_1_source_data.csv`, `table_2_source_data.csv` | Main Tables 1–2 |
| `table_3_source_data.csv`, `table_4_source_data.csv` | Supplementary Tables 5–6 |

Small-cell handling is encoded in display fields such as `n_aih_display`, `weighted_n_aih_display`, and `in_strength_display`; values 1–4 are not imputed. Municipality and hospital identifiers in these files are administrative/organizational identifiers, not patient identifiers. The source data must not be joined back to record-level files in a public repository.
