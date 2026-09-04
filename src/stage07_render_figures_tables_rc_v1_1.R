#!/usr/bin/env Rscript

# R-only renderer for the frozen Stage 7 rc_v1_1 source-data release.
# It performs no statistical refitting and never reads data_raw.

parse_args <- function(x) {
  out <- list()
  i <- 1L
  while (i <= length(x)) {
    if (!startsWith(x[[i]], "--") || i == length(x)) stop("arguments must be --name value pairs")
    out[[substring(x[[i]], 3L)]] <- x[[i + 1L]]
    i <- i + 2L
  }
  out
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
required_args <- c("project-root", "source-dir", "contract", "output-dir")
if (!all(required_args %in% names(args))) stop("required: --project-root --source-dir --contract --output-dir")

project_root <- normalizePath(args[["project-root"]], mustWork = TRUE)
source_dir <- normalizePath(args[["source-dir"]], mustWork = TRUE)
contract_path <- normalizePath(args[["contract"]], mustWork = TRUE)
output_dir <- file.path(project_root, args[["output-dir"]])

expected_contract_sha <- "42499ded0f4a58db683aff43f2bb50e8e60c06bd5c7733394f7e06bc0a71e7db"
expected_source_manifest_sha <- "6b75b8413b09991324f24ea45910b8c5ee107bc3387b6580b35ab3fb3a9136f4"
expected_source_audit_sha <- "41860172666041982b8a912d65dfdd4b9c4eb048061362e497f2081ecccc1ab2"

sha256_file <- function(path) {
  answer <- system2("sha256sum", path, stdout = TRUE, stderr = TRUE)
  if (length(answer) != 1L) stop("sha256sum failed for ", path)
  strsplit(trimws(answer), "[[:space:]]+")[[1L]][[1L]]
}

if (sha256_file(contract_path) != expected_contract_sha) stop("contract SHA-256 mismatch")
if (sha256_file(file.path(source_dir, "source_data_manifest.json")) != expected_source_manifest_sha) stop("source manifest SHA-256 mismatch")
if (sha256_file(file.path(source_dir, "source_data_audit.json")) != expected_source_audit_sha) stop("source audit SHA-256 mismatch")

packages <- c("ggplot2", "patchwork", "svglite", "ragg")
missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) stop("missing R packages: ", paste(missing_packages, collapse = ", "))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))

expected_csv <- c(
  "figure_1_policy_timeline_source_data.csv", "figure_1_cohort_funnel_source_data.csv",
  "figure_2_monthly_uptake_source_data.csv", "figure_2_first_observed_source_data.csv", "figure_2_maintenance_source_data.csv",
  "figure_3_equity_source_data.csv", "figure_3_travel_source_data.csv",
  "figure_4_municipality_coverage_source_data.csv", "figure_4_national_regional_coverage_source_data.csv", "figure_4_vulnerability_gap_source_data.csv",
  "figure_5_suppressed_network_edges_source_data.csv", "figure_5_service_area_source_data.csv", "figure_5_centrality_source_data.csv",
  "figure_6_adjusted_point_estimates_source_data.csv", "figure_6_bootstrap_validity_source_data.csv", "figure_6_sensitivity_status_source_data.csv",
  "figure_7_targeted_removal_source_data.csv", "figure_7_random_benchmark_source_data.csv",
  "table_1_source_data.csv", "table_2_source_data.csv", "table_3_source_data.csv", "table_4_source_data.csv"
)
missing_csv <- expected_csv[!file.exists(file.path(source_dir, expected_csv))]
if (length(missing_csv)) stop("missing frozen source CSVs: ", paste(missing_csv, collapse = ", "))
if (dir.exists(output_dir) || file.exists(output_dir)) stop("output directory already exists; refusing overwrite")

parent_dir <- dirname(output_dir)
dir.create(parent_dir, recursive = TRUE, showWarnings = FALSE)
stage_dir <- tempfile(pattern = paste0(".", basename(output_dir), ".staging-"), tmpdir = parent_dir)
dir.create(stage_dir, recursive = FALSE)
committed <- FALSE
on.exit(if (!committed && dir.exists(stage_dir)) unlink(stage_dir, recursive = TRUE, force = TRUE), add = TRUE)

coerce_csv_logicals <- function(data) {
  for (name in names(data)) {
    values <- data[[name]]
    if (!is.character(values)) next
    normalized <- tolower(trimws(values))
    observed <- !is.na(normalized)
    if (any(observed) && all(normalized[observed] %in% c("true", "false"))) {
      data[[name]] <- ifelse(is.na(normalized), NA, normalized == "true")
    }
  }
  data
}
read_source <- function(name) {
  data <- read.csv(file.path(source_dir, name), stringsAsFactors = FALSE, check.names = FALSE, na.strings = c("NA", ""))
  coerce_csv_logicals(data)
}
wrap_text <- function(x, width = 35L) vapply(x, function(z) paste(strwrap(as.character(z), width = width), collapse = "\n"), character(1))
as_display_number <- function(x) suppressWarnings(as.numeric(as.character(x)))
fmt_count <- function(x) format(round(x), big.mark = ",", scientific = FALSE, trim = TRUE)
fmt_million <- function(x) paste0(format(round(x / 1e6, 1), trim = TRUE, nsmall = 1), "m")
fmt_value <- function(x) {
  y <- suppressWarnings(as.numeric(as.character(x)))
  ifelse(is.na(y), as.character(x), ifelse(abs(y) >= 1000, fmt_count(y), format(signif(y, 4), trim = TRUE, scientific = FALSE)))
}

palette <- c(blue = "#2C7FB8", teal = "#2A9D8F", orange = "#E28E2C", red = "#C94C4C", grey = "#6B7280", light = "#DCE6EE", dark = "#1F2937")
figure_width_mm = 183
theme_pub <- function() {
  theme_classic(base_size = 6.5, base_family = "sans") +
    theme(
      axis.line = element_line(linewidth = 0.32, colour = "black"),
      axis.ticks = element_line(linewidth = 0.32),
      axis.title = element_text(size = 6.5), axis.text = element_text(size = 5.8),
      legend.title = element_text(size = 6.2), legend.text = element_text(size = 5.7),
      strip.text = element_text(size = 6.2, face = "bold"),
      plot.title = element_text(size = 7.2, face = "bold"), plot.subtitle = element_text(size = 5.9),
      plot.caption = element_text(size = 5.2, colour = palette[["grey"]], hjust = 0),
      plot.tag = element_text(size = 8, face = "bold"), panel.grid = element_blank()
    )
}
theme_set(theme_pub())

save_plot <- function(plot, stem, width_mm, height_mm) {
  w <- width_mm / 25.4; h <- height_mm / 25.4
  svg <- file.path(stage_dir, paste0(stem, ".svg")); pdf <- file.path(stage_dir, paste0(stem, ".pdf")); tif <- file.path(stage_dir, paste0(stem, ".tiff"))
  svglite::svglite(svg, width = w, height = h, bg = "white"); print(plot); grDevices::dev.off()
  grDevices::cairo_pdf(pdf, width = w, height = h, family = "sans", bg = "white"); print(plot); grDevices::dev.off()
  ragg::agg_tiff(tif, width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); print(plot); grDevices::dev.off()
}

text_panel <- function(title, lines, colour = palette[["dark"]]) {
  body <- paste(wrap_text(lines, width = 42L), collapse = "\n")
  ggplot() + annotate("text", x = 0, y = 1, label = title, hjust = 0, vjust = 1, size = 2.5, fontface = "bold", colour = colour) +
    annotate("text", x = 0, y = 0.84, label = body, hjust = 0, vjust = 1, size = 2.05, lineheight = 1.15, colour = palette[["dark"]]) +
    xlim(0, 1) + ylim(0, 1) + theme_void()
}

# Figure 1: policy evidence, observation boundary, and cohort funnel.
policy <- read_source("figure_1_policy_timeline_source_data.csv")
policy$date_plot <- as.Date(ifelse(nchar(policy$date) == 7L, paste0(policy$date, "-01"), policy$date))
policy$label <- c("CONITEC recommendation", "SCTIE inclusion decision", "GM/MS coding and financing", "SIGTAP observed window")[seq_len(nrow(policy))]
policy$label_y <- rep(c(0.18, -0.18), length.out = nrow(policy))
policy$display_label <- wrap_text(paste(policy$date, policy$label, sep = "\n"), width = 24L)
p1a <- ggplot(policy, aes(date_plot, 0)) + geom_hline(yintercept = 0, colour = palette[["grey"]], linewidth = 0.4) +
  geom_segment(aes(xend = date_plot, yend = label_y), linewidth = 0.3, colour = palette[["grey"]]) +
  geom_point(size = 2.2, colour = palette[["blue"]]) + geom_text(aes(y = label_y, label = display_label), size = 2.0, lineheight = 0.95) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y", expand = expansion(mult = c(0.08, 0.12))) + coord_cartesian(ylim = c(-0.42, 0.42), clip = "off") +
  labs(title = "Official policy and coding evidence", x = NULL, y = NULL, caption = "Dates and milestones are transcribed from the hash-locked policy evidence table.") +
  theme(axis.line.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())
funnel <- read_source("figure_1_cohort_funnel_source_data.csv")
funnel <- funnel[funnel$result_id %in% c("cohort.a_unique_aih", "cohort.b_final_unique_aih", "uptake.left_censored_202101"), ]
funnel$label <- wrap_text(funnel$display_label, 24)
p1b <- ggplot(funnel, aes(reorder(label, value), as.numeric(value), fill = result_id)) + geom_col(width = 0.64, show.legend = FALSE) + coord_flip() +
  geom_text(aes(label = fmt_count(as.numeric(value))), hjust = -0.08, size = 2.1) + scale_fill_manual(values = c(palette[["blue"]], palette[["teal"]], palette[["orange"]])) +
  expand_limits(y = max(as.numeric(funnel$value)) * 1.18) + labs(title = "Frozen cohort and boundary counts", x = NULL, y = "Records or hospitals")
p1c <- text_panel("Observation boundary", c("January 2021 is left-censored.", "Use: first observed coded use; observed uptake.", "Do not interpret the boundary as national implementation."), palette[["orange"]])
fig1 <- p1a / (p1b | p1c) + plot_layout(heights = c(1.45, 1)) + plot_annotation(tag_levels = "a", title = "Policy, coding, cohort, and observation-window timeline")
save_plot(fig1, "Figure_1", 183, 120)

# Figure 2: observed uptake and maintenance within the eligible risk set.
uptake <- read_source("figure_2_monthly_uptake_source_data.csv")
monthly <- uptake[uptake$source_dataset == "monthly_observed_uptake", ]
monthly$month <- as.Date(paste0(substr(monthly$competence_month, 1, 4), "-", substr(monthly$competence_month, 5, 6), "-01"))
p2a <- ggplot(monthly, aes(month, n_unique_aih, colour = cohort)) + geom_line(linewidth = 0.55) +
  geom_vline(xintercept = as.Date("2021-01-01"), linetype = 2, colour = palette[["orange"]]) +
  annotate("text", x = as.Date("2021-03-01"), y = max(monthly$n_unique_aih, na.rm = TRUE), label = "left-censored boundary", hjust = 0, size = 2.0, colour = palette[["orange"]]) +
  scale_colour_manual(values = c(A = palette[["blue"]], B = palette[["teal"]])) + labs(title = "Monthly observed uptake", x = NULL, y = "Unique AIHs", colour = "Cohort")
annual <- uptake[uptake$source_dataset == "annual_rate_metadata", ]
p2b <- ggplot(annual, aes(year, rate_per_100k, colour = cohort)) + geom_line(linewidth = 0.55) + geom_point(size = 1.5) +
  scale_colour_manual(values = c(A = palette[["blue"]], B = palette[["teal"]])) + labs(title = "Population-standardised descriptive rate", x = "Year", y = "AIHs per 100,000 adults", colour = "Cohort")
first <- read_source("figure_2_first_observed_source_data.csv")
p2c <- ggplot(first, aes(factor(first_observed_year), n_hospitals)) + geom_col(fill = palette[["blue"]], width = 0.68) + geom_text(aes(label = n_hospitals), vjust = -0.25, size = 2.0) +
  labs(title = "First observed coded use", subtitle = "Not a true adoption date", x = "Year", y = "Hospitals")
maintenance <- read_source("figure_2_maintenance_source_data.csv")
maintenance_summary <- data.frame(
  state = c("Maintained ≥6/12", "Maintained first completed 12m", "Maintained last evaluable window"),
  proportion = c(mean(maintenance$ever_maintained_6of12, na.rm = TRUE), mean(maintenance$maintained_first_completed_12m, na.rm = TRUE), mean(maintenance$maintained_last_evaluable_window, na.rm = TRUE))
)
p2d <- ggplot(maintenance_summary, aes(reorder(state, proportion), proportion)) + geom_col(fill = palette[["teal"]], width = 0.64) + coord_flip() +
  geom_text(aes(label = paste0(round(100 * proportion, 1), "%")), hjust = -0.08, size = 2.0) + scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"), limits = c(0, 1.08)) +
  labs(title = "Maintenance within eligible hospitals", subtitle = paste0("Risk-set hospitals: ", nrow(maintenance)), x = NULL, y = "Proportion")
fig2 <- (p2a | p2b) / (p2c | p2d) + plot_layout(heights = c(1.35, 1)) + plot_annotation(tag_levels = "a", title = "National observed uptake, diffusion, and service maintenance")
save_plot(fig2, "Figure_2", 183, 125)

# Figure 3: ecological equity gradients and realised treated-flow road time.
equity <- read_source("figure_3_equity_source_data.csv")
equity$exposure_short <- ifelse(grepl("IVS", equity$exposure), "IVS (ecological)", "Supplementary insurance (ecological)")
p3a <- ggplot(equity, aes(quintile, age_sex_standardised_rate_per_100k, colour = exposure_short)) +
  geom_errorbar(aes(ymin = standardised_lo95, ymax = standardised_hi95), width = 0.12, linewidth = 0.35) + geom_point(size = 1.7) + geom_line(linewidth = 0.45) +
  scale_colour_manual(values = c(palette[["blue"]], palette[["orange"]])) + labs(title = "Directly standardised treated-use rates", x = "Contextual quintile", y = "AIHs per 100,000 adults", colour = NULL)
travel <- read_source("figure_3_travel_source_data.csv")
travel$known_weight <- as_display_number(travel$weighted_n_aih_display)
travel_known <- travel[!is.na(travel$known_weight), ]
travel_sum <- aggregate(known_weight ~ travel_bin_label, travel_known, sum)
travel_sum$order <- ifelse(travel_sum$travel_bin_label == "NOT_ROUTED", -1, suppressWarnings(as.numeric(sub("^([0-9]+).*$", "\\1", travel_sum$travel_bin_label))))
travel_sum <- travel_sum[order(travel_sum$order), ]; travel_sum$travel_bin_label <- factor(travel_sum$travel_bin_label, levels = travel_sum$travel_bin_label)
p3b <- ggplot(travel_sum, aes(travel_bin_label, known_weight)) + geom_col(fill = palette[["teal"]]) +
  geom_vline(xintercept = which(levels(travel_sum$travel_bin_label) %in% c("120–135", "120-135")) - 0.5, linetype = 2, colour = palette[["red"]]) +
  labs(title = "Realised flow-weighted road-time distribution", subtitle = paste0(sum(is.na(travel$known_weight)), " privacy-suppressed strata are not imputed"), x = "15-minute bin", y = "Known weighted AIHs") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
p3c <- text_panel("Interpretation boundary", c("Ecological contextual gradients are associational.", "Road time describes observed treated flows.", "Neither panel estimates population access or individual effects."), palette[["orange"]])
fig3 <- p3a / (p3b | p3c) + plot_layout(heights = c(1.35, 1)) + plot_annotation(tag_levels = "a", title = "Equity gradients and realised travel burden")
save_plot(fig3, "Figure_3", 183, 120)

# Figure 4: potential access. Frozen geometry/region/IVS joins are absent and remain NOT_EVALUATED.
coverage <- read_source("figure_4_national_regional_coverage_source_data.csv")
national <- coverage[coverage$scope == "national" & coverage$status == "EVALUATED", ]
coverage_long <- rbind(
  data.frame(year = national$year, threshold = "120 minutes", share = national$adult_population_share_120),
  data.frame(year = national$year, threshold = "Cumulative 180 minutes", share = national$adult_population_share_180)
)
p4a <- ggplot(coverage_long, aes(year, share, colour = threshold)) + geom_line(linewidth = 0.65) + geom_point(size = 1.7) +
  scale_colour_manual(values = c("120 minutes" = palette[["blue"]], "Cumulative 180 minutes" = palette[["teal"]])) +
  scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"), limits = c(0, 1)) + labs(title = "Adult-population potential access", x = "Year", y = "Population share", colour = NULL)
municipality <- read_source("figure_4_municipality_coverage_source_data.csv")
latest <- municipality[municipality$year == max(municipality$year), ]
latest$category <- ifelse(!latest$anchor_available, "Missing anchor", ifelse(latest$has_provider_120, "≤120 minutes", ifelse(latest$has_provider_180, "120–180 minutes", ">180 minutes")))
cat_pop <- aggregate(adult_population ~ category, latest, sum); cat_pop$share <- cat_pop$adult_population / sum(latest$adult_population)
p4b <- ggplot(cat_pop, aes(reorder(category, share), share, fill = category)) + geom_col(show.legend = FALSE, width = 0.68) + coord_flip() +
  geom_text(aes(label = paste0(round(100 * share, 1), "%")), hjust = -0.08, size = 2.0) + scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"), limits = c(0, max(cat_pop$share) * 1.15)) +
  scale_fill_manual(values = c("≤120 minutes" = palette[["blue"]], "120–180 minutes" = palette[["teal"]], ">180 minutes" = palette[["orange"]], "Missing anchor" = palette[["grey"]])) +
  labs(title = paste0(max(latest$year), " anchored population categories"), x = NULL, y = "Adult-population share")
not_eval <- read_source("figure_4_vulnerability_gap_source_data.csv")
p4c <- text_panel("Frozen-input boundary", c("Geographic maps: NOT_EVALUATED (no frozen geometry).", "Regional coverage: NOT_EVALUATED (no region mapping).", "Vulnerability gap: NOT_EVALUATED (no IVS join).", "Cumulative 180-minute rule retained; potential access is not realised use, capacity, quality, or need."), palette[["orange"]])
fig4 <- p4a / (p4b | p4c) + plot_layout(heights = c(1.4, 1)) + plot_annotation(tag_levels = "a", title = "Potential population access within 120 and 180 minutes")
save_plot(fig4, "Figure_4", 183, 135)

# Figure 5: privacy-suppressed patient-flow network summaries without invented spatial coordinates.
edges <- read_source("figure_5_suppressed_network_edges_source_data.csv")
edges$known_flow <- as_display_number(edges$n_aih_display)
hub <- aggregate(known_flow ~ treat_municipio, edges[!is.na(edges$known_flow) & !is.na(edges$treat_municipio), ], sum)
hub <- hub[order(hub$known_flow, decreasing = TRUE), ]; hub <- head(hub, 20); hub$hub <- factor(as.character(hub$treat_municipio), levels = rev(as.character(hub$treat_municipio)))
p5a <- ggplot(hub, aes(hub, known_flow)) + geom_col(fill = palette[["blue"]], width = 0.7) + coord_flip() +
  scale_y_continuous(labels = fmt_count) + labs(title = "Largest observed destination hubs", subtitle = paste0(sum(is.na(edges$known_flow)), " suppressed edge strata excluded; no value imputed"), x = "Treatment municipality", y = "Known AIHs")
service <- read_source("figure_5_service_area_source_data.csv")
service_obs <- service[service$observed_treatment %in% TRUE & !is.na(service$cross_municipality_share), ]
service_summary <- aggregate(cross_municipality_share ~ cohort + year, service_obs, median)
p5b <- ggplot(service_summary, aes(year, cross_municipality_share, colour = cohort, group = cohort)) + geom_line(linewidth = 0.5) + geom_point(size = 1.4) +
  scale_colour_manual(values = c(A = palette[["blue"]], B = palette[["teal"]])) + scale_y_continuous(labels = function(x) paste0(round(100 * x), "%"), limits = c(0, 1)) +
  labs(title = "Observed cross-municipality flow", subtitle = "Median among municipalities with observed treatment", x = NULL, y = "Share", colour = "Cohort")
centrality <- read_source("figure_5_centrality_source_data.csv")
p5c <- ggplot(centrality, aes(weighted_betweenness_inverse_flow_cost, colour = cohort)) + stat_ecdf(linewidth = 0.5) +
  scale_x_continuous(trans = "log1p", breaks = c(0, 1e3, 1e4, 1e5, 1e6), labels = c("0", "1k", "10k", "100k", "1m")) +
  scale_colour_manual(values = c(A = palette[["blue"]], B = palette[["teal"]])) + labs(title = "Patient-flow network centrality", subtitle = "Empirical distribution; coordinates were not supplied", x = "Weighted betweenness (log1p)", y = "Cumulative share", colour = "Cohort")
fig5 <- p5a / (p5b | p5c) + plot_layout(heights = c(1.35, 1)) + plot_annotation(tag_levels = "a", title = "Intermunicipal patient-flow network and referral concentration", subtitle = "Observed flows do not establish poor access or inappropriate referral.")
save_plot(fig5, "Figure_5", 183, 130)

# Figure 6: associational point estimates only; no accepted bootstrap intervals.
points <- read_source("figure_6_adjusted_point_estimates_source_data.csv")
points$metric_label <- factor(points$metric, levels = c("risk_p10", "risk_p90", "rd", "rr"))
p6a <- ggplot(points, aes(estimator, value, colour = estimator)) + geom_point(size = 2.0, show.legend = FALSE) + facet_wrap(~metric_label, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = c(AS_mean = palette[["blue"]], MPL_Jeffreys = palette[["teal"]])) + labs(title = "Associational point estimates only", subtitle = "Formal intervals: NOT_EVALUATED", x = NULL, y = "Point estimate")
boot <- read_source("figure_6_bootstrap_validity_source_data.csv")
boot_valid <- boot[grepl("valid_replicates", boot$result_id), ]; boot_valid$valid <- as.numeric(boot_valid$value); boot_valid$estimator <- ifelse(grepl("as_mean", boot_valid$result_id), "AS_mean", "MPL_Jeffreys")
p6b <- ggplot(boot_valid, aes(estimator, valid, fill = estimator)) + geom_col(width = 0.62, show.legend = FALSE) + geom_hline(yintercept = 2000, linetype = 2, colour = palette[["grey"]]) +
  geom_text(aes(label = paste0(valid, "/2000")), vjust = -0.3, size = 2.1) + scale_fill_manual(values = c(AS_mean = palette[["blue"]], MPL_Jeffreys = palette[["teal"]])) +
  scale_y_continuous(limits = c(0, 2150)) + labs(title = "Bootstrap valid-replicate yield", subtitle = "Prespecified gate: DOWNGRADE", x = NULL, y = "Valid replicates")
status <- read_source("figure_6_sensitivity_status_source_data.csv")
p6c <- text_panel("Inference boundary", c("Bootstrap gate: DOWNGRADE.", "Accepted confidence intervals: NOT_EVALUATED.", "No rerun or model relaxation.", "Descriptive/associational evidence only."), palette[["red"]])
fig6 <- p6a / (p6b | p6c) + plot_layout(heights = c(1.25, 1)) + plot_annotation(tag_levels = "a", title = "Hospital attributes and in-hospital outcomes", subtitle = "Associational point estimates; no causal interpretation.")
save_plot(fig6, "Figure_6", 183, 120)

# Figure 7: structural stress test, not a real-world intervention counterfactual.
targeted <- read_source("figure_7_targeted_removal_source_data.csv")
p7a <- ggplot(targeted, aes(n_removed, newly_uncovered_complete_case, colour = factor(threshold_minutes))) + geom_line(linewidth = 0.55) +
  scale_colour_manual(values = c(`120` = palette[["blue"]], `180` = palette[["teal"]])) + scale_y_continuous(labels = fmt_million) +
  labs(title = "Targeted sequential observed-hub removal", x = "Hubs removed", y = "Newly uncovered adult population", colour = "Threshold (min)")
random <- read_source("figure_7_random_benchmark_source_data.csv")
p7b <- ggplot(random, aes(n_removed, random_newly_uncovered_mean, colour = factor(threshold_minutes), fill = factor(threshold_minutes))) +
  geom_ribbon(aes(ymin = random_newly_uncovered_p2_5, ymax = random_newly_uncovered_p97_5), alpha = 0.16, colour = NA) + geom_line(linewidth = 0.5) +
  geom_point(aes(y = targeted_newly_uncovered_complete_case), shape = 21, size = 1.7, fill = "white") +
  scale_colour_manual(values = c(`120` = palette[["blue"]], `180` = palette[["teal"]])) + scale_fill_manual(values = c(`120` = palette[["blue"]], `180` = palette[["teal"]])) + scale_y_continuous(labels = fmt_million) +
  labs(title = "1,000-replicate random-removal envelope", subtitle = "Envelope is a structural benchmark, not a confidence interval", x = "Hubs removed", y = "Newly uncovered adult population", colour = "Threshold", fill = "Threshold")
p7c <- ggplot(random, aes(factor(n_removed), targeted_minus_random_mean / 1e6, fill = factor(threshold_minutes))) + geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linewidth = 0.3) + scale_fill_manual(values = c(`120` = palette[["blue"]], `180` = palette[["teal"]])) +
  labs(title = "Targeted minus random mean", subtitle = "Structural stress test; not a real-world counterfactual", x = "Hubs removed", y = "Difference (millions)", fill = "Threshold")
fig7 <- p7a / (p7b | p7c) + plot_layout(heights = c(1.3, 1)) + plot_annotation(tag_levels = "a", title = "Network resilience under observed hub removal versus random removal")
save_plot(fig7, "Figure_7", 183, 120)

# Tables: compact rendered views plus exact frozen CSV exports.
table_specs <- list(
  Table_1 = list(title = "Cohort, hospital, municipality, and regional characteristics", height = 105, source = "table_1_source_data.csv"),
  Table_2 = list(title = "Observed uptake and maintenance of therapeutic ERCP", height = 105, source = "table_2_source_data.csv"),
  Table_3 = list(title = "Equity, travel time, potential access, and patient-flow concentration", height = 130, source = "table_3_source_data.csv"),
  Table_4 = list(title = "Adjusted hospital attributes and in-hospital outcome associations", height = 105, source = "table_4_source_data.csv")
)

draw_table_page <- function(data, title) {
  grid::grid.newpage()
  grid::grid.text(title, x = 0.02, y = 0.965, just = c("left", "top"), gp = grid::gpar(fontfamily = "sans", fontsize = 8, fontface = "bold"))
  headers <- c("Measure", "Value", "Denominator", "Evidence")
  x <- c(0.02, 0.56, 0.73, 0.90)
  grid::grid.rect(x = 0.5, y = 0.89, width = 0.97, height = 0.055, gp = grid::gpar(fill = "#E8EEF3", col = NA))
  for (j in seq_along(headers)) grid::grid.text(headers[[j]], x = x[[j]], y = 0.89, just = c("left", "center"), gp = grid::gpar(fontfamily = "sans", fontsize = 6, fontface = "bold"))
  n <- nrow(data); top <- 0.845; bottom <- 0.09; row_h <- (top - bottom) / max(n, 1)
  for (i in seq_len(n)) {
    y <- top - (i - 0.5) * row_h
    if (i %% 2L == 0L) grid::grid.rect(x = 0.5, y = y, width = 0.97, height = row_h, gp = grid::gpar(fill = "#F7F9FA", col = NA))
    value <- paste(fmt_value(data$value[[i]]), data$unit[[i]])
    denom <- if (is.na(data$denominator_value[[i]])) "—" else paste(fmt_value(data$denominator_value[[i]]), data$denominator_unit[[i]])
    cells <- c(
      wrap_text(data$display_label[[i]], 43),
      wrap_text(value, 20),
      wrap_text(denom, 25),
      wrap_text(data$evidence_level[[i]], 14)
    )
    for (j in seq_along(cells)) grid::grid.text(cells[[j]], x = x[[j]], y = y, just = c("left", "center"), gp = grid::gpar(fontfamily = "sans", fontsize = 5.0))
  }
  grid::grid.text("Values are frozen registry outputs. See source CSV for limitations, suppression status, and provenance.", x = 0.02, y = 0.035, just = c("left", "bottom"), gp = grid::gpar(fontfamily = "sans", fontsize = 5.2, col = palette[["grey"]]))
}

save_table <- function(data, stem, title, width_mm, height_mm) {
  w <- width_mm / 25.4; h <- height_mm / 25.4
  svg <- file.path(stage_dir, paste0(stem, ".svg")); pdf <- file.path(stage_dir, paste0(stem, ".pdf")); tif <- file.path(stage_dir, paste0(stem, ".tiff"))
  svglite::svglite(svg, width = w, height = h, bg = "white"); draw_table_page(data, title); grDevices::dev.off()
  grDevices::cairo_pdf(pdf, width = w, height = h, family = "sans", bg = "white"); draw_table_page(data, title); grDevices::dev.off()
  ragg::agg_tiff(tif, width = w, height = h, units = "in", res = 600, compression = "lzw", background = "white"); draw_table_page(data, title); grDevices::dev.off()
}

for (stem in names(table_specs)) {
  spec <- table_specs[[stem]]; data <- read_source(spec$source)
  save_table(data, stem, spec$title, 183, spec$height)
  if (!file.copy(file.path(source_dir, spec$source), file.path(stage_dir, paste0(stem, ".csv")), overwrite = FALSE)) stop("table CSV copy failed")
}

# Immutable render manifest and audit notes.
rendered <- list.files(stage_dir, full.names = TRUE)
manifest <- data.frame(
  file = basename(rendered),
  sha256 = vapply(rendered, sha256_file, character(1)),
  bytes = as.numeric(file.info(rendered)$size),
  backend = "R 4.5.3",
  contract_sha256 = expected_contract_sha,
  source_manifest_sha256 = expected_source_manifest_sha,
  stringsAsFactors = FALSE
)
write.csv(manifest, file.path(stage_dir, "render_manifest.csv"), row.names = FALSE, quote = TRUE, na = "NA")
audit_lines <- c(
  "schema_version=stage07_render_audit_rc_v1_1",
  paste0("timestamp=", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
  "status=PASS",
  "backend=R_only",
  paste0("contract_sha256=", expected_contract_sha),
  paste0("source_manifest_sha256=", expected_source_manifest_sha),
  "figure_count=7",
  "tables=4",
  "figure_formats=svg,pdf,tiff_600dpi",
  "table_formats=svg,pdf,tiff_600dpi,csv",
  "data_raw_access=none",
  "privacy=No suppressed <5 value was imputed; known-flow summaries explicitly disclose omitted suppressed strata.",
  "Figure_4=Maps, regional coverage, and vulnerability gaps remain NOT_EVALUATED because the frozen contract supplies no geometry, region mapping, or municipality-level IVS join.",
  "Figure_5=No spatial network layout was fabricated because frozen node coordinates are absent; observed hub concentration and centrality distributions are shown.",
  "Figure_6=Bootstrap gate DOWNGRADE; no accepted confidence intervals; associational point estimates only.",
  "Figure_7=Random envelope is not a confidence interval; resilience is a structural stress test, not a real-world counterfactual."
)
writeLines(audit_lines, file.path(stage_dir, "render_audit.txt"), useBytes = TRUE)

for (path in list.files(stage_dir, full.names = TRUE)) Sys.chmod(path, mode = "0444")
if (!file.rename(stage_dir, output_dir)) stop("transactional output commit failed")
committed <- TRUE
cat(sprintf("RENDER_PASS figure_count=7 table_count=4 output=%s\n", output_dir))
