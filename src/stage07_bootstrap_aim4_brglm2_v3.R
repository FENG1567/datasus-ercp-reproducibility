#!/usr/bin/env Rscript
# External, immutable v3 bootstrap only. It never fits the formal point model.

options(warn = 1)
for (v in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")) do.call(Sys.setenv, setNames(list("1"), v))
fail <- function(message) { cat(paste0("ERROR: ", message, "\n"), file = stderr()); quit(status = 2) }
if (!requireNamespace("brglm2", quietly = TRUE)) fail("Package brglm2 is required")
if (!requireNamespace("jsonlite", quietly = TRUE)) fail("Package jsonlite is required")

N_REPLICATES <- 2000L
MASTER_SEED <- 20260830L
CONTROLS <- c(maxit = 500, epsilon = 1e-8, slowit = 0.5, max_step_factor = 6)
SCHEMA <- "aim4_brglm2_v3"

sha256 <- function(path) {
  answer <- suppressWarnings(system2("sha256sum", args = shQuote(path), stdout = TRUE, stderr = FALSE))
  if (length(answer) != 1L || !grepl("^[0-9a-fA-F]{64}\\s+", answer)) fail(paste("Unable to SHA-256", path))
  sub("\\s+.*$", "", answer)
}
atomic_json <- function(path, object) {
  if (file.exists(path)) fail(paste("Refusing to overwrite immutable replicate", path))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temp <- tempfile(pattern = paste0(".", basename(path), "."), tmpdir = dirname(path))
  jsonlite::write_json(object, temp, auto_unbox = TRUE, na = "null", pretty = TRUE, digits = 16)
  if (!file.rename(temp, path)) { unlink(temp); fail(paste("Atomic rename failed", path)) }
}
parse_args <- function(args) {
  out <- list(bootstrap_only = FALSE)
  i <- 1L
  while (i <= length(args)) {
    token <- args[[i]]
    if (identical(token, "--bootstrap-only")) { out$bootstrap_only <- TRUE; i <- i + 1L; next }
  if (!startsWith(token, "--") || i == length(args)) fail("Invalid argument list")
    out[[gsub("-", "_", substring(token, 3))]] <- args[[i + 1L]]; i <- i + 2L
  }
  if (!isTRUE(out$bootstrap_only)) fail("Only explicit --bootstrap-only is permitted")
  for (name in c("input_gz", "design_json", "prefit_manifest", "point_json", "run_manifest", "output_dir")) if (is.null(out[[name]])) fail(paste0("Missing --", gsub("_", "-", name)))
  out$replicate_start <- as.integer(if (is.null(out$replicate_start)) 1L else out$replicate_start)
  out$replicate_end <- as.integer(if (is.null(out$replicate_end)) N_REPLICATES else out$replicate_end)
  if (is.na(out$replicate_start) || is.na(out$replicate_end) || out$replicate_start < 1L || out$replicate_end > N_REPLICATES || out$replicate_start > out$replicate_end) fail("Replicate range must be within 1..2000")
  out
}
require_controls <- function(value) {
  supplied <- unlist(value, use.names = TRUE)
  if (!identical(sort(names(supplied)), sort(names(CONTROLS))) || any(!is.finite(as.numeric(supplied))) || any(as.numeric(supplied[names(CONTROLS)]) != CONTROLS)) fail("Optimization controls do not match frozen v3 amendment")
}
read_inputs <- function(a) {
  cfg <- jsonlite::fromJSON(a$design_json, simplifyVector = FALSE)
  prefit <- jsonlite::fromJSON(a$prefit_manifest, simplifyVector = FALSE)
  point <- jsonlite::fromJSON(a$point_json, simplifyVector = FALSE)
  run_manifest <- jsonlite::fromJSON(a$run_manifest, simplifyVector = FALSE)
  if (!identical(cfg$schema_version, SCHEMA) || !identical(prefit$schema_version, SCHEMA) || !identical(point$schema_version, SCHEMA)) fail("Input/design/prefit/point schema must be aim4_brglm2_v3")
  require_controls(cfg$optimization_controls)
  if (!isTRUE(point$bootstrap_eligibility) || !identical(point$formal_bootstrap_started, FALSE) || !identical(point$primary$status, "valid") || !identical(point$sensitivity$status, "valid") || !identical(point$detectseparation_audit$status, "PASS")) fail("Frozen point gate does not permit bootstrap")
  input_hash <- sha256(a$input_gz); design_hash <- sha256(a$design_json); prefit_hash <- sha256(a$prefit_manifest); point_hash <- sha256(a$point_json)
  if (!identical(point$input_sha256, input_hash) || !identical(point$design_sha256, design_hash) || !identical(point$prefit_manifest_sha256, prefit_hash)) fail("Point provenance does not bind supplied immutable v3 prefit inputs")
  if (!identical(prefit$outputs[[basename(a$input_gz)]], input_hash) || !identical(prefit$outputs[[basename(a$design_json)]], design_hash)) fail("Prefit manifest does not bind supplied scaled input/design")
  run_manifest_hash <- sha256(a$run_manifest)
  if (!identical(run_manifest$schema_version, SCHEMA) || !identical(run_manifest$evidence, "associational/supportive")) fail("Bootstrap run manifest is not the frozen v3 contract")
  if (!identical(run_manifest$inputs[[basename(a$input_gz)]], input_hash) || !identical(run_manifest$inputs[[basename(a$design_json)]], design_hash) || !identical(run_manifest$inputs[[basename(a$prefit_manifest)]], prefit_hash) || !identical(run_manifest$inputs[[basename(a$point_json)]], point_hash)) fail("Bootstrap run manifest does not bind frozen inputs")
  full_args <- commandArgs(trailingOnly = FALSE); file_arg <- full_args[grep("^--file=.*stage07_bootstrap_aim4_brglm2_v3.R$", full_args)]
  if (length(file_arg) != 1L) fail("Cannot resolve executing bootstrap R script for provenance")
  script_path <- sub("^--file=", "", file_arg)
  if (!identical(run_manifest$code_sha256[[basename(script_path)]], sha256(script_path))) fail("Bootstrap run manifest does not bind this R code")
  columns <- unlist(cfg$design_columns, use.names = FALSE)
  if (length(columns) != 96L || !identical(columns, sprintf("x_%04d", 0:95))) fail("Scaled v3 design must be explicit x_0000..x_0095")
  raw <- read.csv(gzfile(a$input_gz), check.names = FALSE, stringsAsFactors = FALSE, colClasses = c(analysis_row_id = "character", y = "integer", cnes7 = "character", state_provider = "character"))
  if (!identical(names(raw), c("analysis_row_id", "y", "cnes7", "state_provider", columns))) fail("Scaled input schema/order does not match v3 design")
  X <- as.matrix(raw[, columns, drop = FALSE]); storage.mode(X) <- "double"; y <- as.numeric(raw$y)
  if (nrow(X) != 30900L || !all(y %in% c(0, 1)) || any(!is.finite(X)) || !all(X[, 1] == 1) || any(raw$cnes7 == "") || any(raw$state_provider == "")) fail("Invalid frozen scaled input")
  if (qr(X)$rank != ncol(X)) fail("Frozen scaled design is rank deficient")
  p10 <- as.numeric(unlist(cfg$volume_basis_p10_scaled, use.names = FALSE)); p90 <- as.numeric(unlist(cfg$volume_basis_p90_scaled, use.names = FALSE)); positions <- as.integer(unlist(cfg$volume_column_indices_zero_based, use.names = FALSE)) + 1L
  if (length(positions) != 3L || length(p10) != 3L || length(p90) != 3L || any(!is.finite(c(p10, p90)))) fail("Frozen scaled contrast is invalid")
  list(cfg = cfg, raw = raw, X = X, y = y, p10 = p10, p90 = p90, positions = positions,
       hashes = list(input_sha256 = input_hash, design_sha256 = design_hash, prefit_manifest_sha256 = prefit_hash, point_sha256 = point_hash, run_manifest_sha256 = run_manifest_hash))
}
seed_for_replicate <- function(master_seed, replicate_id) {
  current <- master_seed
  if (replicate_id > 1L) for (unused in seq_len(replicate_id - 1L)) current <- parallel::nextRNGStream(current)
  current
}
sample_weights <- function(raw, replicate_id, master_seed) {
  assign(".Random.seed", seed_for_replicate(master_seed, replicate_id), envir = .GlobalEnv)
  weights <- numeric(nrow(raw)); state_summary <- character()
  for (state in sort(unique(raw$state_provider))) {
    hospital_ids <- sort(unique(raw$cnes7[raw$state_provider == state]))
    drawn <- sample(hospital_ids, size = length(hospital_ids), replace = TRUE)
    multiplicity <- table(drawn)
    for (hospital in names(multiplicity)) weights[raw$state_provider == state & raw$cnes7 == hospital] <- as.numeric(multiplicity[[hospital]])
    state_summary <- c(state_summary, paste0(state, ":", length(hospital_ids)))
  }
  list(weights = weights, state_hospital_counts = paste(state_summary, collapse = ";"))
}
contrast <- function(beta, inputs) {
  lower <- inputs$X; upper <- inputs$X
  lower[, inputs$positions] <- matrix(inputs$p10, nrow = nrow(lower), ncol = length(inputs$positions), byrow = TRUE)
  upper[, inputs$positions] <- matrix(inputs$p90, nrow = nrow(upper), ncol = length(inputs$positions), byrow = TRUE)
  eta_low <- drop(lower %*% beta); eta_high <- drop(upper %*% beta)
  risk_low <- mean(plogis(eta_low)); risk_high <- mean(plogis(eta_high))
  if (!all(is.finite(c(eta_low, eta_high, risk_low, risk_high))) || risk_low <= 0 || risk_low > 1 || risk_high < 0 || risk_high > 1) stop("contrast_not_estimable")
  list(risk_p10 = risk_low, risk_p90 = risk_high, rd = risk_high - risk_low, rr = risk_high / risk_low)
}
fit_one <- function(X, y, weights, type, inputs) {
  warnings <- character()
  tryCatch(withCallingHandlers({
    if (qr(X)$rank != ncol(X)) stop("rank_deficient")
    control <- brglm2::brglmControl(type = type, maxit = 500, epsilon = 1e-8, slowit = 0.5, max_step_factor = 6)
    model <- brglm2::brglmFit(x = X, y = y, weights = weights, family = binomial(link = "logit"), control = control, intercept = TRUE)
    if (!isTRUE(model$converged)) stop("nonconvergence")
    beta <- as.numeric(model$coefficients)
    if (length(beta) != ncol(X) || any(!is.finite(beta))) stop("nonfinite_estimate")
    values <- contrast(beta, inputs)
    c(list(status = "valid", failure_reason = NA_character_, warnings = warnings), values)
  }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }), error = function(e) {
    message <- conditionMessage(e); known <- c("rank_deficient", "nonconvergence", "nonfinite_estimate", "invalid_prediction", "contrast_not_estimable")
    list(status = "failed", failure_reason = if (message %in% known) message else "runtime_error", warnings = warnings, error_message = message)
  })
}
constant_semantic_columns <- function(X, cfg, tolerance = 1e-12) {
  names <- unlist(cfg$design_names, use.names = FALSE)
  # Centered/scaled absent dummies are constant at -mean/sd rather than zero.
  non_intercept <- seq.int(2L, ncol(X))
  names[non_intercept][which(vapply(non_intercept, function(index) {
    value <- X[, index]
    all(is.finite(value)) && (max(value) - min(value) <= tolerance)
  }, logical(1)))]
}
existing_replicate_matches <- function(path, replicate_id, hashes) {
  if (!file.exists(path)) return(FALSE)
  existing <- tryCatch(jsonlite::read_json(path, simplifyVector = TRUE), error = function(e) NULL)
  if (is.null(existing) || !identical(as.integer(existing$replicate_id), as.integer(replicate_id)) || !identical(existing$schema_version, SCHEMA) || !identical(existing$evidence, "associational/supportive")) fail(paste("Existing replicate is malformed or incompatible", path))
  for (field in names(hashes)) if (!identical(existing[[field]], hashes[[field]])) fail(paste("Existing replicate provenance mismatch", field, path))
  TRUE
}
run_replicate <- function(inputs, replicate_id, master_seed, output_dir) {
  path <- file.path(output_dir, "replicates", sprintf("replicate_%04d.json", replicate_id))
  if (existing_replicate_matches(path, replicate_id, inputs$hashes)) return(invisible("existing_verified"))
  started <- proc.time()[["elapsed"]]
  sampled <- sample_weights(inputs$raw, replicate_id, master_seed)
  positive <- sampled$weights > 0
  X <- inputs$X[positive, , drop = FALSE]; y <- inputs$y[positive]; weights <- sampled$weights[positive]
  as_mean <- fit_one(X, y, weights, "AS_mean", inputs)
  mpl <- fit_one(X, y, weights, "MPL_Jeffreys", inputs)
  constant_vector <- constant_semantic_columns(X, inputs$cfg); constants <- paste(constant_vector, collapse = ";")
  states <- sort(unique(inputs$raw$state_provider))
  state_has_positive_support <- vapply(states, function(state) any(positive[inputs$raw$state_provider == state]), logical(1))
  provider_uf_zero_support <- paste(states[!state_has_positive_support], collapse = ";")
  record <- c(list(schema_version = SCHEMA, evidence = "associational/supportive", replicate_id = replicate_id,
                   status = as_mean$status, failure_reason = as_mean$failure_reason, runtime_seconds = proc.time()[["elapsed"]] - started,
                   state_hospital_counts = sampled$state_hospital_counts, n_positive_rows = nrow(X), n_positive_hospitals = length(unique(inputs$raw$cnes7[positive])), provider_uf_zero_support = provider_uf_zero_support,
                   constant_semantic_columns = constants, calendar_month_constant_columns = paste(grep("calendar_month", constant_vector, value = TRUE), collapse = ";"), hospital_type_constant_columns = paste(grep("hospital_type", constant_vector, value = TRUE), collapse = ";"), constant_column_tolerance = 1e-12,
                   as_mean_status = as_mean$status, as_mean_failure_reason = as_mean$failure_reason, as_mean_warnings = as_mean$warnings,
                   mpl_jeffreys_status = mpl$status, mpl_jeffreys_failure_reason = mpl$failure_reason, mpl_jeffreys_warnings = mpl$warnings), inputs$hashes)
  for (metric in c("risk_p10", "risk_p90", "rd", "rr")) { record[[paste0("as_mean_", metric)]] <- as_mean[[metric]] %||% NA_real_; record[[paste0("mpl_jeffreys_", metric)]] <- mpl[[metric]] %||% NA_real_ }
  atomic_json(path, record)
}
`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a
main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE)); inputs <- read_inputs(args)
  RNGkind("L'Ecuyer-CMRG"); set.seed(MASTER_SEED); master <- .Random.seed
  for (replicate_id in seq.int(args$replicate_start, args$replicate_end)) run_replicate(inputs, replicate_id, master, args$output_dir)
}
# Test-only source mode exposes deterministic helper functions without reading
# analytic inputs or executing a replicate. Production invocation has no such
# environment variable and always enters main().
if (!identical(Sys.getenv("STAGE07_AIM4_V3_BOOTSTRAP_SOURCE_ONLY"), "1")) main()
