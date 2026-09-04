#!/usr/bin/env Rscript
# The sole v3 point-estimation gate for the frozen Aim 4 affine reparameterisation.
# It never starts a bootstrap.  All output is associational/supportive only.

options(warn = 1)
for (v in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")) do.call(Sys.setenv, setNames(list("1"), v))

fail <- function(message) { cat(paste0("ERROR: ", message, "\n"), file = stderr()); quit(status = 2) }
if (!requireNamespace("brglm2", quietly = TRUE)) fail("Package brglm2 is required")
if (!requireNamespace("detectseparation", quietly = TRUE)) fail("Package detectseparation is required")
if (!requireNamespace("jsonlite", quietly = TRUE)) fail("Package jsonlite is required")

sha256 <- function(path) {
  output <- suppressWarnings(system2("sha256sum", args = shQuote(path), stdout = TRUE, stderr = FALSE))
  if (length(output) != 1L || !grepl("^[0-9a-fA-F]{64}\\s+", output)) fail(paste("Unable to SHA-256", path))
  sub("\\s+.*$", "", output)
}
atomic_json <- function(path, object) {
  if (file.exists(path)) fail(paste("Refusing to overwrite immutable point artifact", path))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temp <- tempfile(pattern = paste0(".", basename(path), "."), tmpdir = dirname(path))
  jsonlite::write_json(object, temp, auto_unbox = TRUE, na = "null", pretty = TRUE, digits = 16)
  if (!file.rename(temp, path)) { unlink(temp); fail(paste("Atomic rename failed", path)) }
}
parse_args <- function(args) {
  out <- list(point_only = FALSE)
  index <- 1L
  while (index <= length(args)) {
    token <- args[[index]]
    if (identical(token, "--point-only")) { out$point_only <- TRUE; index <- index + 1L; next }
    if (identical(token, "--bootstrap-only")) fail("v3 point fitter never starts bootstrap; an external authorized bootstrap runner is required")
    if (!startsWith(token, "--") || index == length(args)) fail("Invalid argument list")
    out[[gsub("-", "_", substring(token, 3))]] <- args[[index + 1L]]
    index <- index + 2L
  }
  for (name in c("input_gz", "design_json", "prefit_manifest", "output_dir")) if (is.null(out[[name]])) fail(paste0("Missing --", gsub("_", "-", name)))
  if (!isTRUE(out$point_only)) fail("v3 requires explicit --point-only; bootstrap is not started by this script")
  out
}
read_inputs <- function(a) {
  configuration <- jsonlite::fromJSON(a$design_json, simplifyVector = FALSE)
  prefit <- jsonlite::fromJSON(a$prefit_manifest, simplifyVector = FALSE)
  if (!identical(configuration$schema_version, "aim4_brglm2_v3") || !identical(prefit$schema_version, "aim4_brglm2_v3")) fail("Input/design/prefit manifest must be aim4_brglm2_v3")
  required_controls <- c(maxit = 500, epsilon = 1e-8, slowit = 0.5, max_step_factor = 6)
  supplied_controls <- unlist(configuration$optimization_controls, use.names = TRUE)
  if (!identical(sort(names(supplied_controls)), sort(names(required_controls))) || any(!is.finite(as.numeric(supplied_controls))) || any(as.numeric(supplied_controls[names(required_controls)]) != required_controls)) fail("v3 optimization controls do not match frozen amendment")
  columns <- unlist(configuration$design_columns, use.names = FALSE)
  if (length(columns) != 96L || !identical(columns, sprintf("x_%04d", 0:95))) fail("Scaled design columns must be x_0000..x_0095")
  raw <- read.csv(gzfile(a$input_gz), check.names = FALSE, stringsAsFactors = FALSE,
                  colClasses = c(analysis_row_id = "character", y = "integer", cnes7 = "character", state_provider = "character"))
  if (!identical(names(raw), c("analysis_row_id", "y", "cnes7", "state_provider", columns))) fail("Scaled input does not conform to v3 schema/order")
  X <- as.matrix(raw[, columns, drop = FALSE]); storage.mode(X) <- "double"; y <- as.numeric(raw$y)
  if (!all(y %in% c(0, 1)) || any(!is.finite(X)) || any(raw$cnes7 == "") || any(raw$state_provider == "")) fail("Invalid outcome, identifiers, or scaled design matrix")
  if (!all(X[, 1] == 1) || qr(X)$rank != ncol(X)) fail("Scaled intercept/rank contract failed")
  p10 <- as.numeric(unlist(configuration$volume_basis_p10_scaled, use.names = FALSE)); p90 <- as.numeric(unlist(configuration$volume_basis_p90_scaled, use.names = FALSE))
  positions <- as.integer(unlist(configuration$volume_column_indices_zero_based, use.names = FALSE)) + 1L
  if (length(positions) != 3L || length(p10) != 3L || length(p90) != 3L || any(!is.finite(c(p10, p90)))) fail("Scaled contrast contract failed")
  expected_input_hash <- prefit$outputs[[basename(a$input_gz)]]
  expected_design_hash <- prefit$outputs[[basename(a$design_json)]]
  if (!identical(sha256(a$input_gz), expected_input_hash) || !identical(sha256(a$design_json), expected_design_hash)) fail("Scaled inputs do not match prefit manifest SHA-256")
  list(config = configuration, raw = raw, X = X, y = y, p10 = p10, p90 = p90, positions = positions,
       input_sha256 = sha256(a$input_gz), design_sha256 = sha256(a$design_json), prefit_manifest_sha256 = sha256(a$prefit_manifest))
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
fit_br <- function(X, y, type) {
  control <- brglm2::brglmControl(type = type, maxit = 500, epsilon = 1e-8, slowit = 0.5, max_step_factor = 6)
  brglm2::brglmFit(x = X, y = y, weights = rep(1, length(y)), family = binomial(link = "logit"), control = control, intercept = TRUE)
}
capture_fit <- function(inputs, type) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers({
    model <- fit_br(inputs$X, inputs$y, type)
    if (length(warnings) > 0L) stop("warnings_recorded")
    if (!isTRUE(model$converged)) stop("nonconvergence")
    beta <- as.numeric(model$coefficients)
    if (length(beta) != ncol(inputs$X) || any(!is.finite(beta))) stop("nonfinite_estimate")
    values <- contrast(beta, inputs)
    c(list(status = "valid", failure_reason = NA_character_, warnings = warnings, beta = beta), values)
  }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)) }), error = function(e) {
    known <- c("warnings_recorded", "nonconvergence", "nonfinite_estimate", "contrast_not_estimable")
    message <- conditionMessage(e)
    list(status = "failed", failure_reason = if (message %in% known) message else "runtime_error", warnings = warnings, error_message = message)
  })
  result
}
run_point <- function(inputs, out_dir) {
  separation <- tryCatch({
    detected <- detectseparation::detect_separation(x = inputs$X, y = inputs$y, weights = rep(1, length(inputs$y)), family = binomial(link = "logit"))
    list(status = "PASS", complete_or_quasi_complete = any(is.infinite(detected$coefficients)), coefficients = as.list(detected$coefficients))
  }, error = function(e) list(status = "ERROR", error_message = conditionMessage(e)))
  primary <- capture_fit(inputs, "AS_mean")
  sensitivity <- if (identical(primary$status, "valid")) capture_fit(inputs, "MPL_Jeffreys") else list(status = "not_run", reason = "AS_mean_failed")
  bootstrap_eligible <- identical(primary$status, "valid") && identical(sensitivity$status, "valid") && identical(separation$status, "PASS")
  output <- list(
    schema_version = "aim4_brglm2_v3", evidence = "associational/supportive", formal_bootstrap_started = FALSE,
    estimator_primary = "brglm2 AS_mean (mean bias reduction)", estimator_sensitivity = "brglm2 MPL_Jeffreys (Jeffreys-prior/Firth-style sensitivity)",
    optimization_controls = list(maxit = 500, epsilon = 1e-8, slowit = 0.5, max_step_factor = 6), no_wald_firth_p_values = TRUE,
    primary = primary, sensitivity = sensitivity, bootstrap_eligibility = bootstrap_eligible,
    input_sha256 = inputs$input_sha256, design_sha256 = inputs$design_sha256, prefit_manifest_sha256 = inputs$prefit_manifest_sha256,
    detectseparation_audit = separation, r_version = R.version.string, brglm2_version = as.character(utils::packageVersion("brglm2"))
  )
  output$primary$beta <- NULL; output$sensitivity$beta <- NULL
  atomic_json(file.path(out_dir, "aim4_brglm2_point_estimate_v3.json"), output)
}
main <- function() { args <- parse_args(commandArgs(trailingOnly = TRUE)); dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE); run_point(read_inputs(args), args$output_dir); invisible(0L) }
main()
