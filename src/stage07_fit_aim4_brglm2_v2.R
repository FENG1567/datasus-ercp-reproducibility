#!/usr/bin/env Rscript
# Frozen Aim 4 separation fallback.  All quantities describe associational/supportive evidence.

options(warn = 1)
for (v in c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")) do.call(Sys.setenv, setNames(list("1"), v))

fail <- function(message) { cat(paste0("ERROR: ", message, "\n"), file = stderr()); quit(status = 2) }
if (!requireNamespace("brglm2", quietly = TRUE)) fail("Package brglm2 is required")
if (!requireNamespace("detectseparation", quietly = TRUE)) fail("Package detectseparation is required")
if (!requireNamespace("jsonlite", quietly = TRUE)) fail("Package jsonlite is required")

parse_args <- function(args) {
  out <- list(point_only = FALSE, bootstrap_only = FALSE, overwrite = FALSE)
  index <- 1L
  flags <- c("--point-only", "--bootstrap-only", "--overwrite")
  while (index <= length(args)) {
    token <- args[[index]]
    if (token %in% flags) {
      key <- gsub("-", "_", substring(token, 3))
      out[[key]] <- TRUE
      index <- index + 1L
      next
    }
    if (!startsWith(token, "--") || index == length(args)) fail("Invalid argument list")
    out[[gsub("-", "_", substring(token, 3))]] <- args[[index + 1L]]
    index <- index + 2L
  }
  for (name in c("input_gz", "design_json", "output_dir")) if (is.null(out[[name]])) fail(paste0("Missing --", gsub("_", "-", name)))
  if (isTRUE(out$point_only) && isTRUE(out$bootstrap_only)) fail("--point-only and --bootstrap-only cannot be combined")
  out$replicate_start <- as.integer(if (is.null(out$replicate_start)) 1L else out$replicate_start)
  out$replicate_end <- as.integer(if (is.null(out$replicate_end)) 2000L else out$replicate_end)
  if (!isTRUE(out$point_only) && (is.na(out$replicate_start) || is.na(out$replicate_end) || out$replicate_start < 1L || out$replicate_end > 2000L || out$replicate_start > out$replicate_end)) fail("Replicate range must be within 1..2000")
  out
}

sha256 <- function(path) {
  # sha256sum is part of the specified Linux server baseline; do not relabel MD5.
  output <- suppressWarnings(system2("sha256sum", args = shQuote(path), stdout = TRUE, stderr = FALSE))
  if (length(output) != 1L || !grepl("^[0-9a-fA-F]{64}\\s+", output)) fail(paste("Unable to SHA-256", path))
  sub("\\s+.*$", "", output)
}
atomic_json <- function(path, object, overwrite = FALSE) {
  if (file.exists(path) && !overwrite) return(invisible(FALSE))
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temp <- tempfile(pattern = paste0(".", basename(path), "."), tmpdir = dirname(path))
  jsonlite::write_json(object, temp, auto_unbox = TRUE, na = "null", pretty = TRUE, digits = 16)
  if (!file.rename(temp, path)) { unlink(temp); fail(paste("Atomic rename failed", path)) }
  invisible(TRUE)
}

read_inputs <- function(a) {
  configuration <- jsonlite::fromJSON(a$design_json, simplifyVector = FALSE)
  raw <- read.csv(gzfile(a$input_gz), check.names = FALSE, stringsAsFactors = FALSE,
                  colClasses = c(analysis_row_id = "character", y = "integer", cnes7 = "character", state_provider = "character"))
  columns <- unlist(configuration$design_columns, use.names = FALSE)
  if (!all(c("y", "cnes7", "state_provider", columns) %in% names(raw))) fail("Input does not conform to frozen design manifest")
  X <- as.matrix(raw[, columns, drop = FALSE]); storage.mode(X) <- "double"
  y <- as.numeric(raw$y)
  if (!all(y %in% c(0, 1)) || any(!is.finite(X))) fail("Invalid outcome or design matrix")
  if (qr(X)$rank != ncol(X)) fail("Original frozen design is rank deficient; fallback is not permitted")
  list(config = configuration, raw = raw, X = X, y = y, input_sha256 = sha256(a$input_gz), design_sha256 = sha256(a$design_json))
}

contrast <- function(beta, original_X, configuration) {
  positions <- as.integer(unlist(configuration$volume_column_indices_zero_based, use.names = FALSE)) + 1L
  lower <- original_X; upper <- original_X
  lower[, positions] <- matrix(as.numeric(unlist(configuration$volume_basis_p10, use.names = FALSE)), nrow = nrow(lower), ncol = length(positions), byrow = TRUE)
  upper[, positions] <- matrix(as.numeric(unlist(configuration$volume_basis_p90, use.names = FALSE)), nrow = nrow(upper), ncol = length(positions), byrow = TRUE)
  risk_low <- mean(plogis(drop(lower %*% beta)))
  risk_high <- mean(plogis(drop(upper %*% beta)))
  if (!all(is.finite(c(risk_low, risk_high))) || risk_low < 0 || risk_high < 0 || risk_low > 1 || risk_high > 1) stop("invalid_prediction")
  if (risk_low <= 0) stop("contrast_not_estimable")
  list(risk_p10 = risk_low, risk_p90 = risk_high, rd = risk_high - risk_low, rr = risk_high / risk_low)
}

fit_br <- function(X, y, weights, type) {
  control <- brglm2::brglmControl(type = type, maxit = 120, epsilon = 1e-8)
  brglm2::brglmFit(x = X, y = y, weights = weights, family = binomial(link = "logit"), control = control, intercept = TRUE)
}

capture_fit <- function(X, y, weights, original_X, configuration, type) {
  warnings <- character()
  result <- tryCatch(withCallingHandlers({
    if (qr(X)$rank != ncol(X)) stop("rank_deficient")
    model <- fit_br(X, y, weights, type)
    if (!isTRUE(model$converged)) stop("nonconvergence")
    beta <- as.numeric(model$coefficients)
    if (length(beta) != ncol(X) || any(!is.finite(beta))) stop("nonfinite_estimate")
    values <- contrast(beta, original_X, configuration)
    c(list(status = "valid", failure_reason = NA_character_, warnings = warnings, beta = beta), values)
  }, warning = function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
  error = function(e) {
    known <- c("rank_deficient", "nonconvergence", "nonfinite_estimate", "invalid_prediction", "contrast_not_estimable")
    message <- conditionMessage(e)
    reason <- if (message %in% known) message else "runtime_error"
    list(status = "failed", failure_reason = reason, warnings = warnings, error_message = message)
  })
  result
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

run_point <- function(inputs, out_dir, overwrite) {
  point_file <- file.path(out_dir, "aim4_brglm2_point_estimate_v2.json")
  if (file.exists(point_file) && !overwrite) {
    existing <- tryCatch(jsonlite::read_json(point_file, simplifyVector = TRUE), error = function(e) NULL)
    if (is.null(existing) || !identical(existing$input_sha256, inputs$input_sha256) || !identical(existing$design_sha256, inputs$design_sha256) || !identical(existing$schema_version, "aim4_brglm2_v2")) fail("Existing point artifact is not a compatible resume artifact")
    return(invisible("existing"))
  }
  separation <- tryCatch({
    detected <- detectseparation::detect_separation(x = inputs$X, y = inputs$y, weights = rep(1, length(inputs$y)), family = binomial(link = "logit"))
    list(status = "PASS", complete_or_quasi_complete = any(is.infinite(detected$coefficients)), coefficients = as.list(detected$coefficients))
  }, error = function(e) list(status = "ERROR", error_message = conditionMessage(e)))
  primary <- capture_fit(inputs$X, inputs$y, rep(1, length(inputs$y)), inputs$X, inputs$config, "AS_mean")
  sensitivity <- if (identical(primary$status, "valid")) capture_fit(inputs$X, inputs$y, rep(1, length(inputs$y)), inputs$X, inputs$config, "MPL_Jeffreys") else list(status = "not_run", reason = "AS_mean_failed")
  output <- list(
    schema_version = "aim4_brglm2_v2", evidence = "associational/supportive", estimator_primary = "brglm2 AS_mean (mean bias reduction)",
    estimator_sensitivity = "brglm2 MPL_Jeffreys (Jeffreys-prior/Firth-style sensitivity)", no_wald_firth_p_values = TRUE,
    primary = primary, sensitivity = sensitivity,
    input_sha256 = inputs$input_sha256, design_sha256 = inputs$design_sha256,
    detectseparation_audit = separation,
    r_version = R.version.string, brglm2_version = as.character(utils::packageVersion("brglm2"))
  )
  output$primary$beta <- NULL; output$sensitivity$beta <- NULL
  atomic_json(point_file, output, overwrite)
}

seed_for_replicate <- function(master, replicate_id) {
  current <- master
  if (replicate_id > 1L) for (unused in seq_len(replicate_id - 1L)) current <- parallel::nextRNGStream(current)
  current
}

sample_weights <- function(raw, replicate_id, master_seed) {
  assign(".Random.seed", seed_for_replicate(master_seed, replicate_id), envir = .GlobalEnv)
  weights <- numeric(nrow(raw)); state_summary <- character()
  for (state in sort(unique(raw$state_provider))) {
    hospitals <- sort(unique(raw$cnes7[raw$state_provider == state]))
    drawn <- sample(hospitals, size = length(hospitals), replace = TRUE)
    multiplicity <- table(drawn)
    for (hospital in names(multiplicity)) weights[raw$state_provider == state & raw$cnes7 == hospital] <- as.numeric(multiplicity[[hospital]])
    state_summary <- c(state_summary, paste0(state, ":", length(hospitals)))
  }
  list(weights = weights, state_summary = paste(state_summary, collapse = ";"))
}

zero_columns <- function(X, names) {
  names[which(apply(X, 2, function(column) all(column == 0)))]
}

run_replicate <- function(inputs, replicate_id, master_seed, out_dir, overwrite) {
  file <- file.path(out_dir, "replicates", sprintf("replicate_%04d.json", replicate_id))
  if (file.exists(file) && !overwrite) {
    existing <- tryCatch(jsonlite::read_json(file, simplifyVector = TRUE), error = function(e) NULL)
    if (is.null(existing) || !identical(as.integer(existing$replicate_id), as.integer(replicate_id)) || !identical(existing$input_sha256, inputs$input_sha256) || !identical(existing$design_sha256, inputs$design_sha256)) fail(paste("Existing shard is not a compatible resume artifact", file))
    return(invisible("existing"))
  }
  started <- proc.time()[["elapsed"]]
  drawn <- sample_weights(inputs$raw, replicate_id, master_seed)
  positive <- drawn$weights > 0
  X <- inputs$X[positive, , drop = FALSE]; y <- inputs$y[positive]; weights <- drawn$weights[positive]
  as_mean <- capture_fit(X, y, weights, inputs$X, inputs$config, "AS_mean")
  mpl <- capture_fit(X, y, weights, inputs$X, inputs$config, "MPL_Jeffreys")
  semantic_names <- unlist(inputs$config$design_names, use.names = FALSE)
  semantic_zero <- paste(zero_columns(X, semantic_names), collapse = ";")
  output <- list(
    schema_version = "aim4_brglm2_v2", evidence = "associational/supportive", replicate_id = replicate_id,
    status = as_mean$status, failure_reason = as_mean$failure_reason, runtime_seconds = proc.time()[["elapsed"]] - started,
    state_hospital_counts = drawn$state_summary, zero_design_columns = semantic_zero,
    calendar_month_zero_columns = paste(grep("calendar_month", semantic_zero, value = TRUE), collapse = ";"),
    hospital_type_zero_columns = paste(grep("hospital_type", semantic_zero, value = TRUE), collapse = ";"),
    input_sha256 = inputs$input_sha256, design_sha256 = inputs$design_sha256,
    as_mean_status = as_mean$status, as_mean_failure_reason = as_mean$failure_reason,
    mpl_jeffreys_status = mpl$status, mpl_jeffreys_failure_reason = mpl$failure_reason
  )
  for (metric in c("risk_p10", "risk_p90", "rd", "rr")) {
    output[[paste0("as_mean_", metric)]] <- as_mean[[metric]] %||% NA_real_
    output[[paste0("mpl_jeffreys_", metric)]] <- mpl[[metric]] %||% NA_real_
  }
  atomic_json(file, output, overwrite)
  invisible(output$status)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  inputs <- read_inputs(args)
  RNGkind("L'Ecuyer-CMRG"); set.seed(20260830); master_seed <- .Random.seed
  if (!isTRUE(args$bootstrap_only)) run_point(inputs, args$output_dir, args$overwrite)
  if (!isTRUE(args$point_only)) for (replicate_id in seq.int(args$replicate_start, args$replicate_end)) run_replicate(inputs, replicate_id, master_seed, args$output_dir, args$overwrite)
  invisible(0L)
}

main()
