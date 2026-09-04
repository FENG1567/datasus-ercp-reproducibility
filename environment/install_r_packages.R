#!/usr/bin/env Rscript

# Compatibility installer. Exact historical package versions were not captured.
packages <- c(
  "brglm2", "detectseparation", "jsonlite", "ggplot2", "patchwork",
  "svglite", "ragg"
)
missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
cat("R version:\n", R.version.string, "\n", sep = "")
cat("Installed package versions:\n")
for (pkg in packages) cat(pkg, as.character(packageVersion(pkg)), "\n")
