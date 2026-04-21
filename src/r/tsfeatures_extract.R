#!/usr/bin/env Rscript
# Default tsfeatures::tsfeatures() only — one matrix per channel (rows = windows, cols = 128).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: Rscript tsfeatures_extract.R <input_dir> <output_csv> <channels_csv>")
}

input_dir <- args[[1L]]
output_csv <- args[[2L]]
channels <- strsplit(args[[3L]], ",", fixed = TRUE)[[1L]]

suppressPackageStartupMessages({
  if (!requireNamespace("tsfeatures", quietly = TRUE)) {
    stop("R package 'tsfeatures' is required. Install with: install.packages('tsfeatures')")
  }
  library(tsfeatures)
})

all_df <- NULL

for (ch in channels) {
  path <- file.path(input_dir, paste0(ch, ".csv"))
  if (!file.exists(path)) {
    stop("Missing input file: ", path)
  }
  M <- as.matrix(utils::read.csv(path, header = FALSE))
  storage.mode(M) <- "double"
  if (ncol(M) != 128L) {
    stop("Expected 128 samples per window for channel ", ch, "; got ", ncol(M))
  }

  feat <- tsfeatures::tsfeatures(M)
  feat <- as.data.frame(feat)
  colnames(feat) <- paste(ch, colnames(feat), sep = "__")

  all_df <- if (is.null(all_df)) feat else cbind(all_df, feat)
}

utils::write.csv(all_df, output_csv, row.names = FALSE)
