#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

`%||%` <- function(a, b) if (!is.null(a)) a else b

parse_args <- function(args) {
  out <- list(
    raw_dir = NULL,
    cal_dir = NULL,
    outdir = NULL,
    prefix = "figureS_external_full_4x3_roc_prc_calibratedDCA_isotonicOOF",
    bootstrap = 1000,
    seed = 42,
    cal_bins = 8,
    dca_max_threshold = 0.80,
    dca_ymin = -0.12,
    dca_ymax = 0.20,
    width = 13,
    height = 16,
    dpi = 300
  )

  if (length(args) == 0) {
    return(out)
  }
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      i <- i + 1
      next
    }
    key <- sub("^--", "", key)
    val <- if (i + 1 <= length(args)) args[[i + 1]] else NULL
    if (is.null(val) || startsWith(val, "--")) {
      i <- i + 1
      next
    }
    if (key %in% c("raw_dir", "cal_dir", "outdir", "prefix")) out[[key]] <- val
    if (key %in% c("bootstrap", "seed", "cal_bins", "dpi")) out[[key]] <- as.integer(val)
    if (key %in% c("dca_max_threshold", "dca_ymin", "dca_ymax", "width", "height")) out[[key]] <- as.numeric(val)
    i <- i + 2
  }
  out
}

trapz_auc <- function(x, y) {
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]
  if (length(x) < 2) {
    return(NA_real_)
  }
  sum(diff(x) * (head(y, -1) + tail(y, -1)) / 2)
}

roc_points <- function(y, s) {
  y <- as.integer(y)
  ord <- order(-s, seq_along(s))
  y <- y[ord]
  s <- s[ord]
  p_n <- sum(y == 1)
  n_n <- sum(y == 0)
  if (p_n == 0 || n_n == 0) {
    return(NULL)
  }
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  idx <- which(!duplicated(s, fromLast = TRUE))
  tpr <- c(0, tp[idx] / p_n, 1)
  fpr <- c(0, fp[idx] / n_n, 1)
  tibble(x = fpr, y = tpr, auc = trapz_auc(fpr, tpr))
}

pr_points <- function(y, s) {
  y <- as.integer(y)
  ord <- order(-s, seq_along(s))
  y <- y[ord]
  s <- s[ord]
  p_n <- sum(y == 1)
  if (p_n == 0) {
    return(NULL)
  }
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  idx <- which(!duplicated(s, fromLast = TRUE))
  rec <- c(0, tp[idx] / p_n, 1)
  prec <- c(1, tp[idx] / (tp[idx] + fp[idx]), tail(tp[idx] / (tp[idx] + fp[idx]), 1))
  tibble(x = rec, y = prec, auc = trapz_auc(rec, prec))
}

bootstrap_ci_curve <- function(y, s, metric = c("roc", "pr"), B = 1000, seed = 42, grid_n = 201) {
  metric <- match.arg(metric)
  set.seed(seed)
  n <- length(y)
  grid <- seq(0, 1, length.out = grid_n)
  vals <- matrix(NA_real_, nrow = B, ncol = grid_n)
  aucs <- rep(NA_real_, B)

  for (b in seq_len(B)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    cur <- if (metric == "roc") roc_points(y[idx], s[idx]) else pr_points(y[idx], s[idx])
    if (is.null(cur)) next
    aucs[b] <- cur$auc[1]
    ord <- order(cur$x)
    vals[b, ] <- approx(cur$x[ord], cur$y[ord], xout = grid, rule = 2, ties = "ordered")$y
  }

  keep <- complete.cases(vals)
  vals <- vals[keep, , drop = FALSE]
  aucs <- aucs[is.finite(aucs)]

  if (nrow(vals) == 0 || length(aucs) == 0) {
    return(list(grid = grid, lower = rep(NA_real_, grid_n), upper = rep(NA_real_, grid_n), auc_ci = c(NA_real_, NA_real_)))
  }
  list(
    grid = grid,
    lower = apply(vals, 2, quantile, probs = 0.025, na.rm = TRUE),
    upper = apply(vals, 2, quantile, probs = 0.975, na.rm = TRUE),
    auc_ci = as.numeric(quantile(aucs, probs = c(0.025, 0.975), na.rm = TRUE))
  )
}

calibration_points_equal <- function(y, p, bins = 8) {
  brks <- seq(0, 1, length.out = bins + 1)
  b <- cut(p, breaks = brks, include.lowest = TRUE, labels = FALSE)
  tibble(y = y, p = p, b = b) %>%
    filter(!is.na(b)) %>%
    group_by(b) %>%
    summarise(pred = mean(p), obs = mean(y), n = dplyr::n(), .groups = "drop") %>%
    arrange(b)
}

decision_curve <- function(y, p, thresholds) {
  y <- as.integer(y)
  n <- length(y)
  nb <- numeric(length(thresholds))
  for (i in seq_along(thresholds)) {
    t <- thresholds[i]
    pred <- as.integer(p >= t)
    tp <- sum(y == 1 & pred == 1)
    fp <- sum(y == 0 & pred == 1)
    nb[i] <- (tp / n) - (fp / n) * (t / (1 - t))
  }
  tibble(threshold = thresholds, nb = nb)
}

brier <- function(y, p) mean((y - p)^2)

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  here <- dirname(normalizePath(sys.frame(1)$ofile %||% "src/figureS_external_full_4x3_dedicated.R"))
  root <- normalizePath(file.path(here, ".."))
  if (is.null(args$raw_dir)) args$raw_dir <- file.path(root, "results", "combined_predictions_external")
  if (is.null(args$cal_dir)) args$cal_dir <- file.path(root, "results", "combined_predictions_external_isotonic_oof_calibrated")
  if (is.null(args$outdir)) args$outdir <- file.path(root, "results", "figures_external")
  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

  model_map <- tibble(
    key = c("lightgbm", "xgboost", "catboost"),
    title = c("LightGBM", "XGBoost", "CatBoost")
  )
  version_map <- tibble(
    col = c("v1_prob", "v2_prob", "v3_prob"),
    label = c("Baseline", "+ Min/Max", "+ Shapelets")
  )
  colors <- c("Baseline" = "#5AAE61", "+ Min/Max" = "#D6604D", "+ Shapelets" = "#4393C3")

  roc_plots <- list()
  pr_plots <- list()
  cal_plots <- list()
  dca_plots <- list()
  metric_rows <- list()
  rid <- 1

  thresholds <- seq(0.01, args$dca_max_threshold, length.out = 150)

  for (i in seq_len(nrow(model_map))) {
    key <- model_map$key[i]
    title <- model_map$title[i]
    raw_df <- read.csv(file.path(args$raw_dir, paste0(key, "_predictions.csv")))
    cal_df <- read.csv(file.path(args$cal_dir, paste0(key, "_predictions.csv")))
    y <- as.integer(raw_df$y_true)
    prevalence <- mean(y)

    roc_curve_rows <- list()
    roc_ribbon_rows <- list()
    roc_legend <- c()
    pr_curve_rows <- list()
    pr_ribbon_rows <- list()
    pr_legend <- c()
    cal_rows <- list()
    dca_rows <- list()

    for (j in seq_len(nrow(version_map))) {
      vcol <- version_map$col[j]
      vlab <- version_map$label[j]
      p_raw <- as.numeric(raw_df[[vcol]])
      p_cal <- as.numeric(cal_df[[vcol]])

      roc_pt <- roc_points(y, p_raw)
      roc_ci <- bootstrap_ci_curve(y, p_raw, metric = "roc", B = args$bootstrap, seed = args$seed + i * 1000 + j * 10)
      roc_curve_rows[[j]] <- roc_pt %>% mutate(version = vlab)
      roc_ribbon_rows[[j]] <- tibble(x = roc_ci$grid, ymin = roc_ci$lower, ymax = roc_ci$upper, version = vlab)
      roc_legend[vlab] <- sprintf("%s  %.3f [%.3f-%.3f]", vlab, roc_pt$auc[1], roc_ci$auc_ci[1], roc_ci$auc_ci[2])

      pr_pt <- pr_points(y, p_raw)
      pr_ci <- bootstrap_ci_curve(y, p_raw, metric = "pr", B = args$bootstrap, seed = args$seed + i * 2000 + j * 10)
      pr_curve_rows[[j]] <- pr_pt %>% mutate(version = vlab)
      pr_ribbon_rows[[j]] <- tibble(x = pr_ci$grid, ymin = pr_ci$lower, ymax = pr_ci$upper, version = vlab)
      pr_legend[vlab] <- sprintf("%s  %.3f [%.3f-%.3f]", vlab, pr_pt$auc[1], pr_ci$auc_ci[1], pr_ci$auc_ci[2])

      cal_rows[[j]] <- calibration_points_equal(y, p_cal, bins = args$cal_bins) %>% mutate(version = vlab, brier = brier(y, p_cal))
      dca_rows[[j]] <- decision_curve(y, p_cal, thresholds) %>% mutate(version = vlab)

      metric_rows[[rid]] <- tibble(
        model = title, version = vlab, prevalence = prevalence, auroc = roc_pt$auc[1], auprc = pr_pt$auc[1], brier_calibrated = brier(y, p_cal)
      )
      rid <- rid + 1
    }

    roc_df <- bind_rows(roc_curve_rows) %>% mutate(version = factor(version, levels = version_map$label))
    roc_rb <- bind_rows(roc_ribbon_rows) %>% mutate(version = factor(version, levels = version_map$label))
    pr_df <- bind_rows(pr_curve_rows) %>% mutate(version = factor(version, levels = version_map$label))
    pr_rb <- bind_rows(pr_ribbon_rows) %>% mutate(version = factor(version, levels = version_map$label))
    cdf <- bind_rows(cal_rows) %>% mutate(version = factor(version, levels = version_map$label))
    ddf <- bind_rows(dca_rows) %>% mutate(version = factor(version, levels = version_map$label))

    treat_all <- tibble(threshold = thresholds, nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds)))
    treat_none <- tibble(threshold = thresholds, nb = 0)

    roc_plots[[i]] <- ggplot() +
      geom_ribbon(data = roc_rb, aes(x = x, ymin = ymin, ymax = ymax, fill = version), alpha = 0.18, linewidth = 0) +
      geom_line(data = roc_df, aes(x = x, y = y, color = version), linewidth = 0.9) +
      geom_abline(intercept = 0, slope = 1, linetype = "dotted", color = "grey55") +
      scale_color_manual(values = colors, breaks = version_map$label, labels = roc_legend[version_map$label], drop = FALSE) +
      scale_fill_manual(values = colors, breaks = version_map$label, labels = roc_legend[version_map$label], drop = FALSE) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      labs(title = title, x = "False Positive Rate", y = "True Positive Rate", color = "AUROC [95% CI]", fill = "AUROC [95% CI]") +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12), legend.position = "inside", legend.position.inside = c(0.98, 0.02), legend.justification = c(1, 0), legend.background = element_rect(fill = grDevices::adjustcolor("white", alpha.f = 0.84), color = "grey80"), legend.title = element_text(size = 12), legend.text = element_text(size = 10.0), panel.grid.major = element_line(linewidth = 0.25, color = "grey90"), panel.grid.minor = element_blank())

    pr_plots[[i]] <- ggplot() +
      geom_ribbon(data = pr_rb, aes(x = x, ymin = ymin, ymax = ymax, fill = version), alpha = 0.18, linewidth = 0) +
      geom_line(data = pr_df, aes(x = x, y = y, color = version), linewidth = 0.9) +
      scale_color_manual(values = colors, breaks = version_map$label, labels = pr_legend[version_map$label], drop = FALSE) +
      scale_fill_manual(values = colors, breaks = version_map$label, labels = pr_legend[version_map$label], drop = FALSE) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      labs(title = title, x = "Recall", y = "Precision", color = "AUPRC [95% CI]", fill = "AUPRC [95% CI]") +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12), legend.position = "inside", legend.position.inside = c(0.98, 0.98), legend.justification = c(1, 1), legend.background = element_rect(fill = grDevices::adjustcolor("white", alpha.f = 0.84), color = "grey80"), legend.title = element_text(size = 12), legend.text = element_text(size = 10.0), panel.grid.major = element_line(linewidth = 0.25, color = "grey90"), panel.grid.minor = element_blank())

    cal_legend <- cdf %>%
      group_by(version) %>%
      summarise(brier = first(brier), .groups = "drop") %>%
      arrange(factor(version, levels = version_map$label))
    cal_lbl <- setNames(sprintf("%s  Brier=%.3f", cal_legend$version, cal_legend$brier), cal_legend$version)
    cal_plots[[i]] <- ggplot(cdf, aes(x = pred, y = obs, color = version)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 1.8) +
      geom_abline(intercept = 0, slope = 1, linetype = "dotted", color = "grey55") +
      scale_color_manual(values = colors, breaks = version_map$label, labels = cal_lbl[version_map$label], drop = FALSE) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      labs(title = title, x = "Mean Predicted Probability", y = "Observed AKI Rate", color = NULL) +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12), legend.position = "inside", legend.position.inside = c(0.98, 0.02), legend.justification = c(1, 0), legend.background = element_rect(fill = grDevices::adjustcolor("white", alpha.f = 0.84), color = "grey80"), legend.text = element_text(size = 10.0), panel.grid.major = element_line(linewidth = 0.25, color = "grey90"), panel.grid.minor = element_blank())

    dca_plots[[i]] <- ggplot() +
      geom_line(data = ddf, aes(x = threshold, y = nb, color = version), linewidth = 0.9) +
      geom_line(data = treat_all, aes(x = threshold, y = nb), linetype = "dashed", color = "grey35", linewidth = 0.7) +
      geom_line(data = treat_none, aes(x = threshold, y = nb), linetype = "dotted", color = "grey55", linewidth = 0.7) +
      scale_color_manual(values = colors, breaks = version_map$label, drop = FALSE) +
      coord_cartesian(xlim = c(0, args$dca_max_threshold), ylim = c(args$dca_ymin, args$dca_ymax), expand = FALSE) +
      labs(title = title, x = "Threshold Probability", y = "Net Benefit", color = NULL) +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12), legend.position = "inside", legend.position.inside = c(0.98, 0.98), legend.justification = c(1, 1), legend.background = element_rect(fill = grDevices::adjustcolor("white", alpha.f = 0.84), color = "grey80"), legend.text = element_text(size = 10.0), panel.grid.major = element_line(linewidth = 0.25, color = "grey90"), panel.grid.minor = element_blank())
  }

  fig <- (roc_plots[[1]] + roc_plots[[2]] + roc_plots[[3]]) /
    (pr_plots[[1]] + pr_plots[[2]] + pr_plots[[3]]) /
    (cal_plots[[1]] + cal_plots[[2]] + cal_plots[[3]]) /
    (dca_plots[[1]] + dca_plots[[2]] + dca_plots[[3]])

  out_png <- file.path(args$outdir, paste0(args$prefix, ".png"))
  out_pdf <- file.path(args$outdir, paste0(args$prefix, ".pdf"))
  cap_path <- file.path(args$outdir, paste0(args$prefix, "_caption.txt"))
  stats_path <- file.path(root, "results", "paper_tables", paste0(args$prefix, "_metrics.csv"))
  dir.create(dirname(stats_path), recursive = TRUE, showWarnings = FALSE)

  ggsave(out_png, fig, width = args$width, height = args$height, dpi = args$dpi, bg = "white")
  ggsave(out_pdf, fig, width = args$width, height = args$height, dpi = args$dpi, bg = "white")
  write.csv(bind_rows(metric_rows), stats_path, row.names = FALSE)

  cap <- "Figure Sx. External validation 4x3 panel: ROC, PRC, calibration, and DCA (rows) for LightGBM, XGBoost, and CatBoost (columns). ROC/PRC are computed from raw probabilities; calibration/DCA from isotonic OOF-calibrated probabilities."
  writeLines(cap, cap_path)

  cat("Saved:\n")
  cat(out_png, "\n")
  cat(out_pdf, "\n")
  cat(cap_path, "\n")
  cat(stats_path, "\n")
}

main()
