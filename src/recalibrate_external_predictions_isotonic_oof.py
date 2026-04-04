#!/usr/bin/env python3
"""Cross-fitted isotonic recalibration for external prediction probabilities.

This script calibrates v1/v2/v3 probabilities with 5-fold stratified out-of-fold
isotonic regression (default), which is more flexible than linear logit scaling
and avoids fitting/evaluating calibration on exactly the same samples.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        default="results/combined_predictions_external",
        help="Directory containing *_predictions.csv with y_true and v1/v2/v3 probabilities",
    )
    p.add_argument(
        "--output_dir",
        default="results/combined_predictions_external_isotonic_oof_calibrated",
    )
    p.add_argument(
        "--metric_out",
        default="results/paper_tables/external_isotonic_oof_recalibration_metrics.csv",
    )
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(float), 1e-8, 1 - 1e-8)


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = y.astype(float)
    p = p.astype(float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0, 1, bins + 1)

    b = np.digitize(p, edges[1:-1], right=True)
    total = len(y)
    ece = 0.0
    for i in range(len(edges) - 1):
        mask = b == i
        if mask.sum() == 0:
            continue
        ece += abs(y[mask].mean() - p[mask].mean()) * mask.sum() / total
    return float(ece)


def isotonic_oof(y: np.ndarray, p_raw: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    y = y.astype(int)
    p_raw = p_raw.astype(float)
    out = np.zeros_like(p_raw, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(p_raw.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw[tr], y[tr])
        out[te] = iso.transform(p_raw[te])

    return clip_prob(out)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metric_out), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*_predictions.csv")))
    if not files:
        raise FileNotFoundError(f"No *_predictions.csv found in {args.input_dir}")

    rows = []
    for fp in files:
        name = os.path.basename(fp)
        model = name.replace("_predictions.csv", "")
        df = pd.read_csv(fp)

        if "y_true" not in df.columns:
            raise ValueError(f"{fp} does not contain y_true")

        y = df["y_true"].to_numpy().astype(int)
        out_df = df.copy()

        prob_cols = [c for c in ["v1_prob", "v2_prob", "v3_prob"] if c in df.columns]
        if not prob_cols:
            raise ValueError(f"{fp} has no v1/v2/v3 columns")

        for i, c in enumerate(prob_cols):
            p_raw = clip_prob(df[c].to_numpy())
            p_cal = isotonic_oof(y, p_raw, n_splits=args.n_splits, seed=args.seed + i * 101)
            out_df[c] = p_cal

            rows.append(
                {
                    "model": model,
                    "version_col": c,
                    "prevalence": y.mean(),
                    "brier_raw": brier_score_loss(y, p_raw),
                    "brier_isotonic_oof": brier_score_loss(y, p_cal),
                    "ece_raw": ece_score(y, p_raw, bins=10),
                    "ece_isotonic_oof": ece_score(y, p_cal, bins=10),
                    "min_calibrated": float(np.min(p_cal)),
                    "p99_calibrated": float(np.quantile(p_cal, 0.99)),
                    "max_calibrated": float(np.max(p_cal)),
                }
            )

        out_path = os.path.join(args.output_dir, name)
        out_df.to_csv(out_path, index=False)
        print(f"saved {out_path}")

    pd.DataFrame(rows).to_csv(args.metric_out, index=False)
    print(f"saved {args.metric_out}")


if __name__ == "__main__":
    main()

