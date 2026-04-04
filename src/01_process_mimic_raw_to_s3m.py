#!/usr/bin/env python3
"""Convert MIMIC raw intersection CSVs to S3M Train/Test matrices.

"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/mimiciv/raw")
    parser.add_argument("--processed-dir", default="data/mimiciv/processed")
    parser.add_argument(
        "--split-anchor-file",
        default="dbp_combined_min5_intersection.csv",
        help="File used to derive train/test split.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--save-flat-labels",
        action="store_true",
        help="Also save y_train_2.csv/y_test_2.csv with stay_id,label columns.",
    )
    return parser.parse_args()


def clean_row_float(row: pd.Series, max_length: int | None = None) -> np.ndarray:
    first_non_zero = row.ne(0).idxmax() if not (row == 0).all() else None
    last_non_zero = row.ne(0)[::-1].idxmax() if not (row == 0).all() else None

    if first_non_zero is None or last_non_zero is None:
        return np.zeros(max_length) if max_length is not None else np.array([])

    row_trimmed = row.loc[first_non_zero:last_non_zero]
    row_filled = row_trimmed.replace(0, np.nan).ffill().bfill()
    result = np.round(row_filled.values, 1)

    if max_length is not None:
        if len(result) < max_length:
            pad_len = max_length - len(result)
            first_value = result[0] if len(result) > 0 else 0
            padding = np.full(pad_len, first_value)
            result = np.concatenate([padding, result])
        elif len(result) > max_length:
            result = result[-max_length:]
    return result


def calculate_max_lengths(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    max_lengths: Dict[str, int] = {}
    for key, df in data_dict.items():
        max_length = 0
        for _, row in df.iterrows():
            # Legacy-compatible behavior from notebook: skip stay_id only.
            data_row = row.iloc[1:]
            if not (data_row == 0).all():
                non_zero_indices = data_row.reset_index(drop=True).to_numpy().nonzero()[0]
                if len(non_zero_indices) > 0:
                    first_idx = non_zero_indices[0]
                    last_idx = non_zero_indices[-1]
                    max_length = max(max_length, int(last_idx - first_idx + 1))
        max_lengths[key] = max_length
    return max_lengths


def main() -> None:
    args = parse_args()
    os.makedirs(args.processed_dir, exist_ok=True)

    files = sorted(
        f
        for f in os.listdir(args.raw_dir)
        if f.endswith(".csv") and "med" not in f and "combined_min" in f and "intersection" in f
    )
    if not files:
        raise FileNotFoundError(f"No intersection csv files found in {args.raw_dir}")

    data_dict: Dict[str, pd.DataFrame] = {}
    for f in files:
        key = f.split("_")[0]
        df = pd.read_csv(os.path.join(args.raw_dir, f)).fillna(0)
        if "stay_id" not in df.columns or "label" not in df.columns:
            continue
        data_dict[key] = df

    if not data_dict:
        raise ValueError("No valid data files with stay_id/label found.")

    split_key = args.split_anchor_file.split("_")[0]
    if split_key not in data_dict:
        available = ", ".join(sorted(data_dict.keys()))
        raise ValueError(
            f"Split anchor key '{split_key}' not found. Available: {available}"
        )

    base_df = data_dict[split_key]
    X = base_df.drop("label", axis=1)
    y = base_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    train_labels = pd.DataFrame({"stay_id": X_train["stay_id"].values, "label": y_train.values})
    test_labels = pd.DataFrame({"stay_id": X_test["stay_id"].values, "label": y_test.values})

    # Main labels: index = stay_id (compatible with legacy read_csv(index_col=0))
    y_train_out = train_labels.set_index("stay_id")
    y_test_out = test_labels.set_index("stay_id")
    y_train_path = os.path.join(args.processed_dir, "..", "y_train.csv")
    y_test_path = os.path.join(args.processed_dir, "..", "y_test.csv")
    y_train_out.to_csv(y_train_path)
    y_test_out.to_csv(y_test_path)

    if args.save_flat_labels:
        train_labels.to_csv(os.path.join(args.processed_dir, "..", "y_train_2.csv"), index=False)
        test_labels.to_csv(os.path.join(args.processed_dir, "..", "y_test_2.csv"), index=False)

    indices_train = set(y_train_out.index.tolist())
    indices_test = set(y_test_out.index.tolist())

    max_lengths = calculate_max_lengths(data_dict)

    for key, df in data_dict.items():
        train_df = df[df["stay_id"].isin(indices_train)].set_index("stay_id")
        test_df = df[df["stay_id"].isin(indices_test)].set_index("stay_id")
        current_max_length = max_lengths[key]

        train_filename = os.path.join(args.processed_dir, f"Train_{key}.csv")
        with open(train_filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", lineterminator="\n")
            for _, rows in train_df.iterrows():
                cleaned = clean_row_float(rows.iloc[1:], current_max_length)
                if len(cleaned) > 1:
                    row_cleaned = np.concatenate([[rows.iloc[0]], cleaned])
                    writer.writerow([str(x).strip() for x in row_cleaned])
                else:
                    writer.writerow([rows.iloc[0]])

        test_filename = os.path.join(args.processed_dir, f"Test_{key}.csv")
        with open(test_filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", lineterminator="\n")
            for _, rows in test_df.iterrows():
                cleaned = clean_row_float(rows.iloc[1:], current_max_length)
                if len(cleaned) > 1:
                    row_cleaned = np.concatenate([[rows.iloc[0]], cleaned])
                    writer.writerow([str(x).strip() for x in row_cleaned])
                else:
                    writer.writerow([rows.iloc[0]])

        print(f"Saved {train_filename}")
        print(f"Saved {test_filename}")

    print("Done.")
    print(f"Split sizes: train={len(indices_train)}, test={len(indices_test)}")
    print(f"Labels: {y_train_path}, {y_test_path}")


if __name__ == "__main__":
    main()

