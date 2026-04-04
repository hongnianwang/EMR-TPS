#!/usr/bin/env bash
set -euo pipefail

# 1) Build balanced intersection raw files from long TS tables
python src/01_make_mimic_intersection_raw.py \
  --positive-ts data/mimiciv/data-aki_ts.csv \
  --negative-ts data/mimiciv/data-control_ts.csv \
  --output-dir data/mimiciv/raw \
  --min-count 5

# 2) Convert raw intersection files into S3M-ready Train/Test matrices
python src/01_process_mimic_raw_to_s3m.py \
  --raw-dir data/mimiciv/raw \
  --processed-dir data/mimiciv/processed \
  --split-anchor-file dbp_combined_min5_intersection.csv \
  --test-size 0.2 \
  --random-state 42 \
  --save-flat-labels

