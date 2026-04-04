## Overview

This repository provides the implementation of EMR-TPS for ICU AKI prediction, including temporal pattern (shapelet) discovery, feature integration, model development in MIMIC-IV, and external validation in eICU-CRD.

## Repository Structure

- `src/`: pipeline scripts (preprocessing, feature integration, modeling, validation, visualization)
- `src/utils/`: shapelet evaluation and distance utilities
- `scripts/`: convenience run scripts
- `data/`: local data placeholders (no patient-level files included)
- `results/`: local output placeholders (no heavy artifacts included)

## Setup & Requirements

**Environment:** Python 3.10+, pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, catboost, shap.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**S3M dependency:**

```bash
wget https://github.com/BorgwardtLab/S3M/releases/download/v1.0.0-alpha/s3m-1.0.0-alpha.deb
sudo apt-get update
sudo dpkg -i s3m-1.0.0-alpha.deb || sudo apt --fix-broken install -y
```

**Data access permissions (required):**

1. Create a PhysioNet account: https://physionet.org/
2. Complete credentialing for Credentialed Health Data: https://physionet.org/settings/credentialing/
3. Complete required human-subjects training (typically CITI): https://about.citiprogram.org/
4. Request access and accept DUA:
   - MIMIC-IV: https://physionet.org/content/mimiciv/
   - eICU-CRD: https://physionet.org/content/eicu-crd/

See data notes:

- `data/mimiciv/README.md`
- `data/eicu/README.md`

## Usage Pipeline

1. MIMIC raw preprocessing (min-count intersection + train/test matrix build)
   - `src/01_make_mimic_intersection_raw.py`
   - `src/01_process_mimic_raw_to_s3m.py`
2. Shapelet mining runners
   - `src/02_run_s3m_vital.py`
   - `src/02_run_s3m_lab.py`
3. MIMIC feature integration and internal modeling
   - `src/03_integrate_shapelet_features.py`
   - `src/03_model_machine_learing.py`
4. eICU shapelet feature alignment and external validation
   - `src/03_integrate_shapelet_features_eicu.py`
   - `src/04_external_validate_eicu.py`
5. Optional recalibration
   - `src/recalibrate_external_predictions_isotonic_oof.py`
6. Performance tables
   - `src/generate_table2_table3_performance.py`

## Example Commands

```bash
# MIMIC preprocessing
bash scripts/run_mimic_preprocessing.sh

# MIMIC internal modeling
python src/03_integrate_shapelet_features.py
python src/03_model_machine_learing.py

# eICU feature alignment
python src/03_integrate_shapelet_features_eicu.py

# External validation
python src/04_external_validate_eicu.py \
  --train_file data/mimiciv/processed/processed_train_with_shapelets.csv \
  --test_file data/eicu/processed/processed_test_with_shapelets.csv \
  --versions 1 2 3

# Table generation
bash scripts/run_table_generation.sh <internal_pred_dir> <external_pred_dir>
```

## Results Reproduction

- **Table 1 (baseline characteristics):**
  - `src/table1_combined_mimic_eicu.r`
- **Figure 2 (shapelet pattern analysis):**
  - `src/figure2a_shapelet.py`
  - `src/figure2bc_shapelet_density_boxplot_visualization.r`
  - `src/figure2de_shapelet_decision_tree_visualization.py`
  - `src/figure2f_shapelet_accuracy_analysis.py`
- **Figure 3 (model performance):**
  - `src/figure3_mimic_internal_3x3_dedicated.R`
- **Figure 4 (interpretability):**
  - `src/figure4abc_shap_analysis.py`
- **External supplementary figure(s):**
  - `src/figureS2_external_full_4x3_dedicated.R`

## Acknowledgements

We acknowledge BorgwardtLab for releasing S3M (https://github.com/BorgwardtLab/S3M), which we use as the shapelet mining backend, and related code in src/utils/ is also adapted from the same repository.

## License

MIT
