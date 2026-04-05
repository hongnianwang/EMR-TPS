# MIMIC-IV Inputs

Required access: PhysioNet credentialed MIMIC-IV.

Typical processed files used by this project:
- `processed_train_with_shapelets.csv`
- `processed_test_with_shapelets.csv`
- `X_train_with_shapelets.csv`

Key SQL for cohort construction and flow statistics:
- `sql/mimic/cohort.sql`
- `sql/mimic/mimic_flow_statistics.sql`

Run SQL in PostgreSQL with MIMIC-IV schemas (`mimiciv_hosp`, `mimiciv_icu`, `mimiciv_derived`) available.

Do not commit patient-level files to git.
