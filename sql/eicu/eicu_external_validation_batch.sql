-- ============================================================================
-- eICU external validation extraction (batch mode, low temp-disk footprint)
--
-- Input: public.eicu_cohort_index(patientunitstayid, label, prediction_offset_min)
-- Window: [ICU admit - 6h, prediction_time - 12h]
--
-- Outputs:
--   public.eicu_external_validation_tabular
--   public.eicu_external_validation_ts_long
-- ============================================================================

SET search_path TO public, eicu_crd;
SET max_parallel_workers_per_gather = 0;
SET work_mem = '16MB';

DROP TABLE IF EXISTS public.eicu_external_validation_tabular;
DROP TABLE IF EXISTS public.eicu_external_validation_ts_raw;
DROP TABLE IF EXISTS public.eicu_external_validation_ts_long;

CREATE OR REPLACE TEMP VIEW ev_cohort AS
SELECT
    c.patientunitstayid::bigint AS stay_id,
    c.label::int AS label,
    c.prediction_offset_min::int AS prediction_offset_min,
    (-6 * 60) AS window_start_min,
    (c.prediction_offset_min::int - 12 * 60) AS window_end_min
FROM public.eicu_cohort_index c
JOIN eicu_crd.patient p
    ON p.patientunitstayid = c.patientunitstayid
WHERE
    c.prediction_offset_min BETWEEN 24 * 60 AND 72 * 60
    AND p.unitdischargeoffset >= 24 * 60
    AND (c.prediction_offset_min::int - 12 * 60) >= (-6 * 60);

CREATE OR REPLACE TEMP VIEW ev_demographics AS
SELECT
    p.patientunitstayid::bigint AS stay_id,
    COALESCE(NULLIF(trim(p.gender), ''), 'Unknown') AS gender,
    CASE
        WHEN trim(p.age) = '> 89' THEN 90
        WHEN trim(p.age) ~ '^[0-9]+$' THEN trim(p.age)::int
        ELSE NULL
    END AS age,
    CASE
        -- IMPORTANT: check Caucasian before Asian, otherwise "Caucasian"
        -- would be incorrectly matched by "%asian%".
        WHEN p.ethnicity ILIKE '%caucasian%' OR p.ethnicity ILIKE '%white%' THEN 'White'
        WHEN p.ethnicity ILIKE '%asian%' THEN 'Asian'
        WHEN p.ethnicity ILIKE '%african%' THEN 'Black'
        WHEN p.ethnicity ILIKE '%hispanic%' THEN 'Hispanic'
        WHEN p.ethnicity ILIKE '%other%' AND p.ethnicity NOT ILIKE '%unknown%' THEN 'Other'
        ELSE 'Unknown'
    END AS race
FROM eicu_crd.patient p
JOIN ev_cohort c
    ON c.stay_id = p.patientunitstayid;

CREATE TABLE public.eicu_external_validation_tabular AS
SELECT
    c.stay_id,
    c.label,
    d.gender,
    d.age,
    d.race,

    NULL::numeric AS heart_rate_last,
    NULL::numeric AS sbp_last,
    NULL::numeric AS dbp_last,
    NULL::numeric AS respiratory_rate_last,
    NULL::numeric AS o2_saturation_last,
    NULL::numeric AS temperature_last,

    NULL::numeric AS bun_last,
    NULL::numeric AS wbc_last,
    NULL::numeric AS potassium_last,
    NULL::numeric AS calcium_last,
    NULL::numeric AS creatinine_last,
    NULL::numeric AS glucose_last,
    NULL::numeric AS magnesium_last,
    NULL::numeric AS sodium_last,
    NULL::numeric AS hemoglobin_last,
    NULL::numeric AS platelet_last,
    NULL::numeric AS bicarbonate_last,
    NULL::numeric AS chloride_last,
    NULL::numeric AS lactate_last,
    NULL::numeric AS hematocrit_last,
    NULL::numeric AS rbc_last,

    NULL::numeric AS heart_rate_max,
    NULL::numeric AS heart_rate_min,
    NULL::numeric AS sbp_max,
    NULL::numeric AS sbp_min,
    NULL::numeric AS dbp_max,
    NULL::numeric AS dbp_min,
    NULL::numeric AS respiratory_rate_max,
    NULL::numeric AS respiratory_rate_min,
    NULL::numeric AS o2_saturation_max,
    NULL::numeric AS o2_saturation_min,
    NULL::numeric AS temperature_max,
    NULL::numeric AS temperature_min,

    NULL::numeric AS bun_max,
    NULL::numeric AS bun_min,
    NULL::numeric AS wbc_max,
    NULL::numeric AS wbc_min,
    NULL::numeric AS potassium_max,
    NULL::numeric AS potassium_min,
    NULL::numeric AS calcium_max,
    NULL::numeric AS calcium_min,
    NULL::numeric AS creatinine_max,
    NULL::numeric AS creatinine_min,
    NULL::numeric AS glucose_max,
    NULL::numeric AS glucose_min,
    NULL::numeric AS magnesium_max,
    NULL::numeric AS magnesium_min,
    NULL::numeric AS sodium_max,
    NULL::numeric AS sodium_min,
    NULL::numeric AS hemoglobin_max,
    NULL::numeric AS hemoglobin_min,
    NULL::numeric AS platelet_max,
    NULL::numeric AS platelet_min,
    NULL::numeric AS bicarbonate_max,
    NULL::numeric AS bicarbonate_min,
    NULL::numeric AS chloride_max,
    NULL::numeric AS chloride_min,
    NULL::numeric AS lactate_max,
    NULL::numeric AS lactate_min,
    NULL::numeric AS hematocrit_max,
    NULL::numeric AS hematocrit_min,
    NULL::numeric AS rbc_max,
    NULL::numeric AS rbc_min
FROM ev_cohort c
LEFT JOIN ev_demographics d
    ON d.stay_id = c.stay_id;

CREATE INDEX idx_eicu_external_validation_tabular_stay
    ON public.eicu_external_validation_tabular(stay_id);

-- =============================
-- Vital: heart_rate
-- =============================
WITH ev AS (
    SELECT
        c.stay_id,
        vp.observationoffset::int AS offset_min,
        AVG(vp.heartrate::numeric) AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalperiodic vp
      ON vp.patientunitstayid = c.stay_id
    WHERE
      vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.heartrate BETWEEN 25 AND 225
    GROUP BY c.stay_id, vp.observationoffset
), agg AS (
    SELECT stay_id, MAX(value) AS max_v, MIN(value) AS min_v, MAX(offset_min) AS last_off
    FROM ev
    GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) AS last_v
    FROM ev e
    JOIN agg a ON a.stay_id = e.stay_id AND a.last_off = e.offset_min
    GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET heart_rate_last = l.last_v, heart_rate_max = a.max_v, heart_rate_min = a.min_v
FROM agg a
LEFT JOIN last_val l ON l.stay_id = a.stay_id
WHERE t.stay_id = a.stay_id;

-- =============================
-- Vital: sbp (periodic + aperiodic)
-- =============================
WITH ev_raw AS (
    SELECT c.stay_id, vp.observationoffset::int AS offset_min, vp.systemicsystolic::numeric AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalperiodic vp
      ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.systemicsystolic BETWEEN 25 AND 250

    UNION ALL

    SELECT c.stay_id, va.observationoffset::int AS offset_min, va.noninvasivesystolic::numeric AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalaperiodic va
      ON va.patientunitstayid = c.stay_id
    WHERE va.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND va.noninvasivesystolic BETWEEN 25 AND 250
), ev AS (
    SELECT stay_id, offset_min, AVG(value) AS value
    FROM ev_raw
    GROUP BY stay_id, offset_min
), agg AS (
    SELECT stay_id, MAX(value) AS max_v, MIN(value) AS min_v, MAX(offset_min) AS last_off
    FROM ev
    GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) AS last_v
    FROM ev e
    JOIN agg a ON a.stay_id = e.stay_id AND a.last_off = e.offset_min
    GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET sbp_last = l.last_v, sbp_max = a.max_v, sbp_min = a.min_v
FROM agg a
LEFT JOIN last_val l ON l.stay_id = a.stay_id
WHERE t.stay_id = a.stay_id;

-- =============================
-- Vital: dbp (periodic + aperiodic)
-- =============================
WITH ev_raw AS (
    SELECT c.stay_id, vp.observationoffset::int AS offset_min, vp.systemicdiastolic::numeric AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalperiodic vp
      ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.systemicdiastolic BETWEEN 1 AND 200

    UNION ALL

    SELECT c.stay_id, va.observationoffset::int AS offset_min, va.noninvasivediastolic::numeric AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalaperiodic va
      ON va.patientunitstayid = c.stay_id
    WHERE va.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND va.noninvasivediastolic BETWEEN 1 AND 200
), ev AS (
    SELECT stay_id, offset_min, AVG(value) AS value
    FROM ev_raw
    GROUP BY stay_id, offset_min
), agg AS (
    SELECT stay_id, MAX(value) AS max_v, MIN(value) AS min_v, MAX(offset_min) AS last_off
    FROM ev
    GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) AS last_v
    FROM ev e
    JOIN agg a ON a.stay_id = e.stay_id AND a.last_off = e.offset_min
    GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET dbp_last = l.last_v, dbp_max = a.max_v, dbp_min = a.min_v
FROM agg a
LEFT JOIN last_val l ON l.stay_id = a.stay_id
WHERE t.stay_id = a.stay_id;

-- =============================
-- Vital: respiratory_rate
-- =============================
WITH ev AS (
    SELECT
        c.stay_id,
        vp.observationoffset::int AS offset_min,
        AVG(vp.respiration::numeric) AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalperiodic vp
      ON vp.patientunitstayid = c.stay_id
    WHERE
      vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.respiration BETWEEN 0 AND 70
    GROUP BY c.stay_id, vp.observationoffset
), agg AS (
    SELECT stay_id, MAX(value) AS max_v, MIN(value) AS min_v, MAX(offset_min) AS last_off
    FROM ev
    GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) AS last_v
    FROM ev e
    JOIN agg a ON a.stay_id = e.stay_id AND a.last_off = e.offset_min
    GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET respiratory_rate_last = l.last_v, respiratory_rate_max = a.max_v, respiratory_rate_min = a.min_v
FROM agg a
LEFT JOIN last_val l ON l.stay_id = a.stay_id
WHERE t.stay_id = a.stay_id;

-- =============================
-- Vital: o2_saturation
-- =============================
WITH ev AS (
    SELECT
        c.stay_id,
        vp.observationoffset::int AS offset_min,
        AVG(vp.sao2::numeric) AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalperiodic vp
      ON vp.patientunitstayid = c.stay_id
    WHERE
      vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.sao2 BETWEEN 0 AND 100
    GROUP BY c.stay_id, vp.observationoffset
), agg AS (
    SELECT stay_id, MAX(value) AS max_v, MIN(value) AS min_v, MAX(offset_min) AS last_off
    FROM ev
    GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) AS last_v
    FROM ev e
    JOIN agg a ON a.stay_id = e.stay_id AND a.last_off = e.offset_min
    GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET o2_saturation_last = l.last_v, o2_saturation_max = a.max_v, o2_saturation_min = a.min_v
FROM agg a
LEFT JOIN last_val l ON l.stay_id = a.stay_id
WHERE t.stay_id = a.stay_id;

-- =============================
-- Vital: temperature
-- =============================
WITH ev AS (
    SELECT
        c.stay_id,
        vp.observationoffset::int AS offset_min,
        AVG(vp.temperature::numeric) AS value
    FROM ev_cohort c
    JOIN eicu_crd.vitalperiodic vp
      ON vp.patientunitstayid = c.stay_id
    WHERE
      vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.temperature BETWEEN 25 AND 46
    GROUP BY c.stay_id, vp.observationoffset
), agg AS (
    SELECT stay_id, MAX(value) AS max_v, MIN(value) AS min_v, MAX(offset_min) AS last_off
    FROM ev
    GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) AS last_v
    FROM ev e
    JOIN agg a ON a.stay_id = e.stay_id AND a.last_off = e.offset_min
    GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET temperature_last = l.last_v, temperature_max = a.max_v, temperature_min = a.min_v
FROM agg a
LEFT JOIN last_val l ON l.stay_id = a.stay_id
WHERE t.stay_id = a.stay_id;

-- =============================
-- Lab helper macro pattern: per-variable update
-- =============================
-- bun
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'bun' AND l.labresult > 0 AND l.labresult <= 300
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET bun_last=l.last_v, bun_max=a.max_v, bun_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- wbc
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'wbc x 1000' AND l.labresult > 0 AND l.labresult <= 1000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET wbc_last=l.last_v, wbc_max=a.max_v, wbc_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- potassium
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'potassium' AND l.labresult > 0 AND l.labresult <= 30
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET potassium_last=l.last_v, potassium_max=a.max_v, potassium_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- calcium
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'calcium' AND l.labresult > 0 AND l.labresult <= 10000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET calcium_last=l.last_v, calcium_max=a.max_v, calcium_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- creatinine
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'creatinine' AND l.labresult > 0 AND l.labresult <= 30
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET creatinine_last=l.last_v, creatinine_max=a.max_v, creatinine_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- glucose
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) IN ('glucose', 'bedside glucose') AND l.labresult > 0 AND l.labresult <= 30000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET glucose_last=l.last_v, glucose_max=a.max_v, glucose_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- magnesium
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'magnesium' AND l.labresult > 0 AND l.labresult <= 10000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET magnesium_last=l.last_v, magnesium_max=a.max_v, magnesium_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- sodium
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'sodium' AND l.labresult > 0 AND l.labresult <= 200
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET sodium_last=l.last_v, sodium_max=a.max_v, sodium_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- hemoglobin
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'hgb' AND l.labresult > 0 AND l.labresult <= 50
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET hemoglobin_last=l.last_v, hemoglobin_max=a.max_v, hemoglobin_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- platelet
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'platelets x 1000' AND l.labresult > 0 AND l.labresult <= 10000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET platelet_last=l.last_v, platelet_max=a.max_v, platelet_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- bicarbonate
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) IN ('bicarbonate', 'hco3', 'total co2') AND l.labresult > 0 AND l.labresult <= 10000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET bicarbonate_last=l.last_v, bicarbonate_max=a.max_v, bicarbonate_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- chloride
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'chloride' AND l.labresult > 0 AND l.labresult <= 10000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET chloride_last=l.last_v, chloride_max=a.max_v, chloride_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- lactate
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'lactate' AND l.labresult > 0 AND l.labresult <= 10000
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET lactate_last=l.last_v, lactate_max=a.max_v, lactate_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- hematocrit
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'hct' AND l.labresult > 0 AND l.labresult <= 100
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET hematocrit_last=l.last_v, hematocrit_max=a.max_v, hematocrit_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- rbc
WITH ev AS (
    SELECT c.stay_id, l.labresultoffset::int AS offset_min, AVG(l.labresult::numeric) AS value
    FROM ev_cohort c JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
    WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
      AND lower(l.labname) = 'rbc' AND l.labresult > 0 AND l.labresult <= 10
    GROUP BY c.stay_id, l.labresultoffset
), agg AS (
    SELECT stay_id, MAX(value) max_v, MIN(value) min_v, MAX(offset_min) last_off FROM ev GROUP BY stay_id
), last_val AS (
    SELECT e.stay_id, AVG(e.value) last_v FROM ev e JOIN agg a ON a.stay_id=e.stay_id AND a.last_off=e.offset_min GROUP BY e.stay_id
)
UPDATE public.eicu_external_validation_tabular t
SET rbc_last=l.last_v, rbc_max=a.max_v, rbc_min=a.min_v
FROM agg a LEFT JOIN last_val l ON l.stay_id=a.stay_id
WHERE t.stay_id=a.stay_id;

-- =============================
-- Time series extraction (7 vars)
-- =============================
CREATE TABLE public.eicu_external_validation_ts_raw (
    stay_id bigint,
    label int,
    variable text,
    resample_hours int,
    bin_offset_min int,
    value numeric
);

-- heart_rate (1h)
INSERT INTO public.eicu_external_validation_ts_raw
SELECT
    c.stay_id,
    c.label,
    'heart_rate'::text AS variable,
    1 AS resample_hours,
    FLOOR(vp.observationoffset / 60.0)::int * 60 AS bin_offset_min,
    AVG(vp.heartrate::numeric) AS value
FROM ev_cohort c
JOIN eicu_crd.vitalperiodic vp ON vp.patientunitstayid = c.stay_id
WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
  AND vp.heartrate BETWEEN 25 AND 225
GROUP BY c.stay_id, c.label, FLOOR(vp.observationoffset / 60.0)::int * 60;

-- sbp (1h)
INSERT INTO public.eicu_external_validation_ts_raw
WITH raw AS (
    SELECT c.stay_id, c.label, vp.observationoffset::int AS offset_min, vp.systemicsystolic::numeric AS value
    FROM ev_cohort c JOIN eicu_crd.vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.systemicsystolic BETWEEN 25 AND 250
    UNION ALL
    SELECT c.stay_id, c.label, va.observationoffset::int AS offset_min, va.noninvasivesystolic::numeric AS value
    FROM ev_cohort c JOIN eicu_crd.vitalaperiodic va ON va.patientunitstayid = c.stay_id
    WHERE va.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND va.noninvasivesystolic BETWEEN 25 AND 250
)
SELECT
    stay_id,
    label,
    'sbp'::text AS variable,
    1 AS resample_hours,
    FLOOR(offset_min / 60.0)::int * 60 AS bin_offset_min,
    AVG(value) AS value
FROM raw
GROUP BY stay_id, label, FLOOR(offset_min / 60.0)::int * 60;

-- dbp (1h)
INSERT INTO public.eicu_external_validation_ts_raw
WITH raw AS (
    SELECT c.stay_id, c.label, vp.observationoffset::int AS offset_min, vp.systemicdiastolic::numeric AS value
    FROM ev_cohort c JOIN eicu_crd.vitalperiodic vp ON vp.patientunitstayid = c.stay_id
    WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND vp.systemicdiastolic BETWEEN 1 AND 200
    UNION ALL
    SELECT c.stay_id, c.label, va.observationoffset::int AS offset_min, va.noninvasivediastolic::numeric AS value
    FROM ev_cohort c JOIN eicu_crd.vitalaperiodic va ON va.patientunitstayid = c.stay_id
    WHERE va.observationoffset BETWEEN c.window_start_min AND c.window_end_min
      AND va.noninvasivediastolic BETWEEN 1 AND 200
)
SELECT
    stay_id,
    label,
    'dbp'::text AS variable,
    1 AS resample_hours,
    FLOOR(offset_min / 60.0)::int * 60 AS bin_offset_min,
    AVG(value) AS value
FROM raw
GROUP BY stay_id, label, FLOOR(offset_min / 60.0)::int * 60;

-- o2_saturation (1h)
INSERT INTO public.eicu_external_validation_ts_raw
SELECT
    c.stay_id,
    c.label,
    'o2_saturation'::text AS variable,
    1 AS resample_hours,
    FLOOR(vp.observationoffset / 60.0)::int * 60 AS bin_offset_min,
    AVG(vp.sao2::numeric) AS value
FROM ev_cohort c
JOIN eicu_crd.vitalperiodic vp ON vp.patientunitstayid = c.stay_id
WHERE vp.observationoffset BETWEEN c.window_start_min AND c.window_end_min
  AND vp.sao2 BETWEEN 0 AND 100
GROUP BY c.stay_id, c.label, FLOOR(vp.observationoffset / 60.0)::int * 60;

-- bun (4h)
INSERT INTO public.eicu_external_validation_ts_raw
SELECT
    c.stay_id,
    c.label,
    'bun'::text AS variable,
    4 AS resample_hours,
    FLOOR(l.labresultoffset / 240.0)::int * 240 AS bin_offset_min,
    AVG(l.labresult::numeric) AS value
FROM ev_cohort c
JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
  AND lower(l.labname) = 'bun'
  AND l.labresult > 0 AND l.labresult <= 300
GROUP BY c.stay_id, c.label, FLOOR(l.labresultoffset / 240.0)::int * 240;

-- creatinine (4h)
INSERT INTO public.eicu_external_validation_ts_raw
SELECT
    c.stay_id,
    c.label,
    'creatinine'::text AS variable,
    4 AS resample_hours,
    FLOOR(l.labresultoffset / 240.0)::int * 240 AS bin_offset_min,
    AVG(l.labresult::numeric) AS value
FROM ev_cohort c
JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
  AND lower(l.labname) = 'creatinine'
  AND l.labresult > 0 AND l.labresult <= 30
GROUP BY c.stay_id, c.label, FLOOR(l.labresultoffset / 240.0)::int * 240;

-- potassium (4h)
INSERT INTO public.eicu_external_validation_ts_raw
SELECT
    c.stay_id,
    c.label,
    'potassium'::text AS variable,
    4 AS resample_hours,
    FLOOR(l.labresultoffset / 240.0)::int * 240 AS bin_offset_min,
    AVG(l.labresult::numeric) AS value
FROM ev_cohort c
JOIN eicu_crd.lab l ON l.patientunitstayid = c.stay_id
WHERE l.labresultoffset BETWEEN c.window_start_min AND c.window_end_min
  AND lower(l.labname) = 'potassium'
  AND l.labresult > 0 AND l.labresult <= 30
GROUP BY c.stay_id, c.label, FLOOR(l.labresultoffset / 240.0)::int * 240;

CREATE TABLE public.eicu_external_validation_ts_long AS
WITH counts AS (
    SELECT stay_id, variable, COUNT(*) AS n_obs
    FROM public.eicu_external_validation_ts_raw
    GROUP BY stay_id, variable
), eligible AS (
    SELECT stay_id
    FROM counts
    GROUP BY stay_id
    HAVING COUNT(*) = 7 AND MIN(n_obs) >= 5
)
SELECT
    r.stay_id,
    r.label,
    r.variable,
    r.resample_hours,
    r.bin_offset_min,
    (r.bin_offset_min / 60.0)::numeric(8,2) AS bin_hour_from_icu,
    r.value
FROM public.eicu_external_validation_ts_raw r
JOIN eligible e ON e.stay_id = r.stay_id
ORDER BY r.stay_id, r.variable, r.bin_offset_min;

DROP TABLE IF EXISTS public.eicu_external_validation_ts_raw;

ANALYZE public.eicu_external_validation_tabular;
ANALYZE public.eicu_external_validation_ts_long;

SELECT COUNT(*) AS n_tabular FROM public.eicu_external_validation_tabular;
SELECT variable, resample_hours, COUNT(*) AS n_points
FROM public.eicu_external_validation_ts_long
GROUP BY variable, resample_hours
ORDER BY variable;
