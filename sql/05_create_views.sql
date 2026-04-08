-- =============================================================================
-- 05_create_views.sql
-- NHS Waiting Time Forecasting — Analytical Views
-- Francis Kwesi Acquah | B01821156 | UWS
-- =============================================================================

SET search_path TO nhs, public;

-- ── v_rtt_regional_monthly ────────────────────────────────────────────────────
-- PRIMARY view used by all forecasting scripts.
-- Aggregates to modern-region level, returns all fields needed for ARIMA/ETS/Prophet.
CREATE OR REPLACE VIEW nhs.v_rtt_regional_monthly AS
SELECT
    c.period_date,
    c.year,
    c.month,
    c.month_name,
    c.year_month,
    c.quarter,
    c.is_pre_covid,
    c.is_covid_period,
    c.is_post_covid,
    c.covid_dummy,
    r.modern_region_code                          AS region_code,
    r.modern_region_name                          AS region_name,
    r.region_short,
    t.function_code                               AS treatment_code,
    t.function_name                               AS treatment_name,
    t.specialty_group,
    p.part_type,
    p.part_description,
    SUM(f.total_all)                              AS waiting_list_size,
    SUM(f.patients_within_18wks)                  AS patients_within_18wks,
    SUM(f.patients_over_18wks)                    AS patients_over_18wks,
    CASE
        WHEN SUM(f.total_all) > 0
        THEN ROUND(SUM(f.patients_within_18wks)::NUMERIC / NULLIF(SUM(f.total_all),0) * 100, 4)
        ELSE NULL
    END                                           AS pct_within_18wks,
    SUM(f.wk_52_plus)                             AS waiting_over_52wks,
    SUM(f.total_patients)                         AS total_patients,
    COUNT(DISTINCT f.provider_org_code)           AS provider_count,
    COUNT(DISTINCT f.commissioner_org_code)       AS commissioner_count
FROM nhs.fact_rtt_monthly    f
JOIN nhs.dim_calendar         c ON f.calendar_id  = c.calendar_id
JOIN nhs.dim_region           r ON f.region_id    = r.region_id
JOIN nhs.dim_treatment        t ON f.treatment_id = t.treatment_id
JOIN nhs.dim_rtt_part         p ON f.part_id      = p.part_id
WHERE r.modern_region_code IS NOT NULL
GROUP BY
    c.period_date, c.year, c.month, c.month_name, c.year_month, c.quarter,
    c.is_pre_covid, c.is_covid_period, c.is_post_covid, c.covid_dummy,
    r.modern_region_code, r.modern_region_name, r.region_short,
    t.function_code, t.function_name, t.specialty_group,
    p.part_type, p.part_description
ORDER BY c.period_date, r.modern_region_code, t.function_code;

COMMENT ON VIEW nhs.v_rtt_regional_monthly IS
    'PRIMARY forecasting view — regional RTT aggregation with all COVID flags. Used by all model scripts.';

-- ── v_rtt_specialty_monthly ───────────────────────────────────────────────────
CREATE OR REPLACE VIEW nhs.v_rtt_specialty_monthly AS
SELECT
    c.period_date, c.year, c.month, c.year_month,
    c.is_covid_period, c.covid_dummy,
    r.modern_region_code    AS region_code,
    r.modern_region_name    AS region_name,
    t.function_code         AS treatment_code,
    t.function_name         AS treatment_name,
    t.specialty_group,
    p.part_type,
    SUM(f.total_all)                              AS waiting_list_size,
    SUM(f.patients_within_18wks)                  AS patients_within_18wks,
    SUM(f.patients_over_18wks)                    AS patients_over_18wks,
    CASE
        WHEN SUM(f.total_all) > 0
        THEN ROUND(SUM(f.patients_within_18wks)::NUMERIC / NULLIF(SUM(f.total_all),0) * 100, 4)
        ELSE NULL
    END                                           AS pct_within_18wks,
    SUM(f.wk_52_plus)                             AS waiting_over_52wks
FROM nhs.fact_rtt_monthly    f
JOIN nhs.dim_calendar         c ON f.calendar_id  = c.calendar_id
JOIN nhs.dim_region           r ON f.region_id    = r.region_id
JOIN nhs.dim_treatment        t ON f.treatment_id = t.treatment_id
JOIN nhs.dim_rtt_part         p ON f.part_id      = p.part_id
WHERE t.is_total = FALSE
  AND r.modern_region_code IS NOT NULL
GROUP BY
    c.period_date, c.year, c.month, c.year_month, c.is_covid_period, c.covid_dummy,
    r.modern_region_code, r.modern_region_name,
    t.function_code, t.function_name, t.specialty_group, p.part_type
ORDER BY c.period_date, r.modern_region_code, t.function_code;

COMMENT ON VIEW nhs.v_rtt_specialty_monthly IS 'Specialty-level RTT — excludes C_999 Total rows';

-- ── v_new_periods_regional ────────────────────────────────────────────────────
CREATE OR REPLACE VIEW nhs.v_new_periods_regional AS
SELECT
    c.period_date, c.year, c.month, c.year_month,
    c.is_covid_period, c.covid_dummy,
    r.modern_region_code    AS region_code,
    r.modern_region_name    AS region_name,
    t.function_code         AS treatment_code,
    t.function_name         AS treatment_name,
    SUM(np.new_periods_count)                     AS new_periods_count
FROM nhs.fact_new_periods_monthly  np
JOIN nhs.dim_calendar               c  ON np.calendar_id  = c.calendar_id
JOIN nhs.dim_region                 r  ON np.region_id    = r.region_id
JOIN nhs.dim_treatment              t  ON np.treatment_id = t.treatment_id
WHERE r.modern_region_code IS NOT NULL
GROUP BY
    c.period_date, c.year, c.month, c.year_month, c.is_covid_period, c.covid_dummy,
    r.modern_region_code, r.modern_region_name, t.function_code, t.function_name
ORDER BY c.period_date, r.modern_region_code;

COMMENT ON VIEW nhs.v_new_periods_regional IS 'New RTT referral starts per region per month — regressor for Prophet';

-- ── v_model_leaderboard ───────────────────────────────────────────────────────
CREATE OR REPLACE VIEW nhs.v_model_leaderboard AS
SELECT
    r.modern_region_code        AS region_code,
    r.modern_region_name        AS region_name,
    t.function_code             AS treatment_code,
    t.function_name             AS treatment_name,
    dm.model_name,
    dm.model_type,
    COUNT(cv.cv_id)             AS fold_count,
    ROUND(AVG(cv.mae),  4)      AS mean_mae,
    ROUND(AVG(cv.rmse), 4)      AS mean_rmse,
    ROUND(AVG(cv.mape), 4)      AS mean_mape,
    ROUND(AVG(cv.mase), 4)      AS mean_mase,
    ROUND(AVG(cv.coverage_80), 3) AS mean_coverage_80,
    ROUND(AVG(cv.coverage_95), 3) AS mean_coverage_95,
    RANK() OVER (
        PARTITION BY r.modern_region_code, cv.treatment_id, cv.part_id
        ORDER BY AVG(cv.mae) ASC
    )                           AS mae_rank
FROM nhs.fact_model_cv_results  cv
JOIN nhs.dim_region              r  ON cv.region_id    = r.region_id
JOIN nhs.dim_treatment           t  ON cv.treatment_id = t.treatment_id
JOIN nhs.dim_model               dm ON cv.model_id     = dm.model_id
GROUP BY
    r.modern_region_code, r.modern_region_name,
    t.function_code, t.function_name,
    dm.model_name, dm.model_type,
    cv.treatment_id, cv.part_id
ORDER BY r.modern_region_code, mean_mae ASC;

COMMENT ON VIEW nhs.v_model_leaderboard IS 'Model accuracy leaderboard ranked by mean MAE within each region';

-- ── v_forecast_with_actuals ───────────────────────────────────────────────────
CREATE OR REPLACE VIEW nhs.v_forecast_with_actuals AS
SELECT
    fc.forecast_period_date         AS period_date,
    r.modern_region_code            AS region_code,
    r.modern_region_name            AS region_name,
    t.function_code                 AS treatment_code,
    p.part_type,
    dm.model_name,
    fc.scenario,
    fc.horizon_months,
    fc.predicted_value,
    fc.lower_80,
    fc.upper_80,
    fc.lower_95,
    fc.upper_95,
    fc.mae_cv,
    fc.training_end_date,
    act.waiting_list_size           AS actual_value,
    act.pct_within_18wks            AS actual_pct_18wks,
    CASE
        WHEN act.waiting_list_size IS NULL                              THEN 'future'
        WHEN act.waiting_list_size BETWEEN fc.lower_80 AND fc.upper_80 THEN 'within_80'
        WHEN act.waiting_list_size BETWEEN fc.lower_95 AND fc.upper_95 THEN 'within_95'
        ELSE 'outside_95'
    END                             AS interval_status
FROM nhs.fact_forecast_outputs     fc
JOIN nhs.dim_region                 r   ON fc.region_id    = r.region_id
JOIN nhs.dim_treatment              t   ON fc.treatment_id = t.treatment_id
JOIN nhs.dim_rtt_part               p   ON fc.part_id      = p.part_id
JOIN nhs.dim_model                  dm  ON fc.model_id     = dm.model_id
LEFT JOIN nhs.v_rtt_regional_monthly act
    ON  act.region_code    = r.modern_region_code
    AND act.treatment_code = t.function_code
    AND act.part_type      = p.part_type
    AND act.period_date    = fc.forecast_period_date
ORDER BY fc.forecast_period_date, r.modern_region_code, dm.model_name;

COMMENT ON VIEW nhs.v_forecast_with_actuals IS 'Forecast vs actual — for Power BI fan charts and interval calibration';
