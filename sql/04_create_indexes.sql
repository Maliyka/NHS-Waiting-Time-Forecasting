-- =============================================================================
-- 04_create_indexes.sql
-- NHS Waiting Time Forecasting — Performance Indexes
-- Francis Kwesi Acquah | B01821156 | UWS
-- =============================================================================

SET search_path TO nhs, public;

-- fact_rtt_monthly
CREATE INDEX IF NOT EXISTS idx_rtt_region_part_treatment
    ON nhs.fact_rtt_monthly (region_id, part_id, treatment_id, calendar_id);

CREATE INDEX IF NOT EXISTS idx_rtt_calendar
    ON nhs.fact_rtt_monthly (calendar_id);

CREATE INDEX IF NOT EXISTS idx_rtt_provider
    ON nhs.fact_rtt_monthly (provider_org_code, calendar_id);

CREATE INDEX IF NOT EXISTS idx_rtt_commissioner
    ON nhs.fact_rtt_monthly (commissioner_org_code, calendar_id);

CREATE INDEX IF NOT EXISTS idx_rtt_18wk
    ON nhs.fact_rtt_monthly (region_id, calendar_id, pct_within_18wks);

CREATE INDEX IF NOT EXISTS idx_rtt_52wk
    ON nhs.fact_rtt_monthly (region_id, calendar_id, wk_52_plus)
    WHERE wk_52_plus IS NOT NULL;

-- fact_new_periods_monthly
CREATE INDEX IF NOT EXISTS idx_np_region_treatment
    ON nhs.fact_new_periods_monthly (region_id, treatment_id, calendar_id);

CREATE INDEX IF NOT EXISTS idx_np_calendar
    ON nhs.fact_new_periods_monthly (calendar_id);

-- fact_model_cv_results
CREATE INDEX IF NOT EXISTS idx_cv_region_model
    ON nhs.fact_model_cv_results (region_id, model_id);

CREATE INDEX IF NOT EXISTS idx_cv_treatment_part
    ON nhs.fact_model_cv_results (treatment_id, part_id);

CREATE INDEX IF NOT EXISTS idx_cv_covid_era
    ON nhs.fact_model_cv_results (covid_era, region_id, model_id);

-- fact_forecast_outputs
CREATE INDEX IF NOT EXISTS idx_fc_region_model_date
    ON nhs.fact_forecast_outputs (region_id, model_id, forecast_period_date);

CREATE INDEX IF NOT EXISTS idx_fc_scenario
    ON nhs.fact_forecast_outputs (scenario, region_id, forecast_period_date);

-- dim_calendar
CREATE INDEX IF NOT EXISTS idx_cal_year_month
    ON nhs.dim_calendar (year, month);

CREATE INDEX IF NOT EXISTS idx_cal_covid
    ON nhs.dim_calendar (is_covid_period);

-- dim_region
CREATE INDEX IF NOT EXISTS idx_region_modern_code
    ON nhs.dim_region (modern_region_code);
