-- =============================================================================
-- 03_create_facts.sql
-- NHS Waiting Time Forecasting — Fact Tables
-- Francis Kwesi Acquah | B01821156 | UWS
-- =============================================================================

SET search_path TO nhs, public;

-- ── fact_rtt_monthly ──────────────────────────────────────────────────────────
-- Core fact table. Partitioned by calendar_id range (one partition per year).
-- All 53 weekly bucket columns match CSV columns 14-66 exactly.
CREATE TABLE IF NOT EXISTS nhs.fact_rtt_monthly (
    fact_id                  BIGSERIAL    PRIMARY KEY,
    calendar_id              INTEGER      NOT NULL REFERENCES nhs.dim_calendar(calendar_id),
    region_id                INTEGER      NOT NULL REFERENCES nhs.dim_region(region_id),
    treatment_id             INTEGER      NOT NULL REFERENCES nhs.dim_treatment(treatment_id),
    part_id                  INTEGER      NOT NULL REFERENCES nhs.dim_rtt_part(part_id),
    provider_org_code        VARCHAR(20),
    provider_org_name        TEXT,
    commissioner_org_code    VARCHAR(20),
    commissioner_org_name    TEXT,
    -- 53 weekly buckets. NULL = NHS suppressed value (*), NOT zero.
    wk_00_01  INTEGER, wk_01_02  INTEGER, wk_02_03  INTEGER, wk_03_04  INTEGER,
    wk_04_05  INTEGER, wk_05_06  INTEGER, wk_06_07  INTEGER, wk_07_08  INTEGER,
    wk_08_09  INTEGER, wk_09_10  INTEGER, wk_10_11  INTEGER, wk_11_12  INTEGER,
    wk_12_13  INTEGER, wk_13_14  INTEGER, wk_14_15  INTEGER, wk_15_16  INTEGER,
    wk_16_17  INTEGER, wk_17_18  INTEGER,
    wk_18_19  INTEGER, wk_19_20  INTEGER, wk_20_21  INTEGER, wk_21_22  INTEGER,
    wk_22_23  INTEGER, wk_23_24  INTEGER, wk_24_25  INTEGER, wk_25_26  INTEGER,
    wk_26_27  INTEGER, wk_27_28  INTEGER, wk_28_29  INTEGER, wk_29_30  INTEGER,
    wk_30_31  INTEGER, wk_31_32  INTEGER, wk_32_33  INTEGER, wk_33_34  INTEGER,
    wk_34_35  INTEGER, wk_35_36  INTEGER, wk_36_37  INTEGER, wk_37_38  INTEGER,
    wk_38_39  INTEGER, wk_39_40  INTEGER, wk_40_41  INTEGER, wk_41_42  INTEGER,
    wk_42_43  INTEGER, wk_43_44  INTEGER, wk_44_45  INTEGER, wk_45_46  INTEGER,
    wk_46_47  INTEGER, wk_47_48  INTEGER, wk_48_49  INTEGER, wk_49_50  INTEGER,
    wk_50_51  INTEGER, wk_51_52  INTEGER,
    wk_52_plus INTEGER,
    -- NHS published totals
    total_patients            INTEGER,
    patients_unknown_start    INTEGER,
    total_all                 INTEGER,
    -- Computed 18-week metrics (populated during ingestion)
    patients_within_18wks     INTEGER,
    patients_over_18wks       INTEGER,
    pct_within_18wks          NUMERIC(7,4),
    -- Provenance
    source_file               VARCHAR(300),
    loaded_at                 TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_rtt_row UNIQUE (
        calendar_id, region_id, treatment_id, part_id,
        provider_org_code, commissioner_org_code
    )
);

COMMENT ON TABLE  nhs.fact_rtt_monthly IS 'Core RTT fact table — one row per period × provider × commissioner × part × treatment';
COMMENT ON COLUMN nhs.fact_rtt_monthly.wk_52_plus IS 'Patients waiting > 52 weeks — NHS high-profile metric, target = 0';
COMMENT ON COLUMN nhs.fact_rtt_monthly.pct_within_18wks IS 'Percentage within 18-week target — NHS standard = 92%';

-- ── fact_new_periods_monthly ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.fact_new_periods_monthly (
    np_id                 BIGSERIAL   PRIMARY KEY,
    calendar_id           INTEGER     NOT NULL REFERENCES nhs.dim_calendar(calendar_id),
    region_id             INTEGER     NOT NULL REFERENCES nhs.dim_region(region_id),
    treatment_id          INTEGER     NOT NULL REFERENCES nhs.dim_treatment(treatment_id),
    commissioner_org_code VARCHAR(20),
    commissioner_org_name TEXT,
    new_periods_count     INTEGER,
    source_file           VARCHAR(300),
    loaded_at             TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_new_periods_row UNIQUE (
        calendar_id, region_id, treatment_id, commissioner_org_code
    )
);
COMMENT ON TABLE  nhs.fact_new_periods_monthly IS 'New RTT referral periods started each month — from Commissioner XLS files';
COMMENT ON COLUMN nhs.fact_new_periods_monthly.new_periods_count IS 'Count of new RTT clock starts — used as demand regressor in Prophet';

-- ── fact_model_cv_results ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.fact_model_cv_results (
    cv_id            BIGSERIAL   PRIMARY KEY,
    region_id        INTEGER     NOT NULL REFERENCES nhs.dim_region(region_id),
    treatment_id     INTEGER     NOT NULL REFERENCES nhs.dim_treatment(treatment_id),
    part_id          INTEGER     NOT NULL REFERENCES nhs.dim_rtt_part(part_id),
    model_id         INTEGER     NOT NULL REFERENCES nhs.dim_model(model_id),
    fold_number      INTEGER     NOT NULL,
    train_start_date DATE        NOT NULL,
    train_end_date   DATE        NOT NULL,
    test_start_date  DATE        NOT NULL,
    test_end_date    DATE        NOT NULL,
    horizon_months   INTEGER     NOT NULL,
    covid_era        VARCHAR(15),
    mae              NUMERIC(14,4),
    rmse             NUMERIC(14,4),
    mape             NUMERIC(10,4),
    mase             NUMERIC(10,4),
    coverage_80      NUMERIC(6,3),
    coverage_95      NUMERIC(6,3),
    model_params     JSONB,
    created_at       TIMESTAMP   DEFAULT NOW()
);
COMMENT ON TABLE nhs.fact_model_cv_results IS 'Rolling-origin CV results — one row per model × region × treatment × fold';

-- ── fact_forecast_outputs ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.fact_forecast_outputs (
    forecast_id          BIGSERIAL   PRIMARY KEY,
    region_id            INTEGER     NOT NULL REFERENCES nhs.dim_region(region_id),
    treatment_id         INTEGER     NOT NULL REFERENCES nhs.dim_treatment(treatment_id),
    part_id              INTEGER     NOT NULL REFERENCES nhs.dim_rtt_part(part_id),
    model_id             INTEGER     NOT NULL REFERENCES nhs.dim_model(model_id),
    forecast_period_date DATE        NOT NULL,
    horizon_months       INTEGER     NOT NULL,
    scenario             VARCHAR(30) NOT NULL DEFAULT 'base',
    predicted_value      NUMERIC(14,2),
    lower_80             NUMERIC(14,2),
    upper_80             NUMERIC(14,2),
    lower_95             NUMERIC(14,2),
    upper_95             NUMERIC(14,2),
    training_end_date    DATE,
    model_params         JSONB,
    mae_cv               NUMERIC(14,4),
    rmse_cv              NUMERIC(14,4),
    mape_cv              NUMERIC(10,4),
    created_at           TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_forecast_row UNIQUE (
        region_id, treatment_id, part_id, model_id,
        forecast_period_date, scenario
    )
);
COMMENT ON TABLE  nhs.fact_forecast_outputs IS 'Final model forecast outputs with prediction intervals and scenarios';
COMMENT ON COLUMN nhs.fact_forecast_outputs.scenario IS 'base | demand_plus10 | demand_minus10';
