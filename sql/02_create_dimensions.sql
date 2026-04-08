-- =============================================================================
-- 02_create_dimensions.sql
-- NHS Waiting Time Forecasting — Dimension Tables
-- Francis Kwesi Acquah | B01821156 | UWS
-- =============================================================================

SET search_path TO nhs, public;

-- ── dim_region ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.dim_region (
    region_id          SERIAL       PRIMARY KEY,
    region_code        VARCHAR(10)  NOT NULL,
    region_name        TEXT         NOT NULL,
    region_short       VARCHAR(60),
    modern_region_code VARCHAR(10),
    modern_region_name TEXT,
    is_modern_region   BOOLEAN      DEFAULT FALSE,
    created_at         TIMESTAMP    DEFAULT NOW(),
    CONSTRAINT uq_region_code UNIQUE (region_code)
);
COMMENT ON TABLE  nhs.dim_region IS 'NHS England region codes — 2019 sub-regional and modern 7-region';
COMMENT ON COLUMN nhs.dim_region.modern_region_code IS 'Maps all legacy codes to one of 7 modern region codes for consistent time series';

-- ── dim_calendar ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.dim_calendar (
    calendar_id         SERIAL      PRIMARY KEY,
    period_date         DATE        NOT NULL,
    year                INTEGER     NOT NULL,
    month               INTEGER     NOT NULL CHECK (month BETWEEN 1 AND 12),
    quarter             INTEGER     NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month_name          VARCHAR(20) NOT NULL,
    year_month          VARCHAR(7)  NOT NULL,
    is_pre_covid        BOOLEAN     NOT NULL DEFAULT FALSE,
    is_covid_period     BOOLEAN     NOT NULL DEFAULT FALSE,
    is_post_covid       BOOLEAN     NOT NULL DEFAULT FALSE,
    covid_dummy         SMALLINT    NOT NULL DEFAULT 0 CHECK (covid_dummy IN (0,1)),
    bank_holiday_count  INTEGER     NOT NULL DEFAULT 0,
    created_at          TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_period_date UNIQUE (period_date)
);
COMMENT ON TABLE  nhs.dim_calendar IS 'Monthly calendar 2019-2026 with COVID era flags';
COMMENT ON COLUMN nhs.dim_calendar.covid_dummy IS '1 = Apr 2020 to Mar 2022 inclusive (exogenous regressor for ARIMA)';

-- ── dim_treatment ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.dim_treatment (
    treatment_id    SERIAL      PRIMARY KEY,
    function_code   VARCHAR(20) NOT NULL,
    function_name   TEXT        NOT NULL,
    specialty_group VARCHAR(50),
    is_total        BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_function_code UNIQUE (function_code)
);
COMMENT ON TABLE  nhs.dim_treatment IS 'RTT treatment function (specialty) codes from CSV column 12';
COMMENT ON COLUMN nhs.dim_treatment.is_total IS 'TRUE only for C_999 Total — used to exclude from specialty-level queries';

-- ── dim_rtt_part ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.dim_rtt_part (
    part_id          SERIAL      PRIMARY KEY,
    part_type        VARCHAR(20) NOT NULL,
    part_description TEXT        NOT NULL,
    is_primary       BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_part_type UNIQUE (part_type)
);
COMMENT ON TABLE  nhs.dim_rtt_part IS 'RTT pathway type — Part_2 Incomplete Pathways is the primary forecasting target';
COMMENT ON COLUMN nhs.dim_rtt_part.is_primary IS 'TRUE for Part_2 only — the waiting-list snapshot used for all forecasting';

-- ── dim_model ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nhs.dim_model (
    model_id     SERIAL      PRIMARY KEY,
    model_name   VARCHAR(60) NOT NULL,
    model_type   VARCHAR(30) NOT NULL,
    description  TEXT,
    created_at   TIMESTAMP   DEFAULT NOW(),
    CONSTRAINT uq_model_name UNIQUE (model_name)
);
COMMENT ON TABLE nhs.dim_model IS 'Forecasting model registry — ARIMA, SARIMA, Holt-Winters, Prophet';
