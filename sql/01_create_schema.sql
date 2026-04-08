-- =============================================================================
-- 01_create_schema.sql
-- NHS Waiting Time Forecasting — Create Schema
-- Francis Kwesi Acquah | B01821156 | UWS
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS nhs;

COMMENT ON SCHEMA nhs IS
    'NHS Waiting Time Forecasting — Francis Kwesi Acquah B01821156 UWS';

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER DATABASE nhs_waiting_times SET search_path TO nhs, public;
