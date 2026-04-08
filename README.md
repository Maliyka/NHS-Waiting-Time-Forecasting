# NHS Waiting Time Forecasting Project

**Title:** A Regional Data Analysis Approach for NHS Waiting Time Forecasting  
**Student:** Francis Kwesi Acquah | B01821156  
**University:** University of the West of Scotland  
**Supervisor:** Edward Jennings  

---

## Overview

This project forecasts NHS England waiting times by region using ARIMA/SARIMA,
Holt-Winters (ETS), and Facebook Prophet. Data covers March 2019 – December 2025
(monthly). PostgreSQL stores all data in a star schema; Python handles ingestion,
modelling, and evaluation; Power BI provides the interactive dashboard.

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd ~/nhs_waiting_times_project

# 2. Run the full setup (installs PostgreSQL, Python packages, creates DB)
chmod +x step1_install_dependencies.sh && ./step1_install_dependencies.sh
chmod +x step2_create_database.sh      && ./step2_create_database.sh

# 3. Place your data files
#    RTT CSV files   → data/raw/rtt_csv/
#    New Periods XLS → data/raw/new_periods_xls/

# 4. Activate environment and run the full pipeline
source activate.sh
bash scripts/11_run_pipeline.sh
```

---

## Project Structure

```
nhs_waiting_times_project/
├── README.md
├── requirements.txt
├── .env                          ← DB credentials (do not commit)
├── .env.example
├── activate.sh                   ← source this each session
│
├── config/
│   └── config.yaml               ← all settings live here
│
├── sql/
│   ├── 01_create_schema.sql
│   ├── 02_create_dimensions.sql
│   ├── 03_create_facts.sql
│   ├── 04_create_indexes.sql
│   └── 05_create_views.sql
│
├── scripts/
│   ├── utils/
│   │   ├── db_connect.py         ← SQLAlchemy engine + helpers
│   │   ├── data_helpers.py       ← parsing, cleaning, column mapping
│   │   └── metrics.py            ← MAE, RMSE, MAPE, MASE
│   ├── 01_setup_database.py      ← run SQL files + seed dims
│   ├── 02_ingest_rtt_csv.py      ← load all RTT CSV files
│   ├── 03_ingest_new_periods_xls.py
│   ├── 04_clean_harmonise.py
│   ├── 05_eda.py
│   ├── 06_model_arima_sarima.py
│   ├── 07_model_holt_winters.py
│   ├── 08_model_prophet.py
│   ├── 09_model_evaluation.py
│   ├── 10_generate_forecasts.py
│   └── 11_run_pipeline.sh
│
├── data/
│   ├── raw/
│   │   ├── rtt_csv/              ← place RTT-MONTH-YEAR-full-extract.csv here
│   │   └── new_periods_xls/      ← place New-Periods-Commissioner-*.xls here
│   ├── processed/
│   │   ├── eda_plots/
│   │   ├── model_plots/
│   │   └── model_params/
│   └── forecasts/
│       └── plots/
│
├── logs/                         ← pipeline logs written here
└── notebooks/
    ├── 01_EDA_Exploration.ipynb
    ├── 02_Model_Training.ipynb
    └── 03_Results_Interpretation.ipynb
```

---

## Data Files Required

### RTT CSV Files
- **Location:** `data/raw/rtt_csv/`
- **Filename pattern:** `RTT-MARCH-2019-full-extract.csv`
- **Source:** NHS England RTT Waiting Times statistical work area
- **One file per month** from March 2019 to December 2025 (~75 files)
- ZIP files are automatically extracted

### New Periods XLS Files
- **Location:** `data/raw/new_periods_xls/`
- **Filename pattern:** `New-Periods-Commissioner-Mar19-revised-XLS-553K.xls`
- **Source:** NHS England New Periods Commissioner data
- **One file per month** from March 2019 to December 2025 (~75 files)

---

## Database Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update && sudo apt install -y postgresql postgresql-contrib libpq-dev

# Start service
sudo systemctl start postgresql && sudo systemctl enable postgresql

# Create user and database
sudo -u postgres psql << 'EOF'
CREATE USER nhs_user WITH PASSWORD 'nhs_secure_pass' CREATEDB LOGIN;
CREATE DATABASE nhs_waiting_times OWNER nhs_user ENCODING 'UTF8' TEMPLATE template0;
GRANT ALL PRIVILEGES ON DATABASE nhs_waiting_times TO nhs_user;
\c nhs_waiting_times
GRANT ALL ON SCHEMA public TO nhs_user;
EOF

# Build schema (done automatically by step2_create_database.sh or 01_setup_database.py)
source activate.sh
python scripts/01_setup_database.py
```

---

## Running the Pipeline

```bash
source activate.sh

# Individual scripts
python scripts/01_setup_database.py          # build schema + seed dims
python scripts/02_ingest_rtt_csv.py          # ~15M rows, takes 30-60 mins
python scripts/03_ingest_new_periods_xls.py  # ~75K rows
python scripts/04_clean_harmonise.py         # quality checks + outlier flags
python scripts/05_eda.py                     # 10 charts + summary stats
python scripts/06_model_arima_sarima.py      # ARIMA/SARIMA CV
python scripts/07_model_holt_winters.py      # Holt-Winters CV
python scripts/08_model_prophet.py           # Prophet CV
python scripts/09_model_evaluation.py        # leaderboard + comparison charts
python scripts/10_generate_forecasts.py      # 12-month forecasts × 3 scenarios

# Or run everything at once
bash scripts/11_run_pipeline.sh
```

---

## Power BI Dashboard

1. Install Npgsql PostgreSQL connector on Windows
2. Open Power BI Desktop → Get Data → PostgreSQL
3. Server: `localhost` (or your Linux machine IP), Database: `nhs_waiting_times`
4. Connect to views: `nhs.v_rtt_regional_monthly`, `nhs.v_model_leaderboard`
5. Import: `data/forecasts/forecasts_all_regions_<date>.csv`

If PostgreSQL is on Linux and Power BI on Windows, edit `/etc/postgresql/*/main/postgresql.conf`:
```
listen_addresses = '*'
```
Add to `pg_hba.conf`:
```
host  nhs_waiting_times  nhs_user  0.0.0.0/0  md5
```
Then: `sudo systemctl restart postgresql && sudo ufw allow 5432`

---

## Key Settings (config/config.yaml)

| Setting | Value | Notes |
|---------|-------|-------|
| Primary metric | `Part_2` | Incomplete Pathways (waiting list snapshot) |
| Primary treatment | `C_999` | Total across all specialties |
| 18-week target | 92% | NHS RTT standard |
| Forecast horizon | 12 months | Ahead from last training month |
| CV folds | 6 | Rolling-origin cross-validation |
| COVID start | 2020-04-01 | Elective care suspended |
| COVID end | 2022-04-01 | Recovery phase complete |

---

## NHS Region Codes

| Code | Region | Notes |
|------|--------|-------|
| Q71 | NHS England London | Unchanged across years |
| Y60 | North East and Yorkshire | Modern (2021+) |
| Y61 | Midlands | Modern (2021+) |
| Y62 | East of England | Modern (2021+) |
| Y63 | North West | Modern (2021+) |
| Y58 | South East | Modern (2021+) |
| Y59 | South West | Modern (2021+) |

2019-era sub-regional codes (Q72, Q74–Q88) are automatically consolidated to the 7 modern regions during ingestion.
