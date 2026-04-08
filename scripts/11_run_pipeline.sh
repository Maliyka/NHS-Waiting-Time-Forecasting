#!/bin/bash
# =============================================================================
# 11_run_pipeline.sh — Run the full NHS Forecasting pipeline end-to-end.
# Corrected: uses python3 / venv python, proper error detection.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'

log()     { echo -e "${BOLD}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"; }
success() { echo -e "${GREEN}  ✓ $*${NC}" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}  ✗ $*${NC}" | tee -a "$LOG_FILE"; }

PIPELINE_START=$(date +%s)

echo "" | tee -a "$LOG_FILE"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  NHS Waiting Time Forecasting — Full Pipeline${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  Started: $(date)${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  Log: $LOG_FILE${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ── Activate venv (look in project dir and common locations) ──────────────────
VENV_FOUND=false
for venv_path in \
    "$PROJECT_DIR/venv" \
    "$HOME/nhs_waiting_times_project/venv" \
    "$HOME/nhs_venv"
do
    if [ -f "$venv_path/bin/activate" ]; then
        source "$venv_path/bin/activate"
        log "Virtual environment activated: $(python3 --version) from $venv_path"
        VENV_FOUND=true
        break
    fi
done

if [ "$VENV_FOUND" = false ]; then
    log "WARNING: No venv found — checking for python3 in PATH"
fi

# ── Resolve python command ────────────────────────────────────────────────────
if command -v python &>/dev/null; then
    PYTHON_BIN="python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo -e "${RED}ERROR: python / python3 not found in PATH${NC}" | tee -a "$LOG_FILE"
    echo -e "${RED}Run: chmod +x fix_python_and_venv.sh && ./fix_python_and_venv.sh${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

log "Using Python: $PYTHON_BIN ($(${PYTHON_BIN} --version))"
log "PYTHONPATH: $PROJECT_DIR/scripts"
export PYTHONPATH="$PROJECT_DIR/scripts:${PYTHONPATH:-}"

cd "$PROJECT_DIR"

# ── Step runner ───────────────────────────────────────────────────────────────
run_step() {
    local step_num="$1"
    local step_name="$2"
    local script="$3"

    log ""
    log "STEP $step_num — $step_name"
    STEP_START=$(date +%s)

    if "$PYTHON_BIN" "$script" 2>&1 | tee -a "$LOG_FILE"; then
        STEP_END=$(date +%s)
        ELAPSED=$((STEP_END - STEP_START))
        success "Step $step_num complete in ${ELAPSED}s"
    else
        STEP_END=$(date +%s)
        ELAPSED=$((STEP_END - STEP_START))
        error "Step $step_num FAILED after ${ELAPSED}s"
        error "Check log: $LOG_FILE"
        exit 1
    fi
}

# ── Run all 10 steps ──────────────────────────────────────────────────────────
run_step  1 "Setup Database Schema"         "scripts/01_setup_database.py"
run_step  2 "Ingest RTT CSV Files"          "scripts/02_ingest_rtt_csv.py"
run_step  3 "Ingest New Periods XLS Files"  "scripts/03_ingest_new_periods_xls.py"
run_step  4 "Data Cleaning"                 "scripts/04_clean_harmonise.py"
run_step  5 "Exploratory Data Analysis"     "scripts/05_eda.py"
run_step  6 "ARIMA / SARIMA Modelling"      "scripts/06_model_arima_sarima.py"
run_step  7 "Holt-Winters Modelling"        "scripts/07_model_holt_winters.py"
run_step  8 "Prophet Modelling"             "scripts/08_model_prophet.py"
run_step  9 "Model Evaluation"              "scripts/09_model_evaluation.py"
run_step 10 "Generate Forecasts"            "scripts/10_generate_forecasts.py"

# ── Final summary ─────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL=$((PIPELINE_END - PIPELINE_START))
MINS=$((TOTAL / 60))
SECS=$((TOTAL % 60))

echo "" | tee -a "$LOG_FILE"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  PIPELINE COMPLETE — ${MINS}m ${SECS}s${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  Log:       $LOG_FILE${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  Charts:    $PROJECT_DIR/data/processed/eda_plots/${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}  Forecasts: $PROJECT_DIR/data/forecasts/${NC}" | tee -a "$LOG_FILE"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
