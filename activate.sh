#!/bin/bash
# Run this at the start of every terminal session:
#   source activate.sh

cd "/home/maliyka/Downloads/nhs_waiting_times_project/nhs_project"
source "/home/maliyka/Downloads/nhs_waiting_times_project/nhs_project/venv/bin/activate"
export PYTHONPATH="/home/maliyka/Downloads/nhs_waiting_times_project/nhs_project/scripts:$PYTHONPATH"
echo "NHS project environment activated."
echo "Python: $(python --version) at $(which python)"
echo "Project: /home/maliyka/Downloads/nhs_waiting_times_project/nhs_project"
