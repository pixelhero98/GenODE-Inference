#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

inputs_job=$(sbatch --parsable "${ROOT_DIR}/01_generate_inputs.sbatch")
echo "Submitted input generation: ${inputs_job}"

train_job=$(sbatch --parsable --dependency=afterok:"${inputs_job}" "${ROOT_DIR}/02_train_gap_policies.sbatch")
echo "Submitted gap policy training: ${train_job}"

report_job=$(sbatch --parsable --dependency=afterok:"${train_job}" "${ROOT_DIR}/03_report_unseen_calibration.sbatch")
echo "Submitted calibration unseen reports: ${report_job}"

collect_job=$(sbatch --parsable --dependency=afterok:"${report_job}" "${ROOT_DIR}/04_collect_gap_summary.sbatch")
echo "Submitted summary collection: ${collect_job}"
