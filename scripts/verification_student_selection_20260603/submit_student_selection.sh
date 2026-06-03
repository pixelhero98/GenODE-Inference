#!/usr/bin/env bash
set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_STUDENT_SELECTION_ROOT=${GENODE_STUDENT_SELECTION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603}
GENODE_CPU_PARTITION=${GENODE_CPU_PARTITION:-genoa}
GENODE_GPU_PARTITION=${GENODE_GPU_PARTITION:-ampere}
SCRIPT_DIR="${GENODE_REPO}/scripts/verification_student_selection_20260603"

mkdir -p \
  "${GENODE_STUDENT_SELECTION_ROOT}/jobs" \
  "${GENODE_STUDENT_SELECTION_ROOT}/logs" \
  "${GENODE_STUDENT_SELECTION_ROOT}/summary" \
  "${GENODE_STUDENT_SELECTION_ROOT}/calibration_student_reports"

cp "${SCRIPT_DIR}/01_prepare_inputs.sbatch" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/02_run_student_calibration_reports.sbatch" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/03_collect_selection.sbatch" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/04_locked_selected_report.sbatch" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/build_student_selection_inputs.py" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/run_student_calibration_reports.py" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/collect_student_selection_summary.py" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/run_locked_selected_report.py" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/submit_student_selection.sh" "${GENODE_STUDENT_SELECTION_ROOT}/jobs/"

prep_job=$(
  sbatch --parsable \
    --partition="${GENODE_CPU_PARTITION}" \
    --export=ALL,GENODE_STUDENT_SELECTION_ROOT="${GENODE_STUDENT_SELECTION_ROOT}" \
    "${GENODE_STUDENT_SELECTION_ROOT}/jobs/01_prepare_inputs.sbatch"
)
cal_job=$(
  sbatch --parsable \
    --partition="${GENODE_GPU_PARTITION}" \
    --dependency=afterok:${prep_job} \
    --export=ALL,GENODE_STUDENT_SELECTION_ROOT="${GENODE_STUDENT_SELECTION_ROOT}" \
    "${GENODE_STUDENT_SELECTION_ROOT}/jobs/02_run_student_calibration_reports.sbatch"
)
summary_job=$(
  sbatch --parsable \
    --partition="${GENODE_CPU_PARTITION}" \
    --dependency=afterok:${cal_job} \
    --export=ALL,GENODE_STUDENT_SELECTION_ROOT="${GENODE_STUDENT_SELECTION_ROOT}" \
    "${GENODE_STUDENT_SELECTION_ROOT}/jobs/03_collect_selection.sbatch"
)
locked_job=$(
  sbatch --parsable \
    --partition="${GENODE_GPU_PARTITION}" \
    --dependency=afterok:${summary_job} \
    --export=ALL,GENODE_STUDENT_SELECTION_ROOT="${GENODE_STUDENT_SELECTION_ROOT}" \
    "${GENODE_STUDENT_SELECTION_ROOT}/jobs/04_locked_selected_report.sbatch"
)

export GENODE_CPU_PARTITION GENODE_GPU_PARTITION
export GENODE_STUDENT_SELECTION_PREP_JOB="${prep_job}"
export GENODE_STUDENT_SELECTION_CAL_JOB="${cal_job}"
export GENODE_STUDENT_SELECTION_SUMMARY_JOB="${summary_job}"
export GENODE_STUDENT_SELECTION_LOCKED_JOB="${locked_job}"

python3 - "${GENODE_STUDENT_SELECTION_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "artifact": "genode_student_selection_submitted_jobs",
    "cpu_partition": os.environ["GENODE_CPU_PARTITION"],
    "gpu_partition": os.environ["GENODE_GPU_PARTITION"],
    "prep_job": os.environ["GENODE_STUDENT_SELECTION_PREP_JOB"],
    "calibration_report_job": os.environ["GENODE_STUDENT_SELECTION_CAL_JOB"],
    "summary_job": os.environ["GENODE_STUDENT_SELECTION_SUMMARY_JOB"],
    "locked_selected_job": os.environ["GENODE_STUDENT_SELECTION_LOCKED_JOB"],
    "locked_test_used_for_selection": False,
}
(root / "summary" / "submitted_jobs.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
