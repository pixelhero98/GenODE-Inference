#!/usr/bin/env bash
set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
GENODE_GPU_PARTITION=${GENODE_GPU_PARTITION:-hopper}
SCRIPT_DIR="${GENODE_REPO}/scripts/verification_ess_bins_20260602"

mkdir -p \
  "${GENODE_VERIFICATION_ROOT}/calibration_inputs" \
  "${GENODE_VERIFICATION_ROOT}/policy_runs" \
  "${GENODE_VERIFICATION_ROOT}/locked_test_reports" \
  "${GENODE_VERIFICATION_ROOT}/summary" \
  "${GENODE_VERIFICATION_ROOT}/jobs" \
  "${GENODE_VERIFICATION_ROOT}/logs"

cp "${SCRIPT_DIR}/01_calibration.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/02_train_policy.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/03_locked_report.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/04_collect_summary.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/collect_summary.py" "${GENODE_VERIFICATION_ROOT}/jobs/"

cal_job=$(sbatch --parsable --partition="${GENODE_GPU_PARTITION}" "${GENODE_VERIFICATION_ROOT}/jobs/01_calibration.sbatch")

declare -a run_ids=(
  temp_fixed_005_b128
  temp_fixed_0075_b128
  temp_fixed_010_b128
  temp_fixed_015_b128
  temp_ess20_b128
  temp_ess25_b128
  temp_ess30_b128
  bin_ess25_b64
  bin_ess25_b256
)
declare -A mode=(
  [temp_fixed_005_b128]=fixed
  [temp_fixed_0075_b128]=fixed
  [temp_fixed_010_b128]=fixed
  [temp_fixed_015_b128]=fixed
  [temp_ess20_b128]=adaptive_ess
  [temp_ess25_b128]=adaptive_ess
  [temp_ess30_b128]=adaptive_ess
  [bin_ess25_b64]=adaptive_ess
  [bin_ess25_b256]=adaptive_ess
)
declare -A temperature=(
  [temp_fixed_005_b128]=0.05
  [temp_fixed_0075_b128]=0.075
  [temp_fixed_010_b128]=0.10
  [temp_fixed_015_b128]=0.15
  [temp_ess20_b128]=0.05
  [temp_ess25_b128]=0.05
  [temp_ess30_b128]=0.05
  [bin_ess25_b64]=0.05
  [bin_ess25_b256]=0.05
)
declare -A target_ess=(
  [temp_fixed_005_b128]=2.5
  [temp_fixed_0075_b128]=2.5
  [temp_fixed_010_b128]=2.5
  [temp_fixed_015_b128]=2.5
  [temp_ess20_b128]=2.0
  [temp_ess25_b128]=2.5
  [temp_ess30_b128]=3.0
  [bin_ess25_b64]=2.5
  [bin_ess25_b256]=2.5
)
declare -A bins=(
  [temp_fixed_005_b128]=128
  [temp_fixed_0075_b128]=128
  [temp_fixed_010_b128]=128
  [temp_fixed_015_b128]=128
  [temp_ess20_b128]=128
  [temp_ess25_b128]=128
  [temp_ess30_b128]=128
  [bin_ess25_b64]=64
  [bin_ess25_b256]=256
)

declare -A train_jobs=()
declare -A report_jobs=()
declare -a report_job_list=()

for run_id in "${run_ids[@]}"; do
  train_job=$(
    sbatch --parsable \
      --partition="${GENODE_GPU_PARTITION}" \
      --dependency=afterok:${cal_job} \
      --export=ALL,GENODE_RUN_ID="${run_id}",GENODE_TEMPERATURE_MODE="${mode[${run_id}]}",GENODE_TEACHER_TEMPERATURE="${temperature[${run_id}]}",GENODE_TEACHER_TARGET_ESS="${target_ess[${run_id}]}",GENODE_DENSITY_BIN_COUNT="${bins[${run_id}]}" \
      "${GENODE_VERIFICATION_ROOT}/jobs/02_train_policy.sbatch"
  )
  report_job=$(
    sbatch --parsable \
      --partition="${GENODE_GPU_PARTITION}" \
      --dependency=afterok:${train_job} \
      --export=ALL,GENODE_RUN_ID="${run_id}" \
      "${GENODE_VERIFICATION_ROOT}/jobs/03_locked_report.sbatch"
  )
  train_jobs[${run_id}]="${train_job}"
  report_jobs[${run_id}]="${report_job}"
  report_job_list+=("${report_job}")
done

dependency="afterany:$(IFS=:; echo "${report_job_list[*]}")"
summary_job=$(sbatch --parsable --dependency="${dependency}" "${GENODE_VERIFICATION_ROOT}/jobs/04_collect_summary.sbatch")

train_payload=""
report_payload=""
for run_id in "${run_ids[@]}"; do
  train_payload+="${run_id}=${train_jobs[${run_id}]},"
  report_payload+="${run_id}=${report_jobs[${run_id}]},"
done
export GENODE_SUBMITTED_TRAIN_JOBS="${train_payload%,}"
export GENODE_SUBMITTED_REPORT_JOBS="${report_payload%,}"
export GENODE_GPU_PARTITION

python3 - "${GENODE_VERIFICATION_ROOT}" "${cal_job}" "${summary_job}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "calibration_job": sys.argv[2],
    "summary_job": sys.argv[3],
    "gpu_partition": os.environ.get("GENODE_GPU_PARTITION", "hopper"),
    "train_jobs": {},
    "report_jobs": {},
}
for item in os.environ["GENODE_SUBMITTED_TRAIN_JOBS"].split(","):
    if not item:
        continue
    run_id, job_id = item.split("=", 1)
    payload["train_jobs"][run_id] = job_id
for item in os.environ["GENODE_SUBMITTED_REPORT_JOBS"].split(","):
    if not item:
        continue
    run_id, job_id = item.split("=", 1)
    payload["report_jobs"][run_id] = job_id
(root / "summary" / "submitted_jobs.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
