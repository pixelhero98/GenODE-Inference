#!/usr/bin/env bash
set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603}
GENODE_GPU_PARTITION=${GENODE_GPU_PARTITION:-ampere}
GENODE_CPU_PARTITION=${GENODE_CPU_PARTITION:-genoa}
SCRIPT_DIR="${GENODE_REPO}/scripts/verification_density_diversity_20260603"

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
cp "${SCRIPT_DIR}/05_submit_unseen_winner.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/collect_density_diversity_summary.py" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/collect_unseen_density_diversity_summary.py" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/merge_density_diversity_inputs.py" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/submit_unseen_winner.sh" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/unseen_nfe_report_policy.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"
cp "${SCRIPT_DIR}/unseen_nfe_collect_summary.sbatch" "${GENODE_VERIFICATION_ROOT}/jobs/"

cal_job=$(sbatch --parsable --partition="${GENODE_GPU_PARTITION}" "${GENODE_VERIFICATION_ROOT}/jobs/01_calibration.sbatch")

declare -a run_ids=(
  diverse_fixed005_b128
  diverse_ess25_b128
  diverse_ess40_b128
)
declare -A mode=(
  [diverse_fixed005_b128]=fixed
  [diverse_ess25_b128]=adaptive_ess
  [diverse_ess40_b128]=adaptive_ess
)
declare -A temperature=(
  [diverse_fixed005_b128]=0.05
  [diverse_ess25_b128]=0.05
  [diverse_ess40_b128]=0.05
)
declare -A target_ess=(
  [diverse_fixed005_b128]=2.5
  [diverse_ess25_b128]=2.5
  [diverse_ess40_b128]=4.0
)

declare -A train_jobs=()
declare -A report_jobs=()
declare -a report_job_list=()

for run_id in "${run_ids[@]}"; do
  train_job=$(
    sbatch --parsable \
      --partition="${GENODE_GPU_PARTITION}" \
      --dependency=afterok:${cal_job} \
      --export=ALL,GENODE_RUN_ID="${run_id}",GENODE_TEMPERATURE_MODE="${mode[${run_id}]}",GENODE_TEACHER_TEMPERATURE="${temperature[${run_id}]}",GENODE_TEACHER_TARGET_ESS="${target_ess[${run_id}]}",GENODE_DENSITY_BIN_COUNT=128 \
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
summary_job=$(sbatch --parsable --partition="${GENODE_CPU_PARTITION}" --dependency="${dependency}" "${GENODE_VERIFICATION_ROOT}/jobs/04_collect_summary.sbatch")
unseen_submit_job=$(
  sbatch --parsable \
    --partition="${GENODE_CPU_PARTITION}" \
    --dependency=afterok:${summary_job} \
    --export=ALL,GENODE_VERIFICATION_ROOT="${GENODE_VERIFICATION_ROOT}",GENODE_GPU_PARTITION="${GENODE_GPU_PARTITION}",GENODE_CPU_PARTITION="${GENODE_CPU_PARTITION}" \
    "${GENODE_VERIFICATION_ROOT}/jobs/05_submit_unseen_winner.sbatch"
)

train_payload=""
report_payload=""
for run_id in "${run_ids[@]}"; do
  train_payload+="${run_id}=${train_jobs[${run_id}]},"
  report_payload+="${run_id}=${report_jobs[${run_id}]},"
done
export GENODE_SUBMITTED_TRAIN_JOBS="${train_payload%,}"
export GENODE_SUBMITTED_REPORT_JOBS="${report_payload%,}"
export GENODE_GPU_PARTITION
export GENODE_CPU_PARTITION

python3 - "${GENODE_VERIFICATION_ROOT}" "${cal_job}" "${summary_job}" "${unseen_submit_job}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "calibration_job": sys.argv[2],
    "summary_job": sys.argv[3],
    "unseen_submit_job": sys.argv[4],
    "gpu_partition": os.environ.get("GENODE_GPU_PARTITION", "ampere"),
    "cpu_partition": os.environ.get("GENODE_CPU_PARTITION", "genoa"),
    "run_ids": ["diverse_fixed005_b128", "diverse_ess25_b128", "diverse_ess40_b128"],
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
