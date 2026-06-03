#!/usr/bin/env bash
set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
GENODE_UNSEEN_ROOT=${GENODE_UNSEEN_ROOT:-${GENODE_VERIFICATION_ROOT}/unseen_nfe_locked_winner_6_10_14_16}
GENODE_GPU_PARTITION=${GENODE_GPU_PARTITION:-ampere}
GENODE_CPU_PARTITION=${GENODE_CPU_PARTITION:-genoa}
GENODE_POLICY_RUN_ID=${GENODE_POLICY_RUN_ID:-temp_fixed_005_b128}
SCRIPT_DIR="${GENODE_REPO}/scripts/verification_ess_bins_20260602"

mkdir -p \
  "${GENODE_UNSEEN_ROOT}/inputs" \
  "${GENODE_UNSEEN_ROOT}/report" \
  "${GENODE_UNSEEN_ROOT}/summary" \
  "${GENODE_UNSEEN_ROOT}/jobs" \
  "${GENODE_UNSEEN_ROOT}/logs"

cp "${SCRIPT_DIR}/unseen_nfe_generate_inputs.sbatch" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/unseen_nfe_report_policy.sbatch" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/unseen_nfe_collect_summary.sbatch" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/collect_unseen_nfe_summary.py" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/submit_unseen_nfe.sh" "${GENODE_UNSEEN_ROOT}/jobs/"

input_job=$(
  sbatch --parsable \
    --partition="${GENODE_GPU_PARTITION}" \
    --export=ALL,GENODE_POLICY_RUN_ID="${GENODE_POLICY_RUN_ID}",GENODE_UNSEEN_ROOT="${GENODE_UNSEEN_ROOT}" \
    "${GENODE_UNSEEN_ROOT}/jobs/unseen_nfe_generate_inputs.sbatch"
)
report_job=$(
  sbatch --parsable \
    --partition="${GENODE_GPU_PARTITION}" \
    --dependency=afterok:${input_job} \
    --export=ALL,GENODE_POLICY_RUN_ID="${GENODE_POLICY_RUN_ID}",GENODE_UNSEEN_ROOT="${GENODE_UNSEEN_ROOT}" \
    "${GENODE_UNSEEN_ROOT}/jobs/unseen_nfe_report_policy.sbatch"
)
summary_job=$(
  sbatch --parsable \
    --partition="${GENODE_CPU_PARTITION}" \
    --dependency=afterany:${report_job} \
    --export=ALL,GENODE_POLICY_RUN_ID="${GENODE_POLICY_RUN_ID}",GENODE_UNSEEN_ROOT="${GENODE_UNSEEN_ROOT}" \
    "${GENODE_UNSEEN_ROOT}/jobs/unseen_nfe_collect_summary.sbatch"
)

export GENODE_UNSEEN_INPUT_JOB="${input_job}"
export GENODE_UNSEEN_REPORT_JOB="${report_job}"
export GENODE_UNSEEN_SUMMARY_JOB="${summary_job}"
export GENODE_GPU_PARTITION
export GENODE_CPU_PARTITION
export GENODE_POLICY_RUN_ID

python3 - "${GENODE_UNSEEN_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "policy_run_id": os.environ["GENODE_POLICY_RUN_ID"],
    "gpu_partition": os.environ["GENODE_GPU_PARTITION"],
    "cpu_partition": os.environ["GENODE_CPU_PARTITION"],
    "input_job": os.environ["GENODE_UNSEEN_INPUT_JOB"],
    "report_job": os.environ["GENODE_UNSEEN_REPORT_JOB"],
    "summary_job": os.environ["GENODE_UNSEEN_SUMMARY_JOB"],
    "target_nfe_values": [6, 10, 14, 16],
    "resume_mode": "fresh_unseen_nfe_locked_test",
}
(root / "summary" / "submitted_jobs.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
