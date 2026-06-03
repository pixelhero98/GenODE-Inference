#!/usr/bin/env bash
set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603}
GENODE_SOURCE_UNSEEN_ROOT=${GENODE_SOURCE_UNSEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/unseen_nfe_locked_winner_6_10_14_16}
GENODE_UNSEEN_ROOT=${GENODE_UNSEEN_ROOT:-${GENODE_VERIFICATION_ROOT}/unseen_nfe_locked_winner_6_10_14_16}
GENODE_GPU_PARTITION=${GENODE_GPU_PARTITION:-ampere}
GENODE_CPU_PARTITION=${GENODE_CPU_PARTITION:-genoa}
SCRIPT_DIR="${GENODE_REPO}/scripts/verification_density_diversity_20260603"

if [[ -z "${GENODE_POLICY_RUN_ID:-}" ]]; then
  GENODE_POLICY_RUN_ID=$(
    python3 - "${GENODE_VERIFICATION_ROOT}/summary/verification_matrix.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_id = payload.get("selected_calibration_run_id")
if not run_id:
    raise SystemExit("selected_calibration_run_id is missing")
print(run_id)
PY
  )
fi

mkdir -p \
  "${GENODE_UNSEEN_ROOT}/inputs" \
  "${GENODE_UNSEEN_ROOT}/report" \
  "${GENODE_UNSEEN_ROOT}/summary" \
  "${GENODE_UNSEEN_ROOT}/jobs" \
  "${GENODE_UNSEEN_ROOT}/logs"

cp -a "${GENODE_SOURCE_UNSEEN_ROOT}/inputs/fixed_locked_unseen" "${GENODE_UNSEEN_ROOT}/inputs/"
cp -a "${GENODE_SOURCE_UNSEEN_ROOT}/inputs/ser_locked_unseen" "${GENODE_UNSEEN_ROOT}/inputs/"
cp -a "${GENODE_SOURCE_UNSEEN_ROOT}/inputs/ser_reference_unseen" "${GENODE_UNSEEN_ROOT}/inputs/"
cp "${GENODE_SOURCE_UNSEEN_ROOT}/inputs/locked_context_rows.csv" "${GENODE_UNSEEN_ROOT}/inputs/"
cp "${GENODE_SOURCE_UNSEEN_ROOT}/inputs/locked_context_embeddings.npz" "${GENODE_UNSEEN_ROOT}/inputs/"

python3 - "${GENODE_SOURCE_UNSEEN_ROOT}/inputs/unseen_nfe_manifest.json" "${GENODE_UNSEEN_ROOT}" "${GENODE_VERIFICATION_ROOT}" "${GENODE_POLICY_RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

source_manifest = Path(sys.argv[1])
unseen_root = Path(sys.argv[2])
verification_root = Path(sys.argv[3])
policy_run_id = sys.argv[4]
payload = json.loads(source_manifest.read_text(encoding="utf-8"))
payload.update(
    {
        "artifact": "density_diversity_unseen_nfe_inputs_manifest",
        "copied_from_unseen_root": str(source_manifest.parent.parent),
        "policy_run_id": policy_run_id,
        "policy_summary": str(verification_root / "policy_runs" / policy_run_id / "gipo_training_summary.json"),
        "student_checkpoint": str(verification_root / "policy_runs" / policy_run_id / "gipo_student.pt"),
        "locked_context_rows": str(unseen_root / "inputs" / "locked_context_rows.csv"),
        "locked_context_embeddings": str(unseen_root / "inputs" / "locked_context_embeddings.npz"),
        "locked_test_used_for_selection": False,
    }
)
(unseen_root / "inputs" / "unseen_nfe_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

cp "${SCRIPT_DIR}/unseen_nfe_report_policy.sbatch" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/unseen_nfe_collect_summary.sbatch" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/collect_unseen_density_diversity_summary.py" "${GENODE_UNSEEN_ROOT}/jobs/"
cp "${SCRIPT_DIR}/submit_unseen_winner.sh" "${GENODE_UNSEEN_ROOT}/jobs/"

report_job=$(
  sbatch --parsable \
    --partition="${GENODE_GPU_PARTITION}" \
    --export=ALL,GENODE_VERIFICATION_ROOT="${GENODE_VERIFICATION_ROOT}",GENODE_POLICY_RUN_ID="${GENODE_POLICY_RUN_ID}",GENODE_UNSEEN_ROOT="${GENODE_UNSEEN_ROOT}" \
    "${GENODE_UNSEEN_ROOT}/jobs/unseen_nfe_report_policy.sbatch"
)
summary_job=$(
  sbatch --parsable \
    --partition="${GENODE_CPU_PARTITION}" \
    --dependency=afterany:${report_job} \
    --export=ALL,GENODE_VERIFICATION_ROOT="${GENODE_VERIFICATION_ROOT}",GENODE_POLICY_RUN_ID="${GENODE_POLICY_RUN_ID}",GENODE_UNSEEN_ROOT="${GENODE_UNSEEN_ROOT}" \
    "${GENODE_UNSEEN_ROOT}/jobs/unseen_nfe_collect_summary.sbatch"
)

export GENODE_DIVERSITY_UNSEEN_REPORT_JOB="${report_job}"
export GENODE_DIVERSITY_UNSEEN_SUMMARY_JOB="${summary_job}"
export GENODE_POLICY_RUN_ID GENODE_GPU_PARTITION GENODE_CPU_PARTITION

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
    "report_job": os.environ["GENODE_DIVERSITY_UNSEEN_REPORT_JOB"],
    "summary_job": os.environ["GENODE_DIVERSITY_UNSEEN_SUMMARY_JOB"],
    "target_nfe_values": [6, 10, 14, 16],
    "resume_mode": "density_diversity_unseen_from_calibration_winner",
}
(root / "summary" / "submitted_jobs.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
