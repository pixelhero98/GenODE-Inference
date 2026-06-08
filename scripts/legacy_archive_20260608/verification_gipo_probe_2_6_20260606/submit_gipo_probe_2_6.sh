#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENODE_PROBE_ROOT=${GENODE_PROBE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606}
mkdir -p "${GENODE_PROBE_ROOT}/logs" "${GENODE_PROBE_ROOT}/summary"

refresh_job=$(sbatch --parsable "${ROOT_DIR}/00_refresh_vector_collect.sbatch")
echo "Submitted vector refresh: ${refresh_job}"

train_job=$(sbatch --parsable --dependency=afterok:"${refresh_job}" "${ROOT_DIR}/01_train_probe_policies.sbatch")
echo "Submitted probe policy training: ${train_job}"

validation_job=$(sbatch --parsable --dependency=afterok:"${train_job}" "${ROOT_DIR}/02_report_validation.sbatch")
echo "Submitted validation unseen reports: ${validation_job}"

collect_validation_job=$(sbatch --parsable --dependency=afterok:"${validation_job}" "${ROOT_DIR}/03_collect_validation.sbatch")
echo "Submitted validation collection: ${collect_validation_job}"

followup_train_job=$(sbatch --parsable --dependency=afterok:"${collect_validation_job}" "${ROOT_DIR}/04_train_followups.sbatch")
echo "Submitted conditional follow-up training: ${followup_train_job}"

followup_validation_job=$(sbatch --parsable --dependency=afterok:"${followup_train_job}" "${ROOT_DIR}/02_report_validation.sbatch")
echo "Submitted post-follow-up validation reports: ${followup_validation_job}"

selection_collect_job=$(sbatch --parsable --dependency=afterok:"${followup_validation_job}" --export=ALL,GENODE_COLLECT_MODE=select_locked "${ROOT_DIR}/06_collect_final.sbatch")
echo "Submitted validation-only locked selection collection: ${selection_collect_job}"

locked_job=$(sbatch --parsable --dependency=afterok:"${selection_collect_job}" "${ROOT_DIR}/05_report_locked_selected.sbatch")
echo "Submitted locked reporting for selected runs: ${locked_job}"

final_collect_job=$(sbatch --parsable --dependency=afterok:"${locked_job}" "${ROOT_DIR}/06_collect_final.sbatch")
echo "Submitted final probe summary collection: ${final_collect_job}"
