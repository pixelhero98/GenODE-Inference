#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENODE_BALANCED_SEEN_ROOT=${GENODE_BALANCED_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606}
GENODE_ROOT=${GENODE_ROOT:-${GENODE_BALANCED_SEEN_ROOT}}
export GENODE_BALANCED_SEEN_ROOT GENODE_ROOT
mkdir -p "${GENODE_BALANCED_SEEN_ROOT}/logs" "${GENODE_BALANCED_SEEN_ROOT}/summary"

input_job=$(sbatch --parsable "${ROOT_DIR}/00_generate_balanced_inputs.sbatch")
echo "Submitted balanced seen input generation: ${input_job}"

train_job=$(sbatch --parsable --dependency=afterok:"${input_job}" "${ROOT_DIR}/01_train_balanced_seen_multinfe.sbatch")
echo "Submitted balanced seen multi-NFE training: ${train_job}"

report_job=$(sbatch --parsable --dependency=afterok:"${train_job}" "${ROOT_DIR}/02_report_seen_locked.sbatch")
echo "Submitted balanced seen locked reports: ${report_job}"

collect_job=$(sbatch --parsable --dependency=afterok:"${report_job}" "${ROOT_DIR}/03_collect_balanced_seen_summary.sbatch")
echo "Submitted balanced seen summary collection: ${collect_job}"
