#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENODE_SEEN_ROOT=${GENODE_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606}
mkdir -p "${GENODE_SEEN_ROOT}/logs" "${GENODE_SEEN_ROOT}/summary"

train_job=$(sbatch --parsable "${ROOT_DIR}/01_train_seen_ablation.sbatch")
echo "Submitted seen ablation training: ${train_job}"

report_job=$(sbatch --parsable --dependency=afterok:"${train_job}" "${ROOT_DIR}/02_report_seen_locked.sbatch")
echo "Submitted seen locked reports: ${report_job}"

collect_job=$(sbatch --parsable --dependency=afterok:"${report_job}" "${ROOT_DIR}/03_collect_seen_summary.sbatch")
echo "Submitted seen summary collection: ${collect_job}"
