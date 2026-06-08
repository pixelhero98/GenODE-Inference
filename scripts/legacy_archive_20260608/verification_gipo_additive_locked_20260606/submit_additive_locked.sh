#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_20260606}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/summary"

archive_job=$(sbatch --parsable "${SCRIPT_DIR}/00_archive_legacy_results.sbatch")
generate_job=$(sbatch --parsable --dependency=afterok:${archive_job} "${SCRIPT_DIR}/01_generate_artifacts.sbatch")
train_job=$(sbatch --parsable --dependency=afterok:${generate_job} "${SCRIPT_DIR}/02_train_additive_locked.sbatch")
report_job=$(sbatch --parsable --dependency=afterok:${train_job} "${SCRIPT_DIR}/03_report_locked_seen_unseen.sbatch")
collect_job=$(sbatch --parsable --dependency=afterok:${report_job} "${SCRIPT_DIR}/04_collect_final_summary.sbatch")

cat <<EOF
archive_job=${archive_job}
generate_job=${generate_job}
train_job=${train_job}
report_job=${report_job}
collect_job=${collect_job}
root=${GENODE_ROOT}
EOF
