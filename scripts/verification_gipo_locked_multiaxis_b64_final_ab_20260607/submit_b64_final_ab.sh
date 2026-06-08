#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_final_ab_20260607}

mkdir -p "${GENODE_ROOT}/logs"

prepare_job=$(sbatch --parsable "${SCRIPT_DIR}/00_prepare_reused_artifacts.sbatch")
additive_train_job=$(sbatch --parsable --dependency=afterok:${prepare_job} "${SCRIPT_DIR}/02_train_additive_locked.sbatch")
adaln_train_job=$(sbatch --parsable --dependency=afterok:${prepare_job} "${SCRIPT_DIR}/02_train_adaln_locked.sbatch")
additive_report_job=$(sbatch --parsable --dependency=afterok:${additive_train_job} "${SCRIPT_DIR}/03_report_additive_locked_seen_unseen.sbatch")
adaln_report_job=$(sbatch --parsable --dependency=afterok:${adaln_train_job} "${SCRIPT_DIR}/03_report_adaln_locked_seen_unseen.sbatch")
collect_job=$(sbatch --parsable --dependency=afterok:${additive_report_job}:${adaln_report_job} "${SCRIPT_DIR}/04_collect_b64_final_ab_summary.sbatch")

printf 'prepare_job=%s\nadditive_train_job=%s\nadaln_train_job=%s\nadditive_report_job=%s\nadaln_report_job=%s\ncollect_job=%s\n' \
  "${prepare_job}" \
  "${additive_train_job}" \
  "${adaln_train_job}" \
  "${additive_report_job}" \
  "${adaln_report_job}" \
  "${collect_job}"

