#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

prepare_job=$(sbatch --parsable "${SCRIPT_DIR}/00_prepare_reused_artifacts.sbatch")
train_job=$(sbatch --parsable --dependency=afterok:${prepare_job} "${SCRIPT_DIR}/02_train_adaln_locked.sbatch")
report_job=$(sbatch --parsable --dependency=afterok:${train_job} "${SCRIPT_DIR}/03_report_locked_seen_unseen.sbatch")
collect_job=$(sbatch --parsable --dependency=afterok:${report_job} "${SCRIPT_DIR}/04_collect_ab_summary.sbatch")

printf 'prepare_job=%s\ntrain_job=%s\nreport_job=%s\ncollect_job=%s\n' \
  "${prepare_job}" \
  "${train_job}" \
  "${report_job}" \
  "${collect_job}"
