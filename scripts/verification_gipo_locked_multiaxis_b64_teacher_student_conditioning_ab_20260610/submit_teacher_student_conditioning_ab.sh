#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENODE_ROOT=${GENODE_ROOT:?Set GENODE_ROOT to this run output root}

mkdir -p "${GENODE_ROOT}/logs"

prepare_job=$(sbatch --parsable "${SCRIPT_DIR}/00_prepare_reused_artifacts.sbatch")
teacher_additive_student_adaln_train_job=$(sbatch --parsable --dependency=afterok:${prepare_job} "${SCRIPT_DIR}/02_train_teacher_additive_student_adaln_locked.sbatch")
teacher_adaln_student_additive_train_job=$(sbatch --parsable --dependency=afterok:${prepare_job} "${SCRIPT_DIR}/02_train_teacher_adaln_student_additive_locked.sbatch")
teacher_additive_student_adaln_report_job=$(sbatch --parsable --dependency=afterok:${teacher_additive_student_adaln_train_job} "${SCRIPT_DIR}/03_report_teacher_additive_student_adaln_locked_seen_unseen.sbatch")
teacher_adaln_student_additive_report_job=$(sbatch --parsable --dependency=afterok:${teacher_adaln_student_additive_train_job} "${SCRIPT_DIR}/03_report_teacher_adaln_student_additive_locked_seen_unseen.sbatch")
collect_job=$(sbatch --parsable --dependency=afterok:${teacher_additive_student_adaln_report_job}:${teacher_adaln_student_additive_report_job} "${SCRIPT_DIR}/04_collect_teacher_student_conditioning_ab_summary.sbatch")

printf 'prepare_job=%s\nteacher_additive_student_adaln_train_job=%s\nteacher_adaln_student_additive_train_job=%s\nteacher_additive_student_adaln_report_job=%s\nteacher_adaln_student_additive_report_job=%s\ncollect_job=%s\n' \
  "${prepare_job}" \
  "${teacher_additive_student_adaln_train_job}" \
  "${teacher_adaln_student_additive_train_job}" \
  "${teacher_additive_student_adaln_report_job}" \
  "${teacher_adaln_student_additive_report_job}" \
  "${collect_job}"
