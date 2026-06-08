#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ROOT16=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ckpt16k_20260608
ROOT12=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ckpt12k_20260608
SUMMARY_ROOT=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608
RUN16=additive_locked_b64_normregret_ckpt16k
RUN12=additive_locked_b64_normregret_ckpt12k

EXPORT16=ALL,GENODE_OTFLOW_TRAIN_STEPS=16000,GENODE_BUDGET_LABEL=16k,GENODE_ROOT=${ROOT16},RUN_ID=${RUN16},REPORT_LABEL_PREFIX=additive_b64_normregret_ckpt16k
EXPORT12=ALL,GENODE_OTFLOW_TRAIN_STEPS=12000,GENODE_BUDGET_LABEL=12k,GENODE_ROOT=${ROOT12},RUN_ID=${RUN12},REPORT_LABEL_PREFIX=additive_b64_normregret_ckpt12k

artifact16_job=$(sbatch --parsable --export="${EXPORT16}" "${SCRIPT_DIR}/01_generate_artifacts.sbatch")
artifact12_job=$(sbatch --parsable --export="${EXPORT12}" "${SCRIPT_DIR}/01_generate_artifacts.sbatch")
train16_job=$(sbatch --parsable --dependency=afterok:${artifact16_job} --export="${EXPORT16}" "${SCRIPT_DIR}/02_train_additive.sbatch")
train12_job=$(sbatch --parsable --dependency=afterok:${artifact12_job} --export="${EXPORT12}" "${SCRIPT_DIR}/02_train_additive.sbatch")
report16_job=$(sbatch --parsable --dependency=afterok:${train16_job} --export="${EXPORT16}" "${SCRIPT_DIR}/03_report_additive_seen_unseen.sbatch")
report12_job=$(sbatch --parsable --dependency=afterok:${train12_job} --export="${EXPORT12}" "${SCRIPT_DIR}/03_report_additive_seen_unseen.sbatch")
collect_job=$(sbatch --parsable --dependency=afterok:${report16_job}:${report12_job} --export=ALL,GENODE_SUMMARY_ROOT="${SUMMARY_ROOT}",GENODE_CKPT16_ROOT="${ROOT16}",GENODE_CKPT12_ROOT="${ROOT12}" "${SCRIPT_DIR}/04_collect_checkpoint_maturity_summary.sbatch")

printf 'artifact16_job=%s\nartifact12_job=%s\ntrain16_job=%s\ntrain12_job=%s\nreport16_job=%s\nreport12_job=%s\ncollect_job=%s\n' \
  "${artifact16_job}" \
  "${artifact12_job}" \
  "${train16_job}" \
  "${train12_job}" \
  "${report16_job}" \
  "${report12_job}" \
  "${collect_job}"
