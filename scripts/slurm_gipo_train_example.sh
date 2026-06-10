#!/usr/bin/env bash
# Generic Slurm example for retraining the GIPO teacher/student.
#
# Required environment variables:
#   GENODE_REPO              repository checkout
#   GENODE_ENV_DIR           Python environment with project dependencies
#   GENODE_OUTPUT_ROOT       output root for policy artifacts
#   GENODE_ROWS_CSV          reusable fixed/SER context rows CSV
#   GENODE_CONTEXT_NPZ       reusable context embedding sidecar
#   GENODE_SER_SUMMARY       SER schedule summary JSON
#   GENODE_UNSEEN_ROWS_CSV   train_tuning unseen-NFE rows for selector diagnostics

#SBATCH --job-name=genode_gipo
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

set -euo pipefail

GENODE_REPO=${GENODE_REPO:?set GENODE_REPO}
GENODE_ENV_DIR=${GENODE_ENV_DIR:?set GENODE_ENV_DIR}
GENODE_OUTPUT_ROOT=${GENODE_OUTPUT_ROOT:?set GENODE_OUTPUT_ROOT}
GENODE_ROWS_CSV=${GENODE_ROWS_CSV:?set GENODE_ROWS_CSV}
GENODE_CONTEXT_NPZ=${GENODE_CONTEXT_NPZ:?set GENODE_CONTEXT_NPZ}
GENODE_SER_SUMMARY=${GENODE_SER_SUMMARY:?set GENODE_SER_SUMMARY}
GENODE_UNSEEN_ROWS_CSV=${GENODE_UNSEEN_ROWS_CSV:?set GENODE_UNSEEN_ROWS_CSV}

RUN_ROOT=${GENODE_RUN_ROOT:-${GENODE_OUTPUT_ROOT}/gipo_support_choice}
SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005}
CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
UNSEEN_CONTEXT_NPZ=${GENODE_UNSEEN_CONTEXT_NPZ:-${GENODE_CONTEXT_NPZ}}

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_schedule_summary_evaluator

python -m genode.gipo.train_gipo \
  --rows_csv "${GENODE_ROWS_CSV}" \
  --context_embeddings_npz "${GENODE_CONTEXT_NPZ}" \
  --schedule_summary_json "${GENODE_SER_SUMMARY}" \
  --out_dir "${RUN_ROOT}/policy" \
  --support_schedule_keys "${SUPPORT_KEYS}" \
  --context_sample_count "${CONTEXT_SAMPLE_COUNT}" \
  --context_holdout_fraction "${GENODE_CONTEXT_HOLDOUT_FRACTION:-0.20}" \
  --teacher_unseen_selection_rows_csv "${GENODE_UNSEEN_ROWS_CSV}" \
  --teacher_unseen_selection_context_embeddings_npz "${UNSEEN_CONTEXT_NPZ}" \
  --teacher_checkpoint_every "${GENODE_TEACHER_CHECKPOINT_EVERY:-50}" \
  --teacher_steps "${GENODE_TEACHER_STEPS:-500}" \
  --student_steps "${GENODE_STUDENT_STEPS:-1000}" \
  --teacher_temperature "${GENODE_TEACHER_TEMPERATURE:-0.05}"
