#!/usr/bin/env bash
#SBATCH --job-name=genode_bal_seen_report
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.err

set -euo pipefail

RUN_ID=seen_balanced_additive_b128

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_balanced_seen_multinfe_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_BALANCED_SEEN_ROOT=${GENODE_BALANCED_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606}
GENODE_INPUT_ROOT=${GENODE_INPUT_ROOT:-${GENODE_BALANCED_SEEN_ROOT}}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_SEEN_NFES=${GENODE_SEEN_NFES:-4,8,12}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_BALANCED_SEEN_ROOT}/logs" "${GENODE_BALANCED_SEEN_ROOT}/seen_locked_reports/student" "${GENODE_BALANCED_SEEN_ROOT}/seen_locked_reports/oracle"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts

STANDARD_ROOT="${GENODE_INPUT_ROOT}/standard_inputs"
CONTEXT_ROWS="${GENODE_INPUT_ROOT}/calibration_inputs/locked_context_rows.csv"
CONTEXT_EMBEDDINGS="${GENODE_INPUT_ROOT}/calibration_inputs/locked_context_embeddings.npz"
FIXED_ROWS="${STANDARD_ROOT}/fixed_locked/fixed_locked_context_rows.csv"
SER_ROWS="${STANDARD_ROOT}/ser_locked/ser_locked_context_rows.csv"
SER_SUMMARY="${STANDARD_ROOT}/ser_reference/ser_ptg_schedule_summary.json"
RUN_DIR="${GENODE_BALANCED_SEEN_ROOT}/policy_runs/${RUN_ID}"

COMMON_REPORT_ARGS=(
  --dataset "${GENODE_DATASET}"
  --dataset_root "${GENODE_DATASET_ROOT}"
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
  --output_root "${GENODE_BALANCED_SEEN_ROOT}"
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}"
  --device auto
  --seeds "${GENODE_SEEDS}"
  --solver_names "${GENODE_SOLVERS}"
  --target_nfe_values "${GENODE_SEEN_NFES}"
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}"
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}"
  --baseline_rows "${FIXED_ROWS}"
  --comparator_rows "${SER_ROWS}"
)

STUDENT_OUT="${GENODE_BALANCED_SEEN_ROOT}/seen_locked_reports/student/${RUN_ID}"
if [[ -f "${STUDENT_OUT}/locked_test_gipo_policy_summary.json" ]]; then
  echo "Skipping existing seen locked student report ${RUN_ID}"
else
  python -m genode.gipo.report_locked_test \
    --gipo_student_checkpoint "${RUN_DIR}/gipo_student.pt" \
    --training_summary "${RUN_DIR}/gipo_training_summary.json" \
    --context_rows "${CONTEXT_ROWS}" \
    --context_embeddings_npz "${CONTEXT_EMBEDDINGS}" \
    --split_phase locked_test \
    --selection_mode reporting \
    --report_label seen_locked \
    --out_dir "${STUDENT_OUT}" \
    "${COMMON_REPORT_ARGS[@]}"
fi

ORACLE_OUT="${GENODE_BALANCED_SEEN_ROOT}/seen_locked_reports/oracle/${RUN_ID}"
if [[ -f "${ORACLE_OUT}/locked_test_gipo_teacher_oracle_policy_summary.json" ]]; then
  echo "Skipping existing seen locked oracle report ${RUN_ID}"
else
  python -m genode.gipo.report_teacher_oracle \
    --gipo_teacher_checkpoint "${RUN_DIR}/gipo_teacher.pt" \
    --training_summary "${RUN_DIR}/gipo_training_summary.json" \
    --support_rows "${CONTEXT_ROWS}" \
    --context_embeddings_npz "${CONTEXT_EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --split_phase locked_test \
    --selection_mode reporting \
    --report_label seen_locked \
    --out_dir "${ORACLE_OUT}" \
    "${COMMON_REPORT_ARGS[@]}"
fi
