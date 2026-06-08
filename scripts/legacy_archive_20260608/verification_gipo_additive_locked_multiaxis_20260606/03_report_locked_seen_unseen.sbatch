#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_mx_report
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606/logs/%x.%j.err

set -euo pipefail

RUN_ID=additive_locked_b128_multiaxis_select

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_additive_locked_multiaxis_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_SEEN_NFES=${GENODE_SEEN_NFES:-4,8,12}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/locked_reports"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_additive_locked_multiaxis_20260606/validate_additive_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --validate

RUN_DIR="${GENODE_ROOT}/policy_runs/${RUN_ID}"
FINAL_RETRAIN_METADATA="${RUN_DIR}/final_retrain_metadata.json"
if [[ ! -f "${FINAL_RETRAIN_METADATA}" ]]; then
  echo "Missing final-retrain metadata: ${FINAL_RETRAIN_METADATA}" >&2
  exit 1
fi

report_panel() {
  local panel=$1
  local nfes=$2
  local context_rows="${GENODE_ROOT}/artifacts/${panel}_locked_report_context_rows.csv"
  local context_embeddings="${GENODE_ROOT}/artifacts/${panel}_locked_context_embeddings.npz"
  local support_rows="${GENODE_ROOT}/artifacts/${panel}_locked_teacher_support_rows.csv"
  local fixed_rows="${GENODE_ROOT}/standard_inputs/${panel}/fixed_locked/fixed_locked_context_rows.csv"
  local ser_rows="${GENODE_ROOT}/standard_inputs/${panel}/ser_locked/ser_locked_context_rows.csv"
  local ser_summary="${GENODE_ROOT}/standard_inputs/${panel}/ser_reference/ser_ptg_schedule_summary.json"
  local student_out="${GENODE_ROOT}/locked_reports/${panel}/student/${RUN_ID}"
  local oracle_out="${GENODE_ROOT}/locked_reports/${panel}/oracle/${RUN_ID}"

  local common_report_args=(
    --dataset "${GENODE_DATASET}"
    --dataset_root "${GENODE_DATASET_ROOT}"
    --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
    --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
    --output_root "${GENODE_ROOT}"
    --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}"
    --device auto
    --seeds "${GENODE_SEEDS}"
    --solver_names "${GENODE_SOLVERS}"
    --target_nfe_values "${nfes}"
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}"
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}"
    --baseline_rows "${fixed_rows}"
    --comparator_rows "${ser_rows}"
  )

  if [[ -f "${student_out}/locked_test_gipo_policy_summary.json" ]]; then
    echo "Skipping existing ${panel} locked student report"
  else
    python -m genode.gipo.report_locked_test \
      --gipo_student_checkpoint "${RUN_DIR}/gipo_student.pt" \
      --training_summary "${RUN_DIR}/gipo_training_summary.json" \
      --context_rows "${context_rows}" \
      --context_embeddings_npz "${context_embeddings}" \
      --split_phase locked_test \
      --selection_mode reporting \
      --require_teacher_checkpoint_selection_mode composite_regret_v1 \
      --report_label "${panel}_locked" \
      --out_dir "${student_out}" \
      "${common_report_args[@]}"
  fi

  if [[ -f "${oracle_out}/locked_test_gipo_teacher_oracle_policy_summary.json" ]]; then
    echo "Skipping existing ${panel} locked teacher-oracle report"
  else
    python -m genode.gipo.report_teacher_oracle \
      --gipo_teacher_checkpoint "${RUN_DIR}/gipo_teacher.pt" \
      --training_summary "${RUN_DIR}/gipo_training_summary.json" \
      --support_rows "${support_rows}" \
      --context_embeddings_npz "${context_embeddings}" \
      --schedule_summary_json "${ser_summary}" \
      --split_phase locked_test \
      --selection_mode reporting \
      --require_teacher_checkpoint_selection_mode composite_regret_v1 \
      --report_label "${panel}_locked" \
      --out_dir "${oracle_out}" \
      "${common_report_args[@]}"
  fi
}

report_panel seen "${GENODE_SEEN_NFES}"
report_panel unseen "${GENODE_UNSEEN_NFES}"
