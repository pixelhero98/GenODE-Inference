#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_ckpt_report
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs/%x.%j.err

set -euo pipefail

SELECTION_MODE=weighted_normalized_regret_v1

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_locked_multiaxis_b64_checkpoint_maturity_src_20260608}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:?Set GENODE_OTFLOW_TRAIN_STEPS to 16000 or 12000}
GENODE_BUDGET_LABEL=${GENODE_BUDGET_LABEL:-$((GENODE_OTFLOW_TRAIN_STEPS / 1000))k}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ckpt${GENODE_BUDGET_LABEL}_20260608}
RUN_ID=${RUN_ID:-additive_locked_b64_normregret_ckpt${GENODE_BUDGET_LABEL}}
REPORT_LABEL_PREFIX=${REPORT_LABEL_PREFIX:-additive_b64_normregret_ckpt${GENODE_BUDGET_LABEL}}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_SEEN_NFES=${GENODE_SEEN_NFES:-4,8,12}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/locked_reports"
mkdir -p /scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/validate_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --validate

RUN_DIR="${GENODE_ROOT}/policy_runs/${RUN_ID}"
if [[ ! -f "${RUN_DIR}/final_retrain_metadata.json" ]]; then
  echo "Missing final-retrain metadata under ${RUN_DIR}" >&2
  exit 1
fi

report_panel() {
  local panel=$1
  local nfes=$2
  local context_rows="${GENODE_ROOT}/artifacts/${panel}_locked_report_context_rows.csv"
  local context_embeddings="${GENODE_ROOT}/artifacts/${panel}_locked_context_embeddings.npz"
  local fixed_rows="${GENODE_ROOT}/standard_inputs/${panel}/fixed_locked/fixed_locked_context_rows.csv"
  local ser_rows="${GENODE_ROOT}/standard_inputs/${panel}/ser_locked/ser_locked_context_rows.csv"
  local student_out="${GENODE_ROOT}/locked_reports/${panel}/student/${RUN_ID}"

  if [[ -f "${student_out}/locked_test_gipo_policy_summary.json" ]]; then
    echo "Skipping existing ${panel} locked ${RUN_ID} student report"
  else
    python -m genode.gipo.report_locked_test \
      --gipo_student_checkpoint "${RUN_DIR}/gipo_student.pt" \
      --training_summary "${RUN_DIR}/gipo_training_summary.json" \
      --context_rows "${context_rows}" \
      --context_embeddings_npz "${context_embeddings}" \
      --split_phase locked_test \
      --selection_mode reporting \
      --require_teacher_checkpoint_selection_mode "${SELECTION_MODE}" \
      --report_label "${panel}_locked_${REPORT_LABEL_PREFIX}" \
      --out_dir "${student_out}" \
      --dataset "${GENODE_DATASET}" \
      --dataset_root "${GENODE_DATASET_ROOT}" \
      --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
      --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
      --output_root "${GENODE_ROOT}" \
      --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
      --device auto \
      --seeds "${GENODE_SEEDS}" \
      --solver_names "${GENODE_SOLVERS}" \
      --target_nfe_values "${nfes}" \
      --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
      --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
      --baseline_rows "${fixed_rows}" \
      --comparator_rows "${ser_rows}"
  fi
}

report_panel seen "${GENODE_SEEN_NFES}"
report_panel unseen "${GENODE_UNSEEN_NFES}"
