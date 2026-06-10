#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_adaln_b64final_report
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err

set -euo pipefail

RUN_ID=adaln_locked_b64_normregret_final
REPORT_LABEL_PREFIX=adaln_b64_normregret_final
SELECTION_MODE=weighted_normalized_regret_v1

GENODE_REPO=${GENODE_REPO:?Set GENODE_REPO to the GenODE-Inference repository path}
GENODE_ENV_DIR=${GENODE_ENV_DIR:?Set GENODE_ENV_DIR to the Python environment path}
GENODE_ROOT=${GENODE_ROOT:?Set GENODE_ROOT to this run output root}
GENODE_SOURCE_ROOT=${GENODE_SOURCE_ROOT:?Set GENODE_SOURCE_ROOT to the validated artifact source root}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:?Set GENODE_DATASET_ROOT to the dataset root}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:?Set GENODE_BACKBONE_MANIFEST to backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:?Set GENODE_SHARED_BACKBONE_ROOT to the backbone output root}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_SEEN_NFES=${GENODE_SEEN_NFES:-4,8,12}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/locked_reports"
ln -sfn "${GENODE_SOURCE_ROOT}/artifacts" "${GENODE_ROOT}/artifacts"
ln -sfn "${GENODE_SOURCE_ROOT}/standard_inputs" "${GENODE_ROOT}/standard_inputs"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_locked_multiaxis_b64_final_ab_20260607/validate_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --validate

SELECTION_MODE="${SELECTION_MODE}" python - <<'PY'
import os
from genode.gipo.policy import TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1

if os.environ["SELECTION_MODE"] != TEACHER_CHECKPOINT_SELECTION_WEIGHTED_NORMALIZED_REGRET_V1:
    raise SystemExit("GIPO reports require weighted_normalized_regret_v1")
PY

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
  local fixed_rows="${GENODE_ROOT}/standard_inputs/${panel}/fixed_locked/fixed_locked_context_rows.csv"
  local ser_rows="${GENODE_ROOT}/standard_inputs/${panel}/ser_locked/ser_locked_context_rows.csv"
  local student_out="${GENODE_ROOT}/locked_reports/${panel}/student/${RUN_ID}"

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
    echo "Skipping existing ${panel} locked AdaLN b64 final student report"
  else
    python -m genode.gipo.report_locked_test \
      --gipo_student_checkpoint "${RUN_DIR}/gipo_student.pt" \
      --training_summary "${RUN_DIR}/gipo_training_summary.json" \
      --context_rows "${context_rows}" \
      --context_embeddings_npz "${context_embeddings}" \
      --split_phase locked_test \
      --selection_mode reporting \
      --require_teacher_checkpoint_selection_mode "${SELECTION_MODE}" \
      --allow_noncanonical_conditioning \
      --report_label "${panel}_locked_${REPORT_LABEL_PREFIX}" \
      --out_dir "${student_out}" \
      "${common_report_args[@]}"
  fi
}

report_panel seen "${GENODE_SEEN_NFES}"
report_panel unseen "${GENODE_UNSEEN_NFES}"
