#!/usr/bin/env bash
#SBATCH --job-name=genode_seen_report
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_seen_ablation_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_INPUT_ROOT=${GENODE_INPUT_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_density_form_20260604}
GENODE_SEEN_ROOT=${GENODE_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606}
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

mkdir -p "${GENODE_SEEN_ROOT}/logs" "${GENODE_SEEN_ROOT}/seen_locked_reports/student" "${GENODE_SEEN_ROOT}/seen_locked_reports/oracle"

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

COMMON_REPORT_ARGS=(
  --dataset "${GENODE_DATASET}"
  --dataset_root "${GENODE_DATASET_ROOT}"
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
  --output_root "${GENODE_SEEN_ROOT}"
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

run_student_report() {
  local run_id=$1
  local run_dir="${GENODE_SEEN_ROOT}/policy_runs/${run_id}"
  local out_dir="${GENODE_SEEN_ROOT}/seen_locked_reports/student/${run_id}"
  if [[ -f "${out_dir}/locked_test_gipo_policy_summary.json" ]]; then
    echo "Skipping existing seen locked student report ${run_id}"
    return
  fi
  python -m genode.gipo.report_locked_test \
    --gipo_student_checkpoint "${run_dir}/gipo_student.pt" \
    --training_summary "${run_dir}/gipo_training_summary.json" \
    --context_rows "${CONTEXT_ROWS}" \
    --context_embeddings_npz "${CONTEXT_EMBEDDINGS}" \
    --split_phase locked_test \
    --selection_mode reporting \
    --report_label seen_locked \
    --out_dir "${out_dir}" \
    "${COMMON_REPORT_ARGS[@]}"
}

run_oracle_report() {
  local run_id=$1
  local run_dir="${GENODE_SEEN_ROOT}/policy_runs/${run_id}"
  local out_dir="${GENODE_SEEN_ROOT}/seen_locked_reports/oracle/${run_id}"
  if [[ -f "${out_dir}/locked_test_gipo_teacher_oracle_policy_summary.json" ]]; then
    echo "Skipping existing seen locked oracle report ${run_id}"
    return
  fi
  python -m genode.gipo.report_teacher_oracle \
    --gipo_teacher_checkpoint "${run_dir}/gipo_teacher.pt" \
    --training_summary "${run_dir}/gipo_training_summary.json" \
    --support_rows "${CONTEXT_ROWS}" \
    --context_embeddings_npz "${CONTEXT_EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --split_phase locked_test \
    --selection_mode reporting \
    --report_label seen_locked \
    --out_dir "${out_dir}" \
    "${COMMON_REPORT_ARGS[@]}"
}

RUN_IDS=(
  seen_additive_mase06_b128
  seen_additive_mase075_b128
)

for run_id in "${RUN_IDS[@]}"; do
  run_student_report "${run_id}"
  run_oracle_report "${run_id}"
done
