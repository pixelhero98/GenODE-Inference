#!/usr/bin/env bash
#SBATCH --job-name=genode_gap_report
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/gipo_gap_src_20260603}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_GAP_ROOT=${GENODE_GAP_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603}
GENODE_REFERENCE_ROOT=${GENODE_REFERENCE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_GAP_ROOT}/logs" "${GENODE_GAP_ROOT}/calibration_unseen_reports/student" "${GENODE_GAP_ROOT}/calibration_unseen_reports/oracle"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

UNSEEN_ROOT="${GENODE_GAP_ROOT}/unseen_nfe_inputs"
CONTEXT_ROWS="${UNSEEN_ROOT}/validation_context_rows.csv"
CONTEXT_EMBEDDINGS="${UNSEEN_ROOT}/validation_context_embeddings.npz"
FIXED_ROWS="${UNSEEN_ROOT}/fixed_validation/fixed_validation_rows.csv"
SER_ROWS="${UNSEEN_ROOT}/ser_validation/ser_validation_rows.csv"
SER_SUMMARY="${UNSEEN_ROOT}/ser_reference/ser_ptg_schedule_summary.json"

COMMON_REPORT_ARGS=(
  --dataset "${GENODE_DATASET}"
  --dataset_root "${GENODE_DATASET_ROOT}"
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
  --output_root "${GENODE_GAP_ROOT}"
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}"
  --device auto
  --seeds "${GENODE_SEEDS}"
  --solver_names "${GENODE_SOLVERS}"
  --target_nfe_values "${GENODE_UNSEEN_NFES}"
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}"
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}"
  --baseline_rows "${FIXED_ROWS}"
  --comparator_rows "${SER_ROWS}"
)

run_student_report() {
  local run_id=$1
  local run_root=$2
  local run_dir="${run_root}/policy_runs/${run_id}"
  local out_dir="${GENODE_GAP_ROOT}/calibration_unseen_reports/student/${run_id}"
  if [[ -f "${out_dir}/unseen_validation_gipo_policy_summary.json" ]]; then
    echo "Skipping existing student report ${run_id}"
    return
  fi
  python -m genode.gipo.report_locked_test \
    --gipo_student_checkpoint "${run_dir}/gipo_student.pt" \
    --training_summary "${run_dir}/gipo_training_summary.json" \
    --context_rows "${CONTEXT_ROWS}" \
    --context_embeddings_npz "${CONTEXT_EMBEDDINGS}" \
    --split_phase validation_tuning \
    --selection_mode calibration \
    --report_label unseen_validation \
    --out_dir "${out_dir}" \
    "${COMMON_REPORT_ARGS[@]}"
}

run_oracle_report() {
  local run_id=$1
  local run_root=$2
  local run_dir="${run_root}/policy_runs/${run_id}"
  local out_dir="${GENODE_GAP_ROOT}/calibration_unseen_reports/oracle/${run_id}"
  if [[ -f "${out_dir}/unseen_validation_gipo_teacher_oracle_policy_summary.json" ]]; then
    echo "Skipping existing oracle report ${run_id}"
    return
  fi
  python -m genode.gipo.report_teacher_oracle \
    --gipo_teacher_checkpoint "${run_dir}/gipo_teacher.pt" \
    --training_summary "${run_dir}/gipo_training_summary.json" \
    --support_rows "${CONTEXT_ROWS}" \
    --context_embeddings_npz "${CONTEXT_EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --split_phase validation_tuning \
    --selection_mode calibration \
    --report_label unseen_validation \
    --out_dir "${out_dir}" \
    "${COMMON_REPORT_ARGS[@]}"
}

RUN_IDS=(temp_fixed_005_b128 margin_fixed005_b128 nferich_fixed005_b128 nferich_margin_fixed005_b128 mase06_nferich_margin_fixed005_b128 mase075_nferich_margin_fixed005_b128)
RUN_ROOTS=("${GENODE_REFERENCE_ROOT}" "${GENODE_GAP_ROOT}" "${GENODE_GAP_ROOT}" "${GENODE_GAP_ROOT}" "${GENODE_GAP_ROOT}" "${GENODE_GAP_ROOT}")

for idx in "${!RUN_IDS[@]}"; do
  run_student_report "${RUN_IDS[$idx]}" "${RUN_ROOTS[$idx]}"
  run_oracle_report "${RUN_IDS[$idx]}" "${RUN_ROOTS[$idx]}"
done
