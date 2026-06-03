#!/usr/bin/env bash
#SBATCH --job-name=genode_div_cal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_FIXED_KEYS=${GENODE_FIXED_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,late_power_3_reversed,flowts_power_sampling_reversed,ays_reversed,gits_reversed,ots_reversed}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_NFES=${GENODE_NFES:-4,8,12}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

CAL_ROOT="${GENODE_VERIFICATION_ROOT}/calibration_inputs"
mkdir -p "${CAL_ROOT}" "${GENODE_VERIFICATION_ROOT}/logs" "${GENODE_VERIFICATION_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest \
  tests.test_diffusion_flow_schedules_review_fixes \
  tests.test_gipo \
  tests.test_genode_interfaces \
  tests.test_otflow_paper_prep

COMMON_BACKBONE_ARGS=(
  --dataset_root "${GENODE_DATASET_ROOT}"
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}"
  --seeds "${GENODE_SEEDS}"
  --solver_names "${GENODE_SOLVERS}"
  --target_nfe_values "${GENODE_NFES}"
  --device auto
)

run_fixed_context_rows() {
  local split_phase=$1
  local out_dir=$2
  local prefix=$3
  mkdir -p "${out_dir}"
  python -m genode.evaluation.diffusion_flow_time_reparameterization \
    --out_root "${out_dir}" \
    "${COMMON_BACKBONE_ARGS[@]}" \
    --forecast_datasets "${GENODE_DATASET}" \
    --conditional_generation_datasets "" \
    --split_phase "${split_phase}" \
    --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
    --train_tuning_sampling_mode validation_normalized \
    --write_forecast_context_rows \
    --forecast_context_row_csv_name "${prefix}_context_rows.csv" \
    --forecast_context_embeddings_npz_name "${prefix}_context_embeddings.npz" \
    --row_csv_name "${prefix}_rows.csv" \
    --row_jsonl_name "${prefix}_rows.jsonl" \
    --allow_execute
}

run_ser_context_rows() {
  local split_phase=$1
  local out_dir=$2
  local prefix=$3
  mkdir -p "${out_dir}"
  python -m genode.gipo.evaluate_schedule_summary \
    --dataset "${GENODE_DATASET}" \
    --schedule_summary "${CAL_ROOT}/ser_reference/ser_ptg_schedule_summary.json" \
    --split_phase "${split_phase}" \
    --out_dir "${out_dir}" \
    "${COMMON_BACKBONE_ARGS[@]}" \
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
    --train_tuning_sampling_mode validation_normalized \
    --write_context_rows \
    --context_row_csv_name "${prefix}_context_rows.csv" \
    --context_embeddings_npz_name "${prefix}_context_embeddings.npz" \
    --row_csv_name "${prefix}_rows.csv" \
    --row_jsonl_name "${prefix}_rows.jsonl"
}

run_fixed_context_rows train_tuning "${CAL_ROOT}/fixed_train_tuning" fixed_train
run_fixed_context_rows validation_tuning "${CAL_ROOT}/fixed_validation_tuning" fixed_validation
run_fixed_context_rows locked_test "${CAL_ROOT}/fixed_locked_test" fixed_locked

mkdir -p "${CAL_ROOT}/ser_reference"
python -m genode.gipo.ser_ptg_reference \
  --dataset "${GENODE_DATASET}" \
  --out_dir "${CAL_ROOT}/ser_reference" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --reference_split train_tuning \
  --train_tuning_sampling_mode validation_normalized \
  --density_floor_eta 0.05

run_ser_context_rows train_tuning "${CAL_ROOT}/ser_train_tuning" ser_train
run_ser_context_rows validation_tuning "${CAL_ROOT}/ser_validation_tuning" ser_validation
run_ser_context_rows locked_test "${CAL_ROOT}/ser_locked_test" ser_locked

python scripts/verification_density_diversity_20260603/merge_density_diversity_inputs.py --root "${GENODE_VERIFICATION_ROOT}"
