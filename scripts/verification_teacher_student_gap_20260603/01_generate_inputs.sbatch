#!/usr/bin/env bash
#SBATCH --job-name=genode_gap_inputs
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
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_FIXED_KEYS=${GENODE_FIXED_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_STANDARD_NFES=${GENODE_STANDARD_NFES:-4,8,12}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_GAP_ROOT}/logs" "${GENODE_GAP_ROOT}/summary" "${GENODE_GAP_ROOT}/standard_inputs" "${GENODE_GAP_ROOT}/unseen_nfe_inputs"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

run_fixed_context_rows() {
  local section=$1
  local split_phase=$2
  local nfes=$3
  local prefix=$4
  local out_dir="${GENODE_GAP_ROOT}/${section}/${prefix}_${split_phase%%_*}"
  mkdir -p "${out_dir}"
  python -m genode.evaluation.diffusion_flow_time_reparameterization \
    --out_root "${out_dir}" \
    --dataset_root "${GENODE_DATASET_ROOT}" \
    --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
    --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
    --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
    --seeds "${GENODE_SEEDS}" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --device auto \
    --forecast_datasets "${GENODE_DATASET}" \
    --conditional_generation_datasets "" \
    --split_phase "${split_phase}" \
    --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
    --train_tuning_sampling_mode validation_normalized \
    --write_forecast_context_rows \
    --forecast_context_row_csv_name "${prefix}_${split_phase%%_*}_context_rows.csv" \
    --forecast_context_embeddings_npz_name "${prefix}_${split_phase%%_*}_context_embeddings.npz" \
    --row_csv_name "${prefix}_${split_phase%%_*}_rows.csv" \
    --row_jsonl_name "${prefix}_${split_phase%%_*}_rows.jsonl" \
    --allow_execute
}

build_ser_reference() {
  local section=$1
  local nfes=$2
  local out_dir="${GENODE_GAP_ROOT}/${section}/ser_reference"
  mkdir -p "${out_dir}"
  python -m genode.gipo.ser_ptg_reference \
    --dataset "${GENODE_DATASET}" \
    --out_dir "${out_dir}" \
    --dataset_root "${GENODE_DATASET_ROOT}" \
    --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
    --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
    --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
    --seeds "${GENODE_SEEDS}" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --device auto \
    --reference_split train_tuning \
    --train_tuning_sampling_mode validation_normalized \
    --density_floor_eta 0.05
}

run_ser_context_rows() {
  local section=$1
  local split_phase=$2
  local nfes=$3
  local prefix=$4
  local out_dir="${GENODE_GAP_ROOT}/${section}/${prefix}_${split_phase%%_*}"
  mkdir -p "${out_dir}"
  python -m genode.gipo.evaluate_schedule_summary \
    --dataset "${GENODE_DATASET}" \
    --schedule_summary "${GENODE_GAP_ROOT}/${section}/ser_reference/ser_ptg_schedule_summary.json" \
    --split_phase "${split_phase}" \
    --out_dir "${out_dir}" \
    --dataset_root "${GENODE_DATASET_ROOT}" \
    --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
    --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
    --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
    --seeds "${GENODE_SEEDS}" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --device auto \
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
    --train_tuning_sampling_mode validation_normalized \
    --write_context_rows \
    --context_row_csv_name "${prefix}_${split_phase%%_*}_context_rows.csv" \
    --context_embeddings_npz_name "${prefix}_${split_phase%%_*}_context_embeddings.npz" \
    --row_csv_name "${prefix}_${split_phase%%_*}_rows.csv" \
    --row_jsonl_name "${prefix}_${split_phase%%_*}_rows.jsonl"
}

for split in train_tuning validation_tuning locked_test; do
  run_fixed_context_rows standard_inputs "${split}" "${GENODE_STANDARD_NFES}" fixed
done
build_ser_reference standard_inputs "${GENODE_STANDARD_NFES}"
for split in train_tuning validation_tuning locked_test; do
  run_ser_context_rows standard_inputs "${split}" "${GENODE_STANDARD_NFES}" ser
done

for split in train_tuning validation_tuning locked_test; do
  run_fixed_context_rows unseen_nfe_inputs "${split}" "${GENODE_UNSEEN_NFES}" fixed
done
build_ser_reference unseen_nfe_inputs "${GENODE_UNSEEN_NFES}"
for split in train_tuning validation_tuning locked_test; do
  run_ser_context_rows unseen_nfe_inputs "${split}" "${GENODE_UNSEEN_NFES}" ser
done

python scripts/verification_teacher_student_gap_20260603/merge_gap_inputs.py \
  --root "${GENODE_GAP_ROOT}" \
  --standard_nfes "${GENODE_STANDARD_NFES}" \
  --unseen_nfes "${GENODE_UNSEEN_NFES}"
