#!/usr/bin/env bash
#SBATCH --job-name=genode_seen_train
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
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_CANARY_CONTEXT_SAMPLE_COUNT=${GENODE_CANARY_CONTEXT_SAMPLE_COUNT:-8}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_STEPS=${GENODE_STUDENT_STEPS:-500}
GENODE_CANARY_STUDENT_STEPS=${GENODE_CANARY_STUDENT_STEPS:-2000}

mkdir -p "${GENODE_SEEN_ROOT}/logs" "${GENODE_SEEN_ROOT}/policy_runs" "${GENODE_SEEN_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

ROWS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_rows.csv"
EMBEDDINGS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_embeddings.npz"
SER_SUMMARY="${GENODE_INPUT_ROOT}/standard_inputs/ser_reference/ser_ptg_schedule_summary.json"

train_policy() {
  local run_id=$1
  local conditioning_style=$2
  local crps_weight=$3
  local mase_weight=$4
  local context_sample_count=$5
  local student_steps=$6
  local dropout=$7
  local series_dropout=$8
  local weight_decay=$9
  local context_holdout=${10}
  local series_holdout=${11}
  shift 11
  local out_dir="${GENODE_SEEN_ROOT}/policy_runs/${run_id}"
  if [[ -f "${out_dir}/gipo_training_summary.json" ]]; then
    echo "Skipping existing ${run_id}"
    return
  fi
  mkdir -p "${out_dir}"
  python -m genode.gipo.train_gipo \
    --rows_csv "${ROWS}" \
    --context_embeddings_npz "${EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --out_dir "${out_dir}" \
    --support_schedule_keys "${GENODE_SUPPORT_KEYS}" \
    --context_sample_count "${context_sample_count}" \
    --context_holdout_fraction "${context_holdout}" \
    --series_holdout_fraction "${series_holdout}" \
    --density_bin_count 128 \
    --teacher_steps "${GENODE_TEACHER_STEPS}" \
    --student_steps "${student_steps}" \
    --teacher_temperature_mode fixed \
    --teacher_temperature 0.05 \
    --teacher_architecture density_form_transformer_v1 \
    --student_architecture density_query_transformer_v1 \
    --setting_encoder_mode continuous_v3 \
    --gipo_conditioning_style "${conditioning_style}" \
    --transformer_heads 4 \
    --transformer_dropout "${dropout}" \
    --teacher_utility_crps_weight "${crps_weight}" \
    --teacher_utility_mase_weight "${mase_weight}" \
    --series_unknown_dropout "${series_dropout}" \
    --student_weight_decay "${weight_decay}" \
    --student_nfe_smoothness_weight 0 \
    --student_pseudo_target_weight 0 \
    --device auto \
    "$@"
}

train_policy canary_additive_overfit_b128 additive_mlp_v1 0.40 0.60 "${GENODE_CANARY_CONTEXT_SAMPLE_COUNT}" "${GENODE_CANARY_STUDENT_STEPS}" 0.0 0.0 0.0 0.0 0.0 --teacher_fit_checkpoint_selection
train_policy seen_additive_mase06_b128 additive_mlp_v1 0.40 0.60 "${GENODE_CONTEXT_SAMPLE_COUNT}" "${GENODE_STUDENT_STEPS}" 0.05 0.10 0.0001 0.20 0.20
train_policy seen_additive_mase075_b128 additive_mlp_v1 0.25 0.75 "${GENODE_CONTEXT_SAMPLE_COUNT}" "${GENODE_STUDENT_STEPS}" 0.05 0.10 0.0001 0.20 0.20
