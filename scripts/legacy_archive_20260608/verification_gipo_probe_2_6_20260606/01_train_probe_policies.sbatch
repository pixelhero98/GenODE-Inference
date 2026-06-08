#!/usr/bin/env bash
#SBATCH --job-name=genode_probe_train
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_density_form_src_20260604}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_INPUT_ROOT=${GENODE_INPUT_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_density_form_20260604}
GENODE_PROBE_ROOT=${GENODE_PROBE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606}
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_STEPS=${GENODE_STUDENT_STEPS:-500}

mkdir -p "${GENODE_PROBE_ROOT}/logs" "${GENODE_PROBE_ROOT}/policy_runs" "${GENODE_PROBE_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

ROWS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_rows.csv"
EMBEDDINGS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_embeddings.npz"
SER_SUMMARY="${GENODE_INPUT_ROOT}/standard_inputs/ser_reference/ser_ptg_schedule_summary.json"
PSEUDO_ROWS="${GENODE_INPUT_ROOT}/unseen_nfe_inputs/train_context_rows.csv"
PSEUDO_EMBEDDINGS="${GENODE_INPUT_ROOT}/unseen_nfe_inputs/train_context_embeddings.npz"
PSEUDO_SER_SUMMARY="${GENODE_INPUT_ROOT}/unseen_nfe_inputs/ser_reference/ser_ptg_schedule_summary.json"
PSEUDO_NFES="6,10,14,16"

require_file() {
  local path=$1
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

for path in "${ROWS}" "${EMBEDDINGS}" "${SER_SUMMARY}" "${PSEUDO_ROWS}" "${PSEUDO_EMBEDDINGS}" "${PSEUDO_SER_SUMMARY}"; do
  require_file "${path}"
done

run_policy() {
  local run_id=$1
  local crps_weight=$2
  local mase_weight=$3
  local pseudo_weight=$4
  local student_steps=$5
  shift 5
  local out_dir="${GENODE_PROBE_ROOT}/policy_runs/${run_id}"
  local validator_args=(--student-target-mode soft_mixture)
  case "${run_id}" in
    probe_pseudo025_margin_*)
      validator_args=(--student-target-mode margin_hard_soft --teacher-hard-margin 0.05)
      ;;
  esac
  if [[ -f "${out_dir}/gipo_training_summary.json" ]]; then
    python scripts/verification_gipo_probe_2_6_20260606/validate_training_summary.py \
      --summary "${out_dir}/gipo_training_summary.json" \
      --run-id "${run_id}" \
      --crps-weight "${crps_weight}" \
      --mase-weight "${mase_weight}" \
      --pseudo-weight "${pseudo_weight}" \
      "${validator_args[@]}"
    echo "Skipping verified existing ${run_id}"
    return
  fi
  mkdir -p "${out_dir}"
  python -m genode.gipo.train_gipo \
    --rows_csv "${ROWS}" \
    --context_embeddings_npz "${EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --out_dir "${out_dir}" \
    --support_schedule_keys "${GENODE_SUPPORT_KEYS}" \
    --context_sample_count "${GENODE_CONTEXT_SAMPLE_COUNT}" \
    --density_bin_count 128 \
    --teacher_steps "${GENODE_TEACHER_STEPS}" \
    --student_steps "${student_steps}" \
    --teacher_temperature_mode fixed \
    --teacher_temperature 0.05 \
    --teacher_architecture density_form_transformer_v1 \
    --student_architecture density_query_transformer_v1 \
    --setting_encoder_mode continuous_v3 \
    --transformer_heads 4 \
    --teacher_utility_crps_weight "${crps_weight}" \
    --teacher_utility_mase_weight "${mase_weight}" \
    --student_pseudo_rows_csv "${PSEUDO_ROWS}" \
    --student_pseudo_context_embeddings_npz "${PSEUDO_EMBEDDINGS}" \
    --student_pseudo_schedule_summary_json "${PSEUDO_SER_SUMMARY}" \
    --student_pseudo_target_nfe_values "${PSEUDO_NFES}" \
    --student_pseudo_target_weight "${pseudo_weight}" \
    --device auto \
    "$@"
}

run_policy probe_pseudo010_mase06_b128 0.40 0.60 0.10 "${GENODE_STUDENT_STEPS}"
run_policy probe_pseudo025_mase06_b128 0.40 0.60 0.25 "${GENODE_STUDENT_STEPS}"
run_policy probe_pseudo050_mase06_b128 0.40 0.60 0.50 "${GENODE_STUDENT_STEPS}"
run_policy probe_pseudo025_mase075_b128 0.25 0.75 0.25 "${GENODE_STUDENT_STEPS}"
run_policy probe_pseudo025_margin_mase06_b128 0.40 0.60 0.25 "${GENODE_STUDENT_STEPS}" \
  --student_target_mode margin_hard_soft \
  --teacher_hard_margin 0.05
run_policy probe_pseudo025_margin_mase075_b128 0.25 0.75 0.25 "${GENODE_STUDENT_STEPS}" \
  --student_target_mode margin_hard_soft \
  --teacher_hard_margin 0.05
