#!/usr/bin/env bash
#SBATCH --job-name=genode_probe_followup_train
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=18:00:00
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

mkdir -p "${GENODE_PROBE_ROOT}/logs" "${GENODE_PROBE_ROOT}/policy_runs" "${GENODE_PROBE_ROOT}/summary"

FOLLOWUP_FILE="${GENODE_PROBE_ROOT}/summary/followup_run_ids.txt"
if [[ ! -s "${FOLLOWUP_FILE}" ]]; then
  echo "No follow-up runs requested by validation collector."
  exit 0
fi

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts

ROWS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_rows.csv"
EMBEDDINGS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_embeddings.npz"
SER_SUMMARY="${GENODE_INPUT_ROOT}/standard_inputs/ser_reference/ser_ptg_schedule_summary.json"
PSEUDO_ROWS="${GENODE_INPUT_ROOT}/unseen_nfe_inputs/train_context_rows.csv"
PSEUDO_EMBEDDINGS="${GENODE_INPUT_ROOT}/unseen_nfe_inputs/train_context_embeddings.npz"
PSEUDO_SER_SUMMARY="${GENODE_INPUT_ROOT}/unseen_nfe_inputs/ser_reference/ser_ptg_schedule_summary.json"
PSEUDO_NFES="6,10,14,16"

train_followup() {
  local run_id=$1
  shift
  local out_dir="${GENODE_PROBE_ROOT}/policy_runs/${run_id}"
  if [[ -f "${out_dir}/gipo_training_summary.json" ]]; then
    python scripts/verification_gipo_probe_2_6_20260606/validate_training_summary.py \
      --summary "${out_dir}/gipo_training_summary.json" \
      --run-id "${run_id}" \
      --crps-weight 0.40 \
      --mase-weight 0.60 \
      --pseudo-weight 0.25 \
      --student-target-mode soft_mixture
    echo "Skipping verified existing follow-up ${run_id}"
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
    --teacher_temperature_mode fixed \
    --teacher_temperature 0.05 \
    --teacher_architecture density_form_transformer_v1 \
    --student_architecture density_query_transformer_v1 \
    --setting_encoder_mode continuous_v3 \
    --transformer_heads 4 \
    --teacher_utility_crps_weight 0.40 \
    --teacher_utility_mase_weight 0.60 \
    --student_pseudo_rows_csv "${PSEUDO_ROWS}" \
    --student_pseudo_context_embeddings_npz "${PSEUDO_EMBEDDINGS}" \
    --student_pseudo_schedule_summary_json "${PSEUDO_SER_SUMMARY}" \
    --student_pseudo_target_nfe_values "${PSEUDO_NFES}" \
    --student_pseudo_target_weight 0.25 \
    --device auto \
    "$@"
}

while IFS= read -r run_id; do
  case "${run_id}" in
    probe_pseudo025_mase06_student1000_b128)
      train_followup "${run_id}" --student_steps 1000
      ;;
    probe_pseudo025_mase06_smooth005_b128)
      train_followup "${run_id}" --student_steps 500 --student_nfe_smoothness_weight 0.005
      ;;
    "")
      ;;
    *)
      echo "Unknown follow-up run id: ${run_id}" >&2
      exit 1
      ;;
  esac
done < "${FOLLOWUP_FILE}"
