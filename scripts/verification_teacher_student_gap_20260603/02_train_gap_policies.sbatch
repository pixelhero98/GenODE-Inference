#!/usr/bin/env bash
#SBATCH --job-name=genode_gap_train
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/gipo_gap_src_20260603}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_GAP_ROOT=${GENODE_GAP_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_STEPS=${GENODE_STUDENT_STEPS:-500}

mkdir -p "${GENODE_GAP_ROOT}/logs" "${GENODE_GAP_ROOT}/policy_runs" "${GENODE_GAP_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

ROWS="${GENODE_GAP_ROOT}/calibration_inputs/context_calibration_rows.csv"
EMBEDDINGS="${GENODE_GAP_ROOT}/calibration_inputs/context_calibration_embeddings.npz"
SER_SUMMARY="${GENODE_GAP_ROOT}/standard_inputs/ser_reference/ser_ptg_schedule_summary.json"

run_policy() {
  local run_id=$1
  shift
  local out_dir="${GENODE_GAP_ROOT}/policy_runs/${run_id}"
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
    --context_sample_count "${GENODE_CONTEXT_SAMPLE_COUNT}" \
    --density_bin_count 128 \
    --teacher_steps "${GENODE_TEACHER_STEPS}" \
    --student_steps "${GENODE_STUDENT_STEPS}" \
    --teacher_temperature_mode fixed \
    --teacher_temperature 0.05 \
    --device auto \
    "$@"
}

run_policy margin_fixed005_b128 \
  --student_target_mode margin_hard_soft \
  --teacher_hard_margin 0.05

run_policy nferich_fixed005_b128 \
  --setting_feature_mode nfe_rich_v1

run_policy nferich_margin_fixed005_b128 \
  --setting_feature_mode nfe_rich_v1 \
  --student_target_mode margin_hard_soft \
  --teacher_hard_margin 0.05

run_policy mase06_nferich_margin_fixed005_b128 \
  --setting_feature_mode nfe_rich_v1 \
  --student_target_mode margin_hard_soft \
  --teacher_hard_margin 0.05 \
  --teacher_utility_crps_weight 0.40 \
  --teacher_utility_mase_weight 0.60

run_policy mase075_nferich_margin_fixed005_b128 \
  --setting_feature_mode nfe_rich_v1 \
  --student_target_mode margin_hard_soft \
  --teacher_hard_margin 0.05 \
  --teacher_utility_crps_weight 0.25 \
  --teacher_utility_mase_weight 0.75
