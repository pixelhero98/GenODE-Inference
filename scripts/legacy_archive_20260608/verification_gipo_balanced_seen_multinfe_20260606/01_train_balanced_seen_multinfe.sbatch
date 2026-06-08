#!/usr/bin/env bash
#SBATCH --job-name=genode_bal_seen_train
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.err

set -euo pipefail

RUN_ID=seen_balanced_additive_b128
CONDITIONING_STYLE=additive_mlp_v1
CRPS_WEIGHT=0.50
MASE_WEIGHT=0.50

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_balanced_seen_multinfe_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_BALANCED_SEEN_ROOT=${GENODE_BALANCED_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606}
GENODE_INPUT_ROOT=${GENODE_INPUT_ROOT:-${GENODE_BALANCED_SEEN_ROOT}}
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_STEPS=${GENODE_STUDENT_STEPS:-500}

mkdir -p "${GENODE_BALANCED_SEEN_ROOT}/logs" "${GENODE_BALANCED_SEEN_ROOT}/policy_runs" "${GENODE_BALANCED_SEEN_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

ROWS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_rows.csv"
EMBEDDINGS="${GENODE_INPUT_ROOT}/calibration_inputs/context_calibration_embeddings.npz"
SER_SUMMARY="${GENODE_INPUT_ROOT}/standard_inputs/ser_reference/ser_ptg_schedule_summary.json"
OUT_DIR="${GENODE_BALANCED_SEEN_ROOT}/policy_runs/${RUN_ID}"

if [[ -f "${OUT_DIR}/gipo_training_summary.json" ]]; then
  echo "Skipping existing ${RUN_ID}"
else
  mkdir -p "${OUT_DIR}"
  python -m genode.gipo.train_gipo \
    --rows_csv "${ROWS}" \
    --context_embeddings_npz "${EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --out_dir "${OUT_DIR}" \
    --support_schedule_keys "${GENODE_SUPPORT_KEYS}" \
    --context_sample_count "${GENODE_CONTEXT_SAMPLE_COUNT}" \
    --context_holdout_fraction 0.20 \
    --series_holdout_fraction 0.20 \
    --density_bin_count 128 \
    --teacher_steps "${GENODE_TEACHER_STEPS}" \
    --student_steps "${GENODE_STUDENT_STEPS}" \
    --teacher_temperature_mode fixed \
    --teacher_temperature 0.05 \
    --teacher_architecture density_form_transformer_v1 \
    --student_architecture density_query_transformer_v1 \
    --setting_encoder_mode continuous_v3 \
    --gipo_conditioning_style "${CONDITIONING_STYLE}" \
    --transformer_heads 4 \
    --transformer_dropout 0.05 \
    --teacher_utility_crps_weight "${CRPS_WEIGHT}" \
    --teacher_utility_mase_weight "${MASE_WEIGHT}" \
    --series_unknown_dropout 0 \
    --student_weight_decay 0.0001 \
    --student_nfe_smoothness_weight 0 \
    --student_pseudo_target_weight 0 \
    --device auto
fi

python scripts/verification_gipo_balanced_seen_multinfe_20260606/collect_balanced_seen_summary.py \
  --root "${GENODE_BALANCED_SEEN_ROOT}" \
  --mode training
