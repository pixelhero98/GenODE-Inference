#!/usr/bin/env bash
#SBATCH --job-name=genode_ess_train
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
GENODE_RUN_ID=${GENODE_RUN_ID:?set GENODE_RUN_ID}
GENODE_TEMPERATURE_MODE=${GENODE_TEMPERATURE_MODE:-fixed}
GENODE_TEACHER_TEMPERATURE=${GENODE_TEACHER_TEMPERATURE:-0.05}
GENODE_TEACHER_TARGET_ESS=${GENODE_TEACHER_TARGET_ESS:-2.5}
GENODE_TEACHER_MIN_TEMPERATURE=${GENODE_TEACHER_MIN_TEMPERATURE:-0.01}
GENODE_TEACHER_MAX_TEMPERATURE=${GENODE_TEACHER_MAX_TEMPERATURE:-1.0}
GENODE_DENSITY_BIN_COUNT=${GENODE_DENSITY_BIN_COUNT:-128}
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_CONTEXT_HOLDOUT_FRACTION=${GENODE_CONTEXT_HOLDOUT_FRACTION:-0.20}
GENODE_SERIES_HOLDOUT_FRACTION=${GENODE_SERIES_HOLDOUT_FRACTION:-0.20}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_STEPS=${GENODE_STUDENT_STEPS:-500}

CAL_ROOT="${GENODE_VERIFICATION_ROOT}/calibration_inputs"
RUN_DIR="${GENODE_VERIFICATION_ROOT}/policy_runs/${GENODE_RUN_ID}"
mkdir -p "${RUN_DIR}" "${GENODE_VERIFICATION_ROOT}/logs"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

if [[ -f "${RUN_DIR}/gipo_training_summary.json" ]]; then
  if python - "${RUN_DIR}/gipo_training_summary.json" <<'PY'
import json
import sys
summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
raise SystemExit(0 if summary.get("status") == "completed" else 1)
PY
  then
    echo "Policy ${GENODE_RUN_ID} already completed; skipping."
    exit 0
  fi
fi

python -m genode.gipo.train_gipo \
  --rows_csv "${CAL_ROOT}/context_calibration_rows.csv" \
  --context_embeddings_npz "${CAL_ROOT}/context_calibration_embeddings.npz" \
  --schedule_summary_json "${CAL_ROOT}/ser_reference/ser_ptg_schedule_summary.json" \
  --out_dir "${RUN_DIR}" \
  --support_schedule_keys "${GENODE_SUPPORT_KEYS}" \
  --context_sample_count "${GENODE_CONTEXT_SAMPLE_COUNT}" \
  --context_holdout_fraction "${GENODE_CONTEXT_HOLDOUT_FRACTION}" \
  --series_holdout_fraction "${GENODE_SERIES_HOLDOUT_FRACTION}" \
  --density_bin_count "${GENODE_DENSITY_BIN_COUNT}" \
  --teacher_checkpoint_every 100 \
  --teacher_steps "${GENODE_TEACHER_STEPS}" \
  --student_steps "${GENODE_STUDENT_STEPS}" \
  --teacher_temperature_mode "${GENODE_TEMPERATURE_MODE}" \
  --teacher_temperature "${GENODE_TEACHER_TEMPERATURE}" \
  --teacher_target_ess "${GENODE_TEACHER_TARGET_ESS}" \
  --teacher_min_temperature "${GENODE_TEACHER_MIN_TEMPERATURE}" \
  --teacher_max_temperature "${GENODE_TEACHER_MAX_TEMPERATURE}" \
  --series_unknown_dropout 0.10 \
  --seed 0 \
  --device auto
