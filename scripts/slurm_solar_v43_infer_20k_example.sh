#!/bin/bash
# Generic Slurm example for V4.3 pooled solar inference with the 20k backbone.
# Override GENODE_* variables for your cluster layout before submitting.
#SBATCH --job-name=genode-solar-v43-20k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail

if [ -n "${GENODE_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
  for genode_module in ${GENODE_MODULES}; do
    module load "${genode_module}"
  done
fi

REPO="${GENODE_REPO:-$PWD}"
ENV_DIR="${GENODE_ENV_DIR:-${REPO}/.venv}"
DATASET_ROOT="${GENODE_DATASET_ROOT:-${REPO}/paper_datasets}"
OUTPUT_ROOT="${GENODE_OUTPUT_ROOT:-${REPO}/outputs}"
LOG_ROOT="${GENODE_LOG_ROOT:-${OUTPUT_ROOT}/logs}"
BACKBONE_MANIFEST="${GENODE_BACKBONE_MANIFEST:-outputs/backbone_matrix/backbone_manifest.json}"
OUT_DIR="${GENODE_INFERENCE_OUT_DIR:-outputs/train20_v43_solar_20k}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${REPO}"
if [ ! -e outputs ]; then
  ln -s "${OUTPUT_ROOT}" outputs
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "Python environment not found at ${ENV_DIR}. Run the training example first or set GENODE_ENV_DIR." >&2
  exit 1
fi

source "${ENV_DIR}/bin/activate"

python -m genode.conditional_opd.train20_v43_pooled_calibration \
  --dataset "${GENODE_DATASET:-solar_energy_10m}" \
  --otflow_train_steps "${GENODE_BACKBONE_STEPS:-20000}" \
  --backbone_manifest "${BACKBONE_MANIFEST}" \
  --dataset_root "${DATASET_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --device "${GENODE_DEVICE:-auto}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}" \
  --skip_locked_test \
  --allow_execute
