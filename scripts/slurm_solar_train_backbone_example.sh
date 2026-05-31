#!/bin/bash
# Generic Slurm example for training solar forecast backbones.
# Override GENODE_* variables for your cluster layout before submitting.
#SBATCH --job-name=genode-solar-backbone
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
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${OUTPUT_ROOT}/pip-cache}"

mkdir -p "${ENV_DIR}" "${DATASET_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}" "${PIP_CACHE_DIR}"
cd "${REPO}"
if [ ! -e outputs ]; then
  ln -s "${OUTPUT_ROOT}" outputs
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  python3 -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python -m genode.training.train_backbone \
  --dataset "${GENODE_DATASET:-solar_energy_10m}" \
  --dataset_root "${DATASET_ROOT}" \
  --steps "${GENODE_TRAIN_STEPS:-20000}" \
  --checkpoint_steps "${GENODE_CHECKPOINT_STEPS:-4000,8000,12000,16000,20000}" \
  --val_every "${GENODE_VAL_EVERY:-200}" \
  --batch_size "${GENODE_BATCH_SIZE:-64}" \
  --device "${GENODE_DEVICE:-auto}" \
  --prepare_data
