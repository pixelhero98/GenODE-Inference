#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_b64opt_prepare
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_optim_20260607/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_optim_20260607/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_locked_multiaxis_b64_optim_src_20260607}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_optim_20260607}
GENODE_SOURCE_ROOT=${GENODE_SOURCE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/summary"
ln -sfn "${GENODE_SOURCE_ROOT}/artifacts" "${GENODE_ROOT}/artifacts"
ln -sfn "${GENODE_SOURCE_ROOT}/standard_inputs" "${GENODE_ROOT}/standard_inputs"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_additive_locked_multiaxis_20260606/validate_additive_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --validate
