#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_b64final_collect
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:?Set GENODE_REPO to the GenODE-Inference repository path}
GENODE_ENV_DIR=${GENODE_ENV_DIR:?Set GENODE_ENV_DIR to the Python environment path}
GENODE_ROOT=${GENODE_ROOT:?Set GENODE_ROOT to this run output root}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_locked_multiaxis_b64_final_ab_20260607/collect_b64_final_ab_summary.py \
  --root "${GENODE_ROOT}"
