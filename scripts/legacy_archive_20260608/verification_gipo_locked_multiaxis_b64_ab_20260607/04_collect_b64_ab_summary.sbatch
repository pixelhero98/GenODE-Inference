#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_b64_collect
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ab_20260607/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ab_20260607/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_locked_multiaxis_b64_ab_src_20260607}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ab_20260607}
GENODE_ADDITIVE128_ROOT=${GENODE_ADDITIVE128_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606}
GENODE_ADALN128_ROOT=${GENODE_ADALN128_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_adaln_locked_multiaxis_ab_20260607}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_locked_multiaxis_b64_ab_20260607/collect_b64_ab_summary.py \
  --root "${GENODE_ROOT}" \
  --additive128_root "${GENODE_ADDITIVE128_ROOT}" \
  --adaln128_root "${GENODE_ADALN128_ROOT}"
