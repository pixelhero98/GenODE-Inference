#!/usr/bin/env bash
#SBATCH --job-name=genode_bal_seen_collect
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_balanced_seen_multinfe_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_BALANCED_SEEN_ROOT=${GENODE_BALANCED_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606}

mkdir -p "${GENODE_BALANCED_SEEN_ROOT}/logs" "${GENODE_BALANCED_SEEN_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_gipo_balanced_seen_multinfe_20260606/collect_balanced_seen_summary.py \
  --root "${GENODE_BALANCED_SEEN_ROOT}" \
  --mode final
