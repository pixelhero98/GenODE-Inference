#!/usr/bin/env bash
#SBATCH --job-name=genode_seen_collect
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_seen_ablation_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_SEEN_ROOT=${GENODE_SEEN_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_seen_ablation_20260606}
GENODE_VECTOR_ROOT=${GENODE_VECTOR_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_vector_teacher_20260605}

mkdir -p "${GENODE_SEEN_ROOT}/logs" "${GENODE_SEEN_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_gipo_seen_ablation_20260606/collect_seen_summary.py \
  --root "${GENODE_SEEN_ROOT}" \
  --vector-root "${GENODE_VECTOR_ROOT}"
