#!/usr/bin/env bash
#SBATCH --job-name=genode_div_sum
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603}

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python "${GENODE_VERIFICATION_ROOT}/jobs/collect_density_diversity_summary.py"
