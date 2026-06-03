#!/usr/bin/env bash
#SBATCH --job-name=genode_ess_merge
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1
export GENODE_VERIFICATION_ROOT

python scripts/verification_ess_bins_20260602/merge_calibration_inputs.py --root "${GENODE_VERIFICATION_ROOT}"
