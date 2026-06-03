#!/usr/bin/env bash
#SBATCH --job-name=genode_stusel_prep
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_STUDENT_SELECTION_ROOT=${GENODE_STUDENT_SELECTION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603}
GENODE_ESS_ROOT=${GENODE_ESS_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
GENODE_DIVERSITY_ROOT=${GENODE_DIVERSITY_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603}

mkdir -p "${GENODE_STUDENT_SELECTION_ROOT}/logs" "${GENODE_STUDENT_SELECTION_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_student_selection_20260603/build_student_selection_inputs.py \
  --out_root "${GENODE_STUDENT_SELECTION_ROOT}" \
  --ess_root "${GENODE_ESS_ROOT}" \
  --diversity_root "${GENODE_DIVERSITY_ROOT}"
