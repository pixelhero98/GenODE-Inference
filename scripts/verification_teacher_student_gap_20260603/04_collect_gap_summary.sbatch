#!/usr/bin/env bash
#SBATCH --job-name=genode_gap_collect
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/gipo_gap_src_20260603}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_GAP_ROOT=${GENODE_GAP_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_teacher_student_gap_20260603}
GENODE_REFERENCE_ROOT=${GENODE_REFERENCE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}

mkdir -p "${GENODE_GAP_ROOT}/logs" "${GENODE_GAP_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_teacher_student_gap_20260603/collect_gap_summary.py \
  --root "${GENODE_GAP_ROOT}" \
  --reference_root "${GENODE_REFERENCE_ROOT}"
