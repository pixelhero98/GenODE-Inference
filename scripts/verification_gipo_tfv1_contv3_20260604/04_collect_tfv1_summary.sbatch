#!/usr/bin/env bash
#SBATCH --job-name=genode_tfv1_collect
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_tfv1_contv3_20260604/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_tfv1_contv3_20260604/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_tfv1_src_20260604}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_TFV1_ROOT=${GENODE_TFV1_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_tfv1_contv3_20260604}

mkdir -p "${GENODE_TFV1_ROOT}/logs" "${GENODE_TFV1_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_gipo_tfv1_contv3_20260604/collect_tfv1_summary.py \
  --root "${GENODE_TFV1_ROOT}"
