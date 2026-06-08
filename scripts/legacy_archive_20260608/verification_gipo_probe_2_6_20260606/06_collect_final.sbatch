#!/usr/bin/env bash
#SBATCH --job-name=genode_probe_collect_final
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_density_form_src_20260604}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_INPUT_ROOT=${GENODE_INPUT_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_density_form_20260604}
GENODE_VECTOR_ROOT=${GENODE_VECTOR_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_vector_teacher_20260605}
GENODE_PROBE_ROOT=${GENODE_PROBE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_probe_2_6_20260606}
GENODE_SOURCE_COMMIT=${GENODE_SOURCE_COMMIT:-a37bc9f70507d152a7ee3a5e96baf18a8d4f3240}
GENODE_COLLECT_MODE=${GENODE_COLLECT_MODE:-final}

mkdir -p "${GENODE_PROBE_ROOT}/logs" "${GENODE_PROBE_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_gipo_probe_2_6_20260606/collect_probe_summary.py \
  --mode "${GENODE_COLLECT_MODE}" \
  --root "${GENODE_PROBE_ROOT}" \
  --input-root "${GENODE_INPUT_ROOT}" \
  --vector-root "${GENODE_VECTOR_ROOT}" \
  --source-commit "${GENODE_SOURCE_COMMIT}"
