#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_ckpt_collect
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_locked_multiaxis_b64_checkpoint_maturity_src_20260608}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_SUMMARY_ROOT=${GENODE_SUMMARY_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608}
GENODE_CKPT16_ROOT=${GENODE_CKPT16_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ckpt16k_20260608}
GENODE_CKPT12_ROOT=${GENODE_CKPT12_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ckpt12k_20260608}
GENODE_FINAL20_ROOT=${GENODE_FINAL20_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_final_ab_20260607}

mkdir -p "${GENODE_SUMMARY_ROOT}/logs" "${GENODE_SUMMARY_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/validate_locked_artifacts.py \
  --root "${GENODE_CKPT16_ROOT}" \
  --validate
python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/validate_locked_artifacts.py \
  --root "${GENODE_CKPT12_ROOT}" \
  --validate
python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/collect_checkpoint_maturity_summary.py \
  --summary_root "${GENODE_SUMMARY_ROOT}" \
  --ckpt16_root "${GENODE_CKPT16_ROOT}" \
  --ckpt12_root "${GENODE_CKPT12_ROOT}" \
  --final20_root "${GENODE_FINAL20_ROOT}"
