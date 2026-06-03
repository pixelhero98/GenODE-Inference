#!/usr/bin/env bash
#SBATCH --job-name=genode_div_unseen_submit
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_density_diversity_20260603}
GENODE_UNSEEN_ROOT=${GENODE_UNSEEN_ROOT:-${GENODE_VERIFICATION_ROOT}/unseen_nfe_locked_winner_6_10_14_16}
GENODE_GPU_PARTITION=${GENODE_GPU_PARTITION:-ampere}
GENODE_CPU_PARTITION=${GENODE_CPU_PARTITION:-genoa}

cd "${GENODE_REPO}"
export GENODE_VERIFICATION_ROOT GENODE_UNSEEN_ROOT GENODE_GPU_PARTITION GENODE_CPU_PARTITION

bash "${GENODE_VERIFICATION_ROOT}/jobs/submit_unseen_winner.sh"
