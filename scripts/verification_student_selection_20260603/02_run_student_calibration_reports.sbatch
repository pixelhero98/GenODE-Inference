#!/usr/bin/env bash
#SBATCH --job-name=genode_stusel_cal
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_STUDENT_SELECTION_ROOT=${GENODE_STUDENT_SELECTION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_student_selection_20260603}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_NFES=${GENODE_NFES:-4,8,12}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

mkdir -p "${GENODE_STUDENT_SELECTION_ROOT}/logs" "${GENODE_STUDENT_SELECTION_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python scripts/verification_student_selection_20260603/run_student_calibration_reports.py \
  --root "${GENODE_STUDENT_SELECTION_ROOT}" \
  --dataset "${GENODE_DATASET}" \
  --dataset_root "${GENODE_DATASET_ROOT}" \
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
  --device auto \
  --seeds "${GENODE_SEEDS}" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_NFES}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}"
