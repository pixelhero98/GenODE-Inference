#!/usr/bin/env bash
#SBATCH --job-name=genode_unseen_rep
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/unseen_nfe_locked_winner_6_10_14_16/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/unseen_nfe_locked_winner_6_10_14_16/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
GENODE_UNSEEN_ROOT=${GENODE_UNSEEN_ROOT:-${GENODE_VERIFICATION_ROOT}/unseen_nfe_locked_winner_6_10_14_16}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}
GENODE_POLICY_RUN_ID=${GENODE_POLICY_RUN_ID:-temp_fixed_005_b128}

INPUT_ROOT="${GENODE_UNSEEN_ROOT}/inputs"
REPORT_ROOT="${GENODE_UNSEEN_ROOT}/report/${GENODE_POLICY_RUN_ID}"
POLICY_ROOT="${GENODE_VERIFICATION_ROOT}/policy_runs/${GENODE_POLICY_RUN_ID}"
mkdir -p "${REPORT_ROOT}" "${GENODE_UNSEEN_ROOT}/logs"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

if [[ -f "${REPORT_ROOT}/locked_test_gipo_policy_summary.json" ]]; then
  echo "Unseen-NFE report already exists for ${GENODE_POLICY_RUN_ID}; skipping."
  exit 0
fi

python -m genode.gipo.report_locked_test \
  --gipo_student_checkpoint "${POLICY_ROOT}/gipo_student.pt" \
  --training_summary "${POLICY_ROOT}/gipo_training_summary.json" \
  --context_rows "${INPUT_ROOT}/locked_context_rows.csv" \
  --context_embeddings_npz "${INPUT_ROOT}/locked_context_embeddings.npz" \
  --out_dir "${REPORT_ROOT}" \
  --baseline_rows "${INPUT_ROOT}/fixed_locked_unseen/fixed_unseen_rows.csv" \
  --comparator_rows "${INPUT_ROOT}/ser_locked_unseen/ser_unseen_rows.csv" \
  --dataset "${GENODE_DATASET}" \
  --dataset_root "${GENODE_DATASET_ROOT}" \
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
  --output_root "${GENODE_UNSEEN_ROOT}" \
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
  --device auto \
  --seeds "${GENODE_SEEDS}" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_UNSEEN_NFES}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}"
