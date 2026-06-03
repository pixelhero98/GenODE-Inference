#!/usr/bin/env bash
#SBATCH --job-name=genode_ess_lock
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/home/b35z/pixelhero.b35z/work/GenODE-Inference}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_VERIFICATION_ROOT=${GENODE_VERIFICATION_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_ess_bins_20260602}
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
GENODE_RUN_ID=${GENODE_RUN_ID:?set GENODE_RUN_ID}

CAL_ROOT="${GENODE_VERIFICATION_ROOT}/calibration_inputs"
RUN_DIR="${GENODE_VERIFICATION_ROOT}/policy_runs/${GENODE_RUN_ID}"
REPORT_DIR="${GENODE_VERIFICATION_ROOT}/locked_test_reports/${GENODE_RUN_ID}"
mkdir -p "${REPORT_DIR}" "${GENODE_VERIFICATION_ROOT}/logs"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

if [[ -f "${REPORT_DIR}/locked_test_gipo_policy_summary.json" ]]; then
  echo "Locked-test report ${GENODE_RUN_ID} already exists; skipping."
  exit 0
fi

python -m genode.gipo.report_locked_test \
  --gipo_student_checkpoint "${RUN_DIR}/gipo_student.pt" \
  --training_summary "${RUN_DIR}/gipo_training_summary.json" \
  --context_rows "${CAL_ROOT}/fixed_locked_test/fixed_locked_context_rows.csv,${CAL_ROOT}/ser_locked_test/ser_locked_context_rows.csv" \
  --context_embeddings_npz "${CAL_ROOT}/locked_context_embeddings.npz" \
  --out_dir "${REPORT_DIR}" \
  --baseline_rows "${CAL_ROOT}/fixed_locked_test/fixed_locked_rows.csv" \
  --comparator_rows "${CAL_ROOT}/ser_locked_test/ser_locked_rows.csv" \
  --dataset "${GENODE_DATASET}" \
  --dataset_root "${GENODE_DATASET_ROOT}" \
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}" \
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}" \
  --output_root "${GENODE_VERIFICATION_ROOT}" \
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}" \
  --device auto \
  --seeds "${GENODE_SEEDS}" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_NFES}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}"
