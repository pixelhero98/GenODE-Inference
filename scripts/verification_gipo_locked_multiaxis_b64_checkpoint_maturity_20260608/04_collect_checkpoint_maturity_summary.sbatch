#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_ckpt_collect
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:?Set GENODE_REPO to the GenODE-Inference repository path}
GENODE_ENV_DIR=${GENODE_ENV_DIR:?Set GENODE_ENV_DIR to the Python environment path}
GENODE_SUMMARY_ROOT=${GENODE_SUMMARY_ROOT:?Set GENODE_SUMMARY_ROOT to the maturity summary root}
GENODE_MATURITY_CANDIDATES=${GENODE_MATURITY_CANDIDATES:?Set GENODE_MATURITY_CANDIDATES as semicolon-separated label,root,run_id,train_steps entries}
GENODE_COMPARATOR_ROOT=${GENODE_COMPARATOR_ROOT:?Set GENODE_COMPARATOR_ROOT to the additive comparator root}
GENODE_COMPARATOR_RUN_ID=${GENODE_COMPARATOR_RUN_ID:?Set GENODE_COMPARATOR_RUN_ID to the additive comparator run id}
GENODE_COMPARATOR_LABEL=${GENODE_COMPARATOR_LABEL:-comparator}

mkdir -p "${GENODE_SUMMARY_ROOT}/logs" "${GENODE_SUMMARY_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts

candidate_args=()
IFS=';' read -r -a candidates <<< "${GENODE_MATURITY_CANDIDATES}"
for candidate in "${candidates[@]}"; do
  [[ -n "${candidate}" ]] || continue
  IFS=',' read -r label root run_id train_steps extra <<< "${candidate}"
  if [[ -n "${extra:-}" || -z "${label:-}" || -z "${root:-}" || -z "${run_id:-}" || -z "${train_steps:-}" ]]; then
    echo "Invalid candidate '${candidate}'. Expected label,root,run_id,train_steps." >&2
    exit 2
  fi
  python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/validate_locked_artifacts.py \
    --root "${root}" \
    --validate
  candidate_args+=(--candidate "${candidate}")
done

python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/collect_checkpoint_maturity_summary.py \
  --summary_root "${GENODE_SUMMARY_ROOT}" \
  "${candidate_args[@]}" \
  --comparator_root "${GENODE_COMPARATOR_ROOT}" \
  --comparator_run_id "${GENODE_COMPARATOR_RUN_ID}" \
  --comparator_label "${GENODE_COMPARATOR_LABEL}"
