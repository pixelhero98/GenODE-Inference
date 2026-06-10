#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENODE_SUMMARY_ROOT=${GENODE_SUMMARY_ROOT:?Set GENODE_SUMMARY_ROOT to the maturity summary output root}
GENODE_COMPARATOR_ROOT=${GENODE_COMPARATOR_ROOT:?Set GENODE_COMPARATOR_ROOT to the additive comparator output root}
GENODE_COMPARATOR_RUN_ID=${GENODE_COMPARATOR_RUN_ID:?Set GENODE_COMPARATOR_RUN_ID to the additive comparator run id}
GENODE_COMPARATOR_LABEL=${GENODE_COMPARATOR_LABEL:-comparator}
GENODE_MATURITY_SPECS=${GENODE_MATURITY_SPECS:?Set GENODE_MATURITY_SPECS as semicolon-separated label,train_steps,root,run_id,report_label_prefix specs}

mkdir -p "${GENODE_SUMMARY_ROOT}/logs"

artifact_jobs=()
train_jobs=()
report_jobs=()
candidate_specs=()

IFS=';' read -r -a specs <<< "${GENODE_MATURITY_SPECS}"
for spec in "${specs[@]}"; do
  [[ -n "${spec}" ]] || continue
  IFS=',' read -r label train_steps root run_id report_label_prefix extra <<< "${spec}"
  if [[ -n "${extra:-}" || -z "${label:-}" || -z "${train_steps:-}" || -z "${root:-}" || -z "${run_id:-}" || -z "${report_label_prefix:-}" ]]; then
    echo "Invalid maturity spec '${spec}'. Expected label,train_steps,root,run_id,report_label_prefix." >&2
    exit 2
  fi
  export_payload=ALL,GENODE_OTFLOW_TRAIN_STEPS=${train_steps},GENODE_BUDGET_LABEL=${label},GENODE_ROOT=${root},RUN_ID=${run_id},REPORT_LABEL_PREFIX=${report_label_prefix}
  artifact_job=$(sbatch --parsable --export="${export_payload}" "${SCRIPT_DIR}/01_generate_artifacts.sbatch")
  train_job=$(sbatch --parsable --dependency=afterok:${artifact_job} --export="${export_payload}" "${SCRIPT_DIR}/02_train_additive.sbatch")
  report_job=$(sbatch --parsable --dependency=afterok:${train_job} --export="${export_payload}" "${SCRIPT_DIR}/03_report_additive_seen_unseen.sbatch")
  artifact_jobs+=("${label}:${artifact_job}")
  train_jobs+=("${label}:${train_job}")
  report_jobs+=("${label}:${report_job}")
  candidate_specs+=("${label},${root},${run_id},${train_steps}")
done

if [[ ${#candidate_specs[@]} -eq 0 ]]; then
  echo "GENODE_MATURITY_SPECS did not contain any candidates." >&2
  exit 2
fi

report_dependency=$(IFS=:; echo "${report_jobs[*]##*:}")
maturity_candidates=$(IFS=';'; echo "${candidate_specs[*]}")
collect_job=$(sbatch --parsable \
  --dependency=afterok:${report_dependency} \
  --export=ALL,GENODE_SUMMARY_ROOT="${GENODE_SUMMARY_ROOT}",GENODE_MATURITY_CANDIDATES="${maturity_candidates}",GENODE_COMPARATOR_ROOT="${GENODE_COMPARATOR_ROOT}",GENODE_COMPARATOR_RUN_ID="${GENODE_COMPARATOR_RUN_ID}",GENODE_COMPARATOR_LABEL="${GENODE_COMPARATOR_LABEL}" \
  "${SCRIPT_DIR}/04_collect_checkpoint_maturity_summary.sbatch")

printf 'artifact_jobs=%s\ntrain_jobs=%s\nreport_jobs=%s\ncollect_job=%s\n' \
  "$(IFS=,; echo "${artifact_jobs[*]}")" \
  "$(IFS=,; echo "${train_jobs[*]}")" \
  "$(IFS=,; echo "${report_jobs[*]}")" \
  "${collect_job}"
