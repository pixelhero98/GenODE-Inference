#!/usr/bin/env bash
#SBATCH --job-name=genode_bal_seen_inputs
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_balanced_seen_multinfe_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_balanced_seen_multinfe_20260606}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:-20000}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_SEEN_NFES=${GENODE_SEEN_NFES:-4,8,12}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_FIXED_KEYS=${GENODE_FIXED_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}

CAL_ROOT="${GENODE_ROOT}/calibration_inputs"
STD_ROOT="${GENODE_ROOT}/standard_inputs"
FIXED_TRAIN_ROOT="${STD_ROOT}/fixed_train_tuning"
SER_REF_ROOT="${STD_ROOT}/ser_reference"
SER_TRAIN_ROOT="${STD_ROOT}/ser_train_tuning"
FIXED_LOCKED_ROOT="${STD_ROOT}/fixed_locked"
SER_LOCKED_ROOT="${STD_ROOT}/ser_locked"

mkdir -p \
  "${GENODE_ROOT}/logs" \
  "${GENODE_ROOT}/summary" \
  "${CAL_ROOT}" \
  "${FIXED_TRAIN_ROOT}" \
  "${SER_REF_ROOT}" \
  "${SER_TRAIN_ROOT}" \
  "${FIXED_LOCKED_ROOT}" \
  "${SER_LOCKED_ROOT}"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

COMMON_BACKBONE_ARGS=(
  --dataset_root "${GENODE_DATASET_ROOT}"
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}"
  --device auto
)

python -m genode.evaluation.diffusion_flow_time_reparameterization \
  --out_root "${FIXED_TRAIN_ROOT}" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --forecast_datasets "${GENODE_DATASET}" \
  --conditional_generation_datasets "" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_SEEN_NFES}" \
  --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
  --seeds "${GENODE_SEEDS}" \
  --split_phase train_tuning \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
  --write_forecast_context_rows \
  --forecast_context_row_csv_name fixed_train_context_rows.csv \
  --forecast_context_embeddings_npz_name fixed_train_context_embeddings.npz \
  --row_csv_name fixed_train_rows.csv \
  --row_jsonl_name fixed_train_rows.jsonl \
  --allow_execute \
  --no_resume

python -m genode.gipo.ser_ptg_reference \
  --dataset "${GENODE_DATASET}" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --seeds "${GENODE_SEEDS}" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_SEEN_NFES}" \
  --reference_split train_tuning \
  --out_dir "${SER_REF_ROOT}"

SER_SUMMARY="${SER_REF_ROOT}/ser_ptg_schedule_summary.json"

python -m genode.gipo.evaluate_schedule_summary \
  --schedule_summary "${SER_SUMMARY}" \
  --split_phase train_tuning \
  --out_dir "${SER_TRAIN_ROOT}" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --dataset "${GENODE_DATASET}" \
  --seeds "${GENODE_SEEDS}" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_SEEN_NFES}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
  --write_context_rows \
  --context_row_csv_name ser_train_context_rows.csv \
  --context_embeddings_npz_name ser_train_context_embeddings.npz \
  --row_csv_name ser_train_rows.csv \
  --row_jsonl_name ser_train_rows.jsonl \
  --summary_output_name ser_train_schedule_summary.json

python -m genode.evaluation.diffusion_flow_time_reparameterization \
  --out_root "${FIXED_LOCKED_ROOT}" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --forecast_datasets "${GENODE_DATASET}" \
  --conditional_generation_datasets "" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_SEEN_NFES}" \
  --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
  --seeds "${GENODE_SEEDS}" \
  --split_phase locked_test \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
  --write_forecast_context_rows \
  --forecast_context_row_csv_name fixed_locked_context_rows.csv \
  --forecast_context_embeddings_npz_name fixed_locked_context_embeddings.npz \
  --row_csv_name fixed_locked_rows.csv \
  --row_jsonl_name fixed_locked_rows.jsonl \
  --allow_execute \
  --no_resume

python -m genode.gipo.evaluate_schedule_summary \
  --schedule_summary "${SER_SUMMARY}" \
  --split_phase locked_test \
  --out_dir "${SER_LOCKED_ROOT}" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --dataset "${GENODE_DATASET}" \
  --seeds "${GENODE_SEEDS}" \
  --solver_names "${GENODE_SOLVERS}" \
  --target_nfe_values "${GENODE_SEEN_NFES}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
  --write_context_rows \
  --context_row_csv_name ser_locked_context_rows.csv \
  --context_embeddings_npz_name ser_locked_context_embeddings.npz \
  --row_csv_name ser_locked_rows.csv \
  --row_jsonl_name ser_locked_rows.jsonl \
  --summary_output_name ser_locked_schedule_summary.json \
  --baseline_rows "${FIXED_LOCKED_ROOT}/fixed_locked_context_rows.csv"

GENODE_ROOT="${GENODE_ROOT}" python - <<'PY'
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["GENODE_ROOT"])
cal = root / "calibration_inputs"
fixed_train = root / "standard_inputs" / "fixed_train_tuning" / "fixed_train_context_rows.csv"
ser_train = root / "standard_inputs" / "ser_train_tuning" / "ser_train_context_rows.csv"
fixed_locked = root / "standard_inputs" / "fixed_locked" / "fixed_locked_context_rows.csv"
fixed_locked_embeddings = root / "standard_inputs" / "fixed_locked" / "fixed_locked_context_embeddings.npz"
out_train = cal / "context_calibration_rows.csv"

def read_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def write_rows(path, rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

rows = read_rows(fixed_train) + read_rows(ser_train)
write_rows(out_train, rows)
shutil.copy2(root / "standard_inputs" / "fixed_train_tuning" / "fixed_train_context_embeddings.npz", cal / "context_calibration_embeddings.npz")
shutil.copy2(fixed_locked, cal / "locked_context_rows.csv")
shutil.copy2(fixed_locked_embeddings, cal / "locked_context_embeddings.npz")

def key(row):
    return (
        row.get("dataset"),
        row.get("split_phase"),
        row.get("seed"),
        row.get("solver_key"),
        row.get("context_id"),
        row.get("series_id"),
    )

groups = defaultdict(set)
for row in rows:
    groups[key(row)].add(int(row["target_nfe"]))
summary = {
    "artifact": "balanced_seen_input_manifest",
    "context_calibration_rows": str(out_train),
    "context_calibration_row_count": len(rows),
    "logical_seed_rows": sum(1 for row in rows if str(row.get("logical_seed", "")).strip() != ""),
    "evaluation_seed_rows": sum(1 for row in rows if str(row.get("evaluation_seed", "")).strip() != ""),
    "multi_nfe_group_count": sum(1 for values in groups.values() if len(values) > 1),
}
counts = defaultdict(int)
for values in groups.values():
    counts[tuple(sorted(values))] += 1
summary["nfe_sets_top"] = [
    {"nfes": list(values), "count": count}
    for values, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
]
if summary["multi_nfe_group_count"] <= 0:
    raise SystemExit("No multi-NFE context groups were generated.")
(root / "summary" / "input_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
