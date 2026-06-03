#!/usr/bin/env bash
#SBATCH --job-name=genode_unseen_in
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
GENODE_FIXED_KEYS=${GENODE_FIXED_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}
GENODE_POLICY_RUN_ID=${GENODE_POLICY_RUN_ID:-temp_fixed_005_b128}

INPUT_ROOT="${GENODE_UNSEEN_ROOT}/inputs"
mkdir -p "${INPUT_ROOT}" "${GENODE_UNSEEN_ROOT}/logs" "${GENODE_UNSEEN_ROOT}/summary"

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
  --seeds "${GENODE_SEEDS}"
  --solver_names "${GENODE_SOLVERS}"
  --target_nfe_values "${GENODE_UNSEEN_NFES}"
  --device auto
)

python -m genode.evaluation.diffusion_flow_time_reparameterization \
  --out_root "${INPUT_ROOT}/fixed_locked_unseen" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --forecast_datasets "${GENODE_DATASET}" \
  --conditional_generation_datasets "" \
  --split_phase locked_test \
  --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
  --train_tuning_sampling_mode validation_normalized \
  --write_forecast_context_rows \
  --forecast_context_row_csv_name fixed_unseen_context_rows.csv \
  --forecast_context_embeddings_npz_name fixed_unseen_context_embeddings.npz \
  --row_csv_name fixed_unseen_rows.csv \
  --row_jsonl_name fixed_unseen_rows.jsonl \
  --allow_execute

mkdir -p "${INPUT_ROOT}/ser_reference_unseen"
python -m genode.gipo.ser_ptg_reference \
  --dataset "${GENODE_DATASET}" \
  --out_dir "${INPUT_ROOT}/ser_reference_unseen" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --reference_split train_tuning \
  --train_tuning_sampling_mode validation_normalized \
  --density_floor_eta 0.05

python -m genode.gipo.evaluate_schedule_summary \
  --dataset "${GENODE_DATASET}" \
  --schedule_summary "${INPUT_ROOT}/ser_reference_unseen/ser_ptg_schedule_summary.json" \
  --split_phase locked_test \
  --out_dir "${INPUT_ROOT}/ser_locked_unseen" \
  "${COMMON_BACKBONE_ARGS[@]}" \
  --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
  --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
  --train_tuning_sampling_mode validation_normalized \
  --write_context_rows \
  --context_row_csv_name ser_unseen_context_rows.csv \
  --context_embeddings_npz_name ser_unseen_context_embeddings.npz \
  --row_csv_name ser_unseen_rows.csv \
  --row_jsonl_name ser_unseen_rows.jsonl

export GENODE_VERIFICATION_ROOT GENODE_UNSEEN_ROOT GENODE_POLICY_RUN_ID
export GENODE_UNSEEN_NFES GENODE_SOLVERS GENODE_SEEDS GENODE_FIXED_KEYS GENODE_DATASET

python - <<'PY'
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from genode.gipo.policy import (
    GIPO_PROTOCOL,
    context_id_from_row,
    load_context_embedding_table,
    save_context_embedding_table,
)

root = Path(os.environ["GENODE_UNSEEN_ROOT"])
verification_root = Path(os.environ["GENODE_VERIFICATION_ROOT"])
inputs = root / "inputs"
policy_run_id = os.environ["GENODE_POLICY_RUN_ID"]
nfes = {int(value) for value in os.environ["GENODE_UNSEEN_NFES"].split(",") if value}
solvers = {value for value in os.environ["GENODE_SOLVERS"].split(",") if value}
seeds = {int(value) for value in os.environ["GENODE_SEEDS"].split(",") if value}
fixed_keys = {value for value in os.environ["GENODE_FIXED_KEYS"].split(",") if value}
ser_key = "ser_ptg_local_defect_eta005"

fixed_context_path = inputs / "fixed_locked_unseen" / "fixed_unseen_context_rows.csv"
ser_context_path = inputs / "ser_locked_unseen" / "ser_unseen_context_rows.csv"
fixed_rows_path = inputs / "fixed_locked_unseen" / "fixed_unseen_rows.csv"
ser_rows_path = inputs / "ser_locked_unseen" / "ser_unseen_rows.csv"
fixed_npz = inputs / "fixed_locked_unseen" / "fixed_unseen_context_embeddings.npz"
ser_npz = inputs / "ser_locked_unseen" / "ser_unseen_context_embeddings.npz"
ser_summary_path = inputs / "ser_reference_unseen" / "ser_ptg_schedule_summary.json"
policy_summary_path = verification_root / "policy_runs" / policy_run_id / "gipo_training_summary.json"
student_path = verification_root / "policy_runs" / policy_run_id / "gipo_student.pt"


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


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
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_context_rows(label, rows, expected_schedules):
    if not rows:
        raise ValueError(f"{label} context rows are empty.")
    split_values = {str(row.get("split_phase", row.get("split", ""))) for row in rows}
    if split_values != {"locked_test"}:
        raise ValueError(f"{label} rows must be locked_test only, got {sorted(split_values)}.")
    row_nfes = {int(row["target_nfe"]) for row in rows}
    if row_nfes != nfes:
        raise ValueError(f"{label} rows have target NFEs {sorted(row_nfes)}, expected {sorted(nfes)}.")
    row_solvers = {str(row["solver_key"]) for row in rows}
    if row_solvers != solvers:
        raise ValueError(f"{label} rows have solvers {sorted(row_solvers)}, expected {sorted(solvers)}.")
    row_schedules = {str(row["scheduler_key"]) for row in rows}
    if row_schedules != expected_schedules:
        raise ValueError(f"{label} rows have schedules {sorted(row_schedules)}, expected {sorted(expected_schedules)}.")


def validate_aggregate_rows(label, rows, expected_schedules):
    validate_context_rows(label, rows, expected_schedules)
    cells = {
        (int(row["seed"]), str(row["solver_key"]), int(row["target_nfe"]), str(row["scheduler_key"]))
        for row in rows
    }
    expected = {
        (seed, solver, nfe, schedule)
        for seed in seeds
        for solver in solvers
        for nfe in nfes
        for schedule in expected_schedules
    }
    missing = sorted(expected - cells)
    extra = sorted(cells - expected)
    if missing or extra:
        raise ValueError(f"{label} aggregate coverage mismatch: missing={missing[:8]} extra={extra[:8]}.")


def merge_embeddings(paths):
    merged = {}
    for path in paths:
        table = load_context_embedding_table(path)
        for key, value in table.items():
            arr = np.asarray(value, dtype=np.float32)
            if key in merged and not np.allclose(merged[key], arr, atol=1e-5):
                raise ValueError(f"Conflicting embedding for {key} from {path}.")
            merged[key] = arr
    return merged


fixed_context_rows = read_rows(fixed_context_path)
ser_context_rows = read_rows(ser_context_path)
fixed_rows = read_rows(fixed_rows_path)
ser_rows = read_rows(ser_rows_path)
validate_context_rows("fixed locked context", fixed_context_rows, fixed_keys)
validate_context_rows("SER locked context", ser_context_rows, {ser_key})
validate_aggregate_rows("fixed locked aggregate", fixed_rows, fixed_keys)
validate_aggregate_rows("SER locked aggregate", ser_rows, {ser_key})

ser_summary = json.loads(ser_summary_path.read_text(encoding="utf-8"))
if str(ser_summary.get("reference_split")) != "train_tuning":
    raise ValueError(f"SER reference must be train_tuning, got {ser_summary.get('reference_split')!r}.")
if {int(value) for value in ser_summary.get("target_nfe_values", [])} != nfes:
    raise ValueError("SER reference NFE coverage does not match unseen NFEs.")

policy_summary = json.loads(policy_summary_path.read_text(encoding="utf-8"))
if str(policy_summary.get("protocol")) != GIPO_PROTOCOL:
    raise ValueError("Policy summary is not GIPO.")
if bool(policy_summary.get("locked_test_used_for_selection", False)):
    raise ValueError("Policy summary indicates locked-test selection.")
checkpoint = torch.load(student_path, map_location="cpu")
if bool(checkpoint.get("locked_test_used_for_selection", False)):
    raise ValueError("Student checkpoint indicates locked-test selection.")

embeddings = merge_embeddings([fixed_npz, ser_npz])
combined_context_rows = fixed_context_rows + ser_context_rows
missing_embeddings = sorted({context_id_from_row(row) for row in combined_context_rows} - set(embeddings))
if missing_embeddings:
    raise KeyError(f"Merged unseen embeddings are missing contexts: {missing_embeddings[:8]}")

write_rows(inputs / "locked_context_rows.csv", combined_context_rows)
save_context_embedding_table(inputs / "locked_context_embeddings.npz", embeddings)

manifest = {
    "artifact": "unseen_nfe_locked_inputs_manifest",
    "dataset": os.environ["GENODE_DATASET"],
    "policy_run_id": policy_run_id,
    "policy_summary": str(policy_summary_path),
    "student_checkpoint": str(student_path),
    "target_nfe_values": sorted(nfes),
    "solver_names": sorted(solvers),
    "seeds": sorted(seeds),
    "locked_test_only_inputs": True,
    "locked_test_used_for_selection": False,
    "ser_reference_split": str(ser_summary.get("reference_split")),
    "fixed_locked_rows": str(fixed_rows_path),
    "ser_locked_rows": str(ser_rows_path),
    "locked_context_rows": str(inputs / "locked_context_rows.csv"),
    "locked_context_embeddings": str(inputs / "locked_context_embeddings.npz"),
    "fixed_context_row_count": len(fixed_context_rows),
    "ser_context_row_count": len(ser_context_rows),
    "combined_context_row_count": len(combined_context_rows),
    "locked_context_count": len({context_id_from_row(row) for row in combined_context_rows}),
    "fixed_aggregate_row_count": len(fixed_rows),
    "ser_aggregate_row_count": len(ser_rows),
    "fixed_schedule_counts": dict(Counter(row["scheduler_key"] for row in fixed_context_rows)),
    "ser_schedule_counts": dict(Counter(row["scheduler_key"] for row in ser_context_rows)),
}
(inputs / "unseen_nfe_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
