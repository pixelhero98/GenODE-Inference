#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_ckpt_artifacts
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs/%x.%j.err

set -euo pipefail

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_locked_multiaxis_b64_checkpoint_maturity_src_20260608}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_DATASET_ROOT=${GENODE_DATASET_ROOT:-/projects/b35z/genode/paper_datasets}
GENODE_BACKBONE_MANIFEST=${GENODE_BACKBONE_MANIFEST:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json}
GENODE_SHARED_BACKBONE_ROOT=${GENODE_SHARED_BACKBONE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:?Set GENODE_OTFLOW_TRAIN_STEPS to 16000 or 12000}
GENODE_BUDGET_LABEL=${GENODE_BUDGET_LABEL:-$((GENODE_OTFLOW_TRAIN_STEPS / 1000))k}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_ckpt${GENODE_BUDGET_LABEL}_20260608}
GENODE_DATASET=${GENODE_DATASET:-solar_energy_10m}
GENODE_SOLVERS=${GENODE_SOLVERS:-euler,heun,midpoint_rk2,dpmpp2m}
GENODE_SEEN_NFES=${GENODE_SEEN_NFES:-4,8,12}
GENODE_UNSEEN_NFES=${GENODE_UNSEEN_NFES:-6,10,14,16}
GENODE_SEEDS=${GENODE_SEEDS:-0,1,2}
GENODE_FIXED_KEYS=${GENODE_FIXED_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,late_power_3_reversed,flowts_power_sampling_reversed,ays_reversed,gits_reversed,ots_reversed}
GENODE_NUM_EVAL_SAMPLES=${GENODE_NUM_EVAL_SAMPLES:-5}
GENODE_FORECAST_EVAL_BATCH_SIZE=${GENODE_FORECAST_EVAL_BATCH_SIZE:-64}
GENODE_TRAIN_TUNING_FRACTION=${GENODE_TRAIN_TUNING_FRACTION:-0.0001}
GENODE_TRAIN_TUNING_STRATA=${GENODE_TRAIN_TUNING_STRATA:-32}
GENODE_TRAIN_TUNING_SAMPLING_MODE=${GENODE_TRAIN_TUNING_SAMPLING_MODE:-train_window_fraction}
GENODE_LOCKED_EVAL_WINDOWS=${GENODE_LOCKED_EVAL_WINDOWS:-32}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/summary" "${GENODE_ROOT}/artifacts" "${GENODE_ROOT}/standard_inputs"
mkdir -p /scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/logs

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces

GENODE_BACKBONE_MANIFEST="${GENODE_BACKBONE_MANIFEST}" \
GENODE_DATASET="${GENODE_DATASET}" \
GENODE_OTFLOW_TRAIN_STEPS="${GENODE_OTFLOW_TRAIN_STEPS}" \
python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

manifest = Path(os.environ["GENODE_BACKBONE_MANIFEST"])
dataset = os.environ["GENODE_DATASET"]
steps = int(os.environ["GENODE_OTFLOW_TRAIN_STEPS"])
payload = json.loads(manifest.read_text(encoding="utf-8"))

def walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)

candidates = [
    item for item in walk(payload)
    if str(item.get("dataset_key", item.get("dataset", ""))) == dataset
    and int(item.get("train_steps", item.get("train_step", 0)) or 0) == steps
    and "forecast" in str(item.get("benchmark_family", item.get("family", "forecast_extrapolation")))
]
if not candidates:
    raise SystemExit(f"Missing forecast backbone manifest entry for {dataset} at {steps} steps.")
print(json.dumps({"dataset": dataset, "train_steps": steps, "candidate_count": len(candidates)}, sort_keys=True))
PY

COMMON_BACKBONE_ARGS=(
  --dataset_root "${GENODE_DATASET_ROOT}"
  --shared_backbone_root "${GENODE_SHARED_BACKBONE_ROOT}"
  --backbone_manifest "${GENODE_BACKBONE_MANIFEST}"
  --otflow_train_steps "${GENODE_OTFLOW_TRAIN_STEPS}"
  --device auto
)

generate_panel() {
  local panel=$1
  local nfes=$2
  local std_root="${GENODE_ROOT}/standard_inputs/${panel}"
  local fixed_train_root="${std_root}/fixed_train_tuning"
  local ser_ref_root="${std_root}/ser_reference"
  local ser_train_root="${std_root}/ser_train_tuning"
  local fixed_locked_root="${std_root}/fixed_locked"
  local ser_locked_root="${std_root}/ser_locked"

  mkdir -p "${fixed_train_root}" "${ser_ref_root}" "${ser_train_root}" "${fixed_locked_root}" "${ser_locked_root}"

  python -m genode.evaluation.diffusion_flow_time_reparameterization \
    --out_root "${fixed_train_root}" \
    "${COMMON_BACKBONE_ARGS[@]}" \
    --forecast_datasets "${GENODE_DATASET}" \
    --conditional_generation_datasets "" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
    --seeds "${GENODE_SEEDS}" \
    --split_phase train_tuning \
    --eval_train_fraction "${GENODE_TRAIN_TUNING_FRACTION}" \
    --train_tuning_strata "${GENODE_TRAIN_TUNING_STRATA}" \
    --train_tuning_sampling_mode "${GENODE_TRAIN_TUNING_SAMPLING_MODE}" \
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
    --target_nfe_values "${nfes}" \
    --reference_split train_tuning \
    --train_tuning_fraction "${GENODE_TRAIN_TUNING_FRACTION}" \
    --train_tuning_strata "${GENODE_TRAIN_TUNING_STRATA}" \
    --train_tuning_sampling_mode "${GENODE_TRAIN_TUNING_SAMPLING_MODE}" \
    --out_dir "${ser_ref_root}"

  local ser_summary="${ser_ref_root}/ser_ptg_schedule_summary.json"

  python -m genode.gipo.evaluate_schedule_summary \
    --schedule_summary "${ser_summary}" \
    --split_phase train_tuning \
    --out_dir "${ser_train_root}" \
    "${COMMON_BACKBONE_ARGS[@]}" \
    --dataset "${GENODE_DATASET}" \
    --seeds "${GENODE_SEEDS}" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --eval_train_fraction "${GENODE_TRAIN_TUNING_FRACTION}" \
    --train_tuning_strata "${GENODE_TRAIN_TUNING_STRATA}" \
    --train_tuning_sampling_mode "${GENODE_TRAIN_TUNING_SAMPLING_MODE}" \
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
    --write_context_rows \
    --context_row_csv_name ser_train_context_rows.csv \
    --context_embeddings_npz_name ser_train_context_embeddings.npz \
    --row_csv_name ser_train_rows.csv \
    --row_jsonl_name ser_train_rows.jsonl \
    --summary_output_name ser_train_schedule_summary.json

  python -m genode.evaluation.diffusion_flow_time_reparameterization \
    --out_root "${fixed_locked_root}" \
    "${COMMON_BACKBONE_ARGS[@]}" \
    --forecast_datasets "${GENODE_DATASET}" \
    --conditional_generation_datasets "" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --baseline_scheduler_names "${GENODE_FIXED_KEYS}" \
    --seeds "${GENODE_SEEDS}" \
    --split_phase locked_test \
    --eval_windows_test "${GENODE_LOCKED_EVAL_WINDOWS}" \
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
    --schedule_summary "${ser_summary}" \
    --split_phase locked_test \
    --out_dir "${ser_locked_root}" \
    "${COMMON_BACKBONE_ARGS[@]}" \
    --dataset "${GENODE_DATASET}" \
    --seeds "${GENODE_SEEDS}" \
    --solver_names "${GENODE_SOLVERS}" \
    --target_nfe_values "${nfes}" \
    --eval_windows_test "${GENODE_LOCKED_EVAL_WINDOWS}" \
    --num_eval_samples "${GENODE_NUM_EVAL_SAMPLES}" \
    --forecast_eval_batch_size "${GENODE_FORECAST_EVAL_BATCH_SIZE}" \
    --write_context_rows \
    --context_row_csv_name ser_locked_context_rows.csv \
    --context_embeddings_npz_name ser_locked_context_embeddings.npz \
    --row_csv_name ser_locked_rows.csv \
    --row_jsonl_name ser_locked_rows.jsonl \
    --summary_output_name ser_locked_schedule_summary.json \
    --baseline_rows "${fixed_locked_root}/fixed_locked_context_rows.csv"
}

generate_panel seen "${GENODE_SEEN_NFES}"
generate_panel unseen "${GENODE_UNSEEN_NFES}"

python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/validate_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --build \
  --validate
