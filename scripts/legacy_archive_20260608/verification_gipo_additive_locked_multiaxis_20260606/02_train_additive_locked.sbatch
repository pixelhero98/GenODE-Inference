#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_mx_train
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606/logs/%x.%j.out
#SBATCH --error=/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606/logs/%x.%j.err

set -euo pipefail

RUN_ID=additive_locked_b128_multiaxis_select
CONDITIONING_STYLE=additive_mlp_v1
SELECTION_MODE=composite_regret_v1
CRPS_WEIGHT=0.50
MASE_WEIGHT=0.50
DENSITY_BIN_COUNT=128
STUDENT_NFE_SMOOTHNESS_WEIGHT=0
STUDENT_PSEUDO_TARGET_WEIGHT=0

GENODE_REPO=${GENODE_REPO:-/scratch/b35z/pixelhero.b35z/genode/exec/genode_gipo_additive_locked_multiaxis_src_20260606}
GENODE_ENV_DIR=${GENODE_ENV_DIR:-/projects/b35z/genode/envs/genode-py311}
GENODE_ROOT=${GENODE_ROOT:-/scratch/b35z/pixelhero.b35z/genode/outputs/verification_gipo_additive_locked_multiaxis_20260606}
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,late_power_3_reversed,flowts_power_sampling_reversed,ays_reversed,gits_reversed,ots_reversed,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_STEPS=${GENODE_STUDENT_STEPS:-500}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/policy_runs" "${GENODE_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces
python scripts/verification_gipo_additive_locked_multiaxis_20260606/validate_additive_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --validate

ROWS="${GENODE_ROOT}/artifacts/seen_train_supervision_rows.csv"
EMBEDDINGS="${GENODE_ROOT}/artifacts/seen_train_context_embeddings.npz"
SER_SUMMARY="${GENODE_ROOT}/standard_inputs/seen/ser_reference/ser_ptg_schedule_summary.json"
OUT_DIR="${GENODE_ROOT}/policy_runs/${RUN_ID}"

if [[ -f "${OUT_DIR}/gipo_training_summary.json" ]]; then
  echo "Skipping existing ${RUN_ID}"
else
  mkdir -p "${OUT_DIR}"
  python -m genode.gipo.train_gipo \
    --rows_csv "${ROWS}" \
    --context_embeddings_npz "${EMBEDDINGS}" \
    --schedule_summary_json "${SER_SUMMARY}" \
    --out_dir "${OUT_DIR}" \
    --support_schedule_keys "${GENODE_SUPPORT_KEYS}" \
    --context_sample_count "${GENODE_CONTEXT_SAMPLE_COUNT}" \
    --context_holdout_fraction 0.20 \
    --series_holdout_fraction 0.20 \
    --density_bin_count "${DENSITY_BIN_COUNT}" \
    --teacher_steps "${GENODE_TEACHER_STEPS}" \
    --student_steps "${GENODE_STUDENT_STEPS}" \
    --teacher_checkpoint_selection_mode "${SELECTION_MODE}" \
    --teacher_selection_component_weights context=0.40,density_family=0.25,nfe_proxy=0.35 \
    --teacher_selection_temperature 0.05 \
    --teacher_density_holdout_schedule_keys flowts_power_sampling,flowts_power_sampling_reversed \
    --teacher_nfe_proxy_anchor_values 4,8,12 \
    --teacher_temperature_mode fixed \
    --teacher_temperature 0.05 \
    --teacher_architecture density_form_transformer_v1 \
    --student_architecture density_query_transformer_v1 \
    --setting_encoder_mode continuous_v3 \
    --gipo_conditioning_style "${CONDITIONING_STYLE}" \
    --transformer_heads 4 \
    --transformer_dropout 0.05 \
    --teacher_utility_crps_weight "${CRPS_WEIGHT}" \
    --teacher_utility_mase_weight "${MASE_WEIGHT}" \
    --series_unknown_dropout 0 \
    --student_weight_decay 0.0001 \
    --student_nfe_smoothness_weight "${STUDENT_NFE_SMOOTHNESS_WEIGHT}" \
    --student_pseudo_target_weight "${STUDENT_PSEUDO_TARGET_WEIGHT}" \
    --device auto
fi

RUN_ID="${RUN_ID}" \
SELECTION_MODE="${SELECTION_MODE}" \
CONDITIONING_STYLE="${CONDITIONING_STYLE}" \
DENSITY_BIN_COUNT="${DENSITY_BIN_COUNT}" \
STUDENT_NFE_SMOOTHNESS_WEIGHT="${STUDENT_NFE_SMOOTHNESS_WEIGHT}" \
STUDENT_PSEUDO_TARGET_WEIGHT="${STUDENT_PSEUDO_TARGET_WEIGHT}" \
GENODE_ROOT="${GENODE_ROOT}" \
ROWS="${ROWS}" \
EMBEDDINGS="${EMBEDDINGS}" \
SER_SUMMARY="${SER_SUMMARY}" \
GENODE_SUPPORT_KEYS="${GENODE_SUPPORT_KEYS}" \
python - <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["GENODE_ROOT"])
run_id = os.environ["RUN_ID"]
run_dir = root / "policy_runs" / run_id
summary_path = run_dir / "gipo_training_summary.json"
if not summary_path.exists():
    raise FileNotFoundError(f"Training summary is required before final-retrain metadata: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))

metadata = {
    "artifact": "gipo_additive_locked_multiaxis_final_retrain_metadata",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "canonical_run_id": "additive_locked_b128_multiaxis_select",
    "canonical_run_count": 1,
    "selection_mode": os.environ["SELECTION_MODE"],
    "selection_policy": "composite_regret_v1",
    "selection": {
        "mode": os.environ["SELECTION_MODE"],
        "candidate_run_ids": [run_id],
        "selected_run_id": run_id,
        "selected_before_locked_reporting": True,
        "locked_test_used_for_selection": False,
    },
    "final_retrain": {
        "enabled": True,
        "completed": summary.get("status") == "completed",
        "rows_csv": os.environ["ROWS"],
        "context_embeddings_npz": os.environ["EMBEDDINGS"],
        "schedule_summary_json": os.environ["SER_SUMMARY"],
        "support_schedule_keys": [part for part in os.environ["GENODE_SUPPORT_KEYS"].split(",") if part],
        "source_split_phases_allowed": ["train_tuning"],
        "training_target_nfe_values": [4, 8, 12],
        "locked_report_target_nfe_values": {"seen": [4, 8, 12], "unseen": [6, 10, 14, 16]},
        "locked_test_used_for_selection": False,
    },
    "script_contract": {
        "conditioning_style": os.environ["CONDITIONING_STYLE"],
        "density_bin_count": int(os.environ["DENSITY_BIN_COUNT"]),
        "student_nfe_smoothness_weight": float(os.environ["STUDENT_NFE_SMOOTHNESS_WEIGHT"]),
        "student_pseudo_target_weight": float(os.environ["STUDENT_PSEUDO_TARGET_WEIGHT"]),
        "locked_test_used_for_selection": False,
    },
    "training_summary_path": str(summary_path),
    "policy_id": summary.get("policy_id", ""),
}
metadata_path = run_dir / "final_retrain_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"final_retrain_metadata": str(metadata_path), "selection_mode": metadata["selection_mode"]}, sort_keys=True))
PY
