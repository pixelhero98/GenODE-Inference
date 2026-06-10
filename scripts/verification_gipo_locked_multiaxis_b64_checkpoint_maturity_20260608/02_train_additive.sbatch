#!/usr/bin/env bash
#SBATCH --job-name=genode_gipo_ckpt_train
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err

set -euo pipefail

CONDITIONING_STYLE=additive_mlp_v1
SELECTION_MODE=weighted_normalized_regret_v1
STUDENT_SELECTOR_MODE=validation_ce_v1
CRPS_WEIGHT=0.50
MASE_WEIGHT=0.50
DENSITY_BIN_COUNT=64

GENODE_REPO=${GENODE_REPO:?Set GENODE_REPO to the GenODE-Inference repository path}
GENODE_ENV_DIR=${GENODE_ENV_DIR:?Set GENODE_ENV_DIR to the Python environment path}
GENODE_OTFLOW_TRAIN_STEPS=${GENODE_OTFLOW_TRAIN_STEPS:?Set GENODE_OTFLOW_TRAIN_STEPS to the backbone maturity step budget}
GENODE_BUDGET_LABEL=${GENODE_BUDGET_LABEL:-$((GENODE_OTFLOW_TRAIN_STEPS / 1000))k}
GENODE_ROOT=${GENODE_ROOT:?Set GENODE_ROOT to this run output root}
RUN_ID=${RUN_ID:-additive_locked_b64_normregret_ckpt${GENODE_BUDGET_LABEL}}
GENODE_SUPPORT_KEYS=${GENODE_SUPPORT_KEYS:-uniform,late_power_3,flowts_power_sampling,ays,gits,ots,late_power_3_reversed,flowts_power_sampling_reversed,ays_reversed,gits_reversed,ots_reversed,ser_ptg_local_defect_eta005}
GENODE_CONTEXT_SAMPLE_COUNT=${GENODE_CONTEXT_SAMPLE_COUNT:-256}
GENODE_TEACHER_STEPS=${GENODE_TEACHER_STEPS:-500}
GENODE_STUDENT_BUDGET=${GENODE_STUDENT_BUDGET:-1000}
GENODE_TEACHER_CHECKPOINT_EVERY=${GENODE_TEACHER_CHECKPOINT_EVERY:-50}
GENODE_TEACHER_LOG_EVERY=${GENODE_TEACHER_LOG_EVERY:-50}
GENODE_STUDENT_LOG_EVERY=${GENODE_STUDENT_LOG_EVERY:-50}
GENODE_STUDENT_CHECKPOINT_EVERY=${GENODE_STUDENT_CHECKPOINT_EVERY:-50}
GENODE_STUDENT_SELECTION_HOLDOUT_FRACTION=${GENODE_STUDENT_SELECTION_HOLDOUT_FRACTION:-0.10}

mkdir -p "${GENODE_ROOT}/logs" "${GENODE_ROOT}/policy_runs" "${GENODE_ROOT}/summary"

source "${GENODE_ENV_DIR}/bin/activate"
cd "${GENODE_REPO}"
export PYTHONPATH="${GENODE_REPO}/src"
export PYTHONDONTWRITEBYTECODE=1

python -m compileall -q src tests scripts
python -m unittest tests.test_gipo tests.test_genode_interfaces
python scripts/verification_gipo_locked_multiaxis_b64_checkpoint_maturity_20260608/validate_locked_artifacts.py \
  --root "${GENODE_ROOT}" \
  --validate

ROWS="${GENODE_ROOT}/artifacts/seen_train_supervision_rows.csv"
EMBEDDINGS="${GENODE_ROOT}/artifacts/seen_train_context_embeddings.npz"
SER_SUMMARY="${GENODE_ROOT}/standard_inputs/seen/ser_reference/ser_ptg_schedule_summary.json"
UNSEEN_SELECTION_ROWS="${GENODE_ROOT}/artifacts/unseen_train_supervision_rows.csv"
UNSEEN_SELECTION_EMBEDDINGS="${GENODE_ROOT}/artifacts/unseen_train_context_embeddings.npz"
UNSEEN_SER_SUMMARY="${GENODE_ROOT}/standard_inputs/unseen/ser_reference/ser_ptg_schedule_summary.json"
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
    --teacher_steps "${GENODE_TEACHER_STEPS}" \
    --teacher_checkpoint_every "${GENODE_TEACHER_CHECKPOINT_EVERY}" \
    --teacher_loss_log_every "${GENODE_TEACHER_LOG_EVERY}" \
    --student_steps "${GENODE_STUDENT_BUDGET}" \
    --student_log_every "${GENODE_STUDENT_LOG_EVERY}" \
    --student_checkpoint_every "${GENODE_STUDENT_CHECKPOINT_EVERY}" \
    --student_selection_holdout_fraction "${GENODE_STUDENT_SELECTION_HOLDOUT_FRACTION}" \
    --teacher_density_holdout_schedule_keys flowts_power_sampling,flowts_power_sampling_reversed \
    --teacher_unseen_selection_rows_csv "${UNSEEN_SELECTION_ROWS}" \
    --teacher_unseen_selection_context_embeddings_npz "${UNSEEN_SELECTION_EMBEDDINGS}" \
    --teacher_unseen_selection_schedule_summary_json "${UNSEEN_SER_SUMMARY}" \
    --teacher_unseen_selection_target_nfe_values 6,10,14,16 \
    --teacher_temperature 0.05 \
    --gipo_conditioning_style "${CONDITIONING_STYLE}" \
    --transformer_heads 4 \
    --transformer_dropout 0.05 \
    --teacher_utility_crps_weight "${CRPS_WEIGHT}" \
    --teacher_utility_mase_weight "${MASE_WEIGHT}" \
    --student_weight_decay 0.0001 \
    --device auto
fi

RUN_ID="${RUN_ID}" \
SELECTION_MODE="${SELECTION_MODE}" \
CONDITIONING_STYLE="${CONDITIONING_STYLE}" \
STUDENT_SELECTOR_MODE="${STUDENT_SELECTOR_MODE}" \
DENSITY_BIN_COUNT="${DENSITY_BIN_COUNT}" \
GENODE_OTFLOW_TRAIN_STEPS="${GENODE_OTFLOW_TRAIN_STEPS}" \
GENODE_STUDENT_BUDGET="${GENODE_STUDENT_BUDGET}" \
GENODE_TEACHER_CHECKPOINT_EVERY="${GENODE_TEACHER_CHECKPOINT_EVERY}" \
GENODE_TEACHER_LOG_EVERY="${GENODE_TEACHER_LOG_EVERY}" \
GENODE_STUDENT_LOG_EVERY="${GENODE_STUDENT_LOG_EVERY}" \
GENODE_STUDENT_CHECKPOINT_EVERY="${GENODE_STUDENT_CHECKPOINT_EVERY}" \
GENODE_STUDENT_SELECTION_HOLDOUT_FRACTION="${GENODE_STUDENT_SELECTION_HOLDOUT_FRACTION}" \
GENODE_ROOT="${GENODE_ROOT}" \
ROWS="${ROWS}" \
EMBEDDINGS="${EMBEDDINGS}" \
SER_SUMMARY="${SER_SUMMARY}" \
UNSEEN_SELECTION_ROWS="${UNSEEN_SELECTION_ROWS}" \
UNSEEN_SELECTION_EMBEDDINGS="${UNSEEN_SELECTION_EMBEDDINGS}" \
UNSEEN_SER_SUMMARY="${UNSEEN_SER_SUMMARY}" \
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
summary = json.loads(summary_path.read_text(encoding="utf-8"))

metadata = {
    "artifact": "gipo_locked_multiaxis_b64_checkpoint_maturity_final_retrain_metadata",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "run_role": "canonical_additive_checkpoint_maturity",
    "selection_mode": os.environ["SELECTION_MODE"],
    "otflow_train_steps": int(os.environ["GENODE_OTFLOW_TRAIN_STEPS"]),
    "final_retrain": {
        "enabled": True,
        "completed": summary.get("status") == "completed",
        "rows_csv": os.environ["ROWS"],
        "context_embeddings_npz": os.environ["EMBEDDINGS"],
        "schedule_summary_json": os.environ["SER_SUMMARY"],
        "unseen_selection_rows_csv": os.environ["UNSEEN_SELECTION_ROWS"],
        "unseen_selection_context_embeddings_npz": os.environ["UNSEEN_SELECTION_EMBEDDINGS"],
        "unseen_selection_schedule_summary_json": os.environ["UNSEEN_SER_SUMMARY"],
        "support_schedule_keys": [part for part in os.environ["GENODE_SUPPORT_KEYS"].split(",") if part],
        "source_split_phases_allowed": ["train_tuning"],
        "training_target_nfe_values": [4, 8, 12],
        "locked_report_target_nfe_values": {"seen": [4, 8, 12], "unseen": [6, 10, 14, 16]},
        "locked_test_used_for_selection": False,
    },
    "script_contract": {
        "conditioning_style": os.environ["CONDITIONING_STYLE"],
        "density_bin_count": int(os.environ["DENSITY_BIN_COUNT"]),
        "teacher_checkpoint_every": int(os.environ["GENODE_TEACHER_CHECKPOINT_EVERY"]),
        "teacher_loss_log_every": int(os.environ["GENODE_TEACHER_LOG_EVERY"]),
        "student_selector_mode": os.environ["STUDENT_SELECTOR_MODE"],
        "student_budget": int(os.environ["GENODE_STUDENT_BUDGET"]),
        "student_log_every": int(os.environ["GENODE_STUDENT_LOG_EVERY"]),
        "student_checkpoint_every": int(os.environ["GENODE_STUDENT_CHECKPOINT_EVERY"]),
        "student_selection_holdout_fraction": float(os.environ["GENODE_STUDENT_SELECTION_HOLDOUT_FRACTION"]),
        "locked_test_used_for_selection": False,
    },
    "training_summary_path": str(summary_path),
    "policy_id": summary.get("policy_id", ""),
}
metadata_path = run_dir / "final_retrain_metadata.json"
metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"final_retrain_metadata": str(metadata_path), "run_id": run_id}, sort_keys=True))
PY
