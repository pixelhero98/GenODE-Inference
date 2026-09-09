"""Common configuration-based training CLI for every retained task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genode.gico.evidence import prepare_evidence
from genode.gico.policy import load_context_embedding_table
from genode.gico.training import SCORE_WEIGHTS, TrainingConfig, fit


def read_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_config(path) -> dict:
    location = Path(path).resolve()
    config = json.loads(location.read_text(encoding="utf-8"))
    allowed = {
        "rows",
        "contexts",
        "calibration_rows",
        "output",
        "student_kind",
        "teacher_score_weight",
        "steps",
        "seed",
        "device",
        "purpose",
        "batch_size",
    }
    if set(config) - allowed or not {"rows", "contexts", "output"} <= set(config):
        raise ValueError("Training config requires rows/contexts/output and only documented GICO options.")
    for key in ("rows", "contexts", "calibration_rows", "output"):
        if key in config:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else location.parent / value)
    return config


def run_config(config: dict, *, dry_run: bool = False) -> dict:
    values = dict(config)
    if values.get("student_kind", "both") not in ("deterministic", "stochastic", "both"):
        raise ValueError("student_kind must be deterministic, stochastic, or both.")
    rows = read_rows(values.pop("rows"))
    contexts = load_context_embedding_table(values.pop("contexts"))
    calibration = read_rows(values.pop("calibration_rows")) if "calibration_rows" in values else None
    if dry_run:
        training = TrainingConfig(
            steps=values.get("steps", 2000),
            seed=values.get("seed", 0),
            teacher_score_weight=values.get("teacher_score_weight", 0.01),
            batch_size=values.get("batch_size", 32),
        )
        evidence = prepare_evidence(
            rows, contexts, calibration_rows=calibration, purpose=values.get("purpose", "research")
        )
        return {
            "task": evidence.task,
            "backbone": evidence.backbone,
            "purpose": evidence.purpose,
            "paired_cells": len(evidence.cells),
            "condition_width": evidence.conditioning.width,
            "reward_calibrations": {key: c.to_payload() for key, c in evidence.calibrations.items()},
            "teacher_score_weight": training.teacher_score_weight,
            "dry_run": True,
        }
    return fit(rows, contexts, calibration_rows=calibration, **values)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON configuration; paths are relative to this file.")
    parser.add_argument("--student-kind", choices=("deterministic", "stochastic", "both"))
    parser.add_argument("--teacher-score-weight", type=float, choices=SCORE_WEIGHTS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    if args.student_kind is not None:
        config["student_kind"] = args.student_kind
    if args.teacher_score_weight is not None:
        config["teacher_score_weight"] = args.teacher_score_weight
    print(json.dumps(run_config(config, dry_run=args.dry_run), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
