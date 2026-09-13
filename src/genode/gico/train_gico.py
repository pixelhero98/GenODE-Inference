"""Common configuration-based training CLI for every retained task."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from genode.gico.evidence import prepare_evidence
from genode.gico.policy import load_context_embedding_table
from genode.gico.profiles import SCORE_WEIGHTS, TrainingConfig, resolve_profile
from genode.gico.training import fit, reuse_teacher


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
        "device",
        "purpose",
        "teacher_artifact",
        "selection_evaluator",
    } | {field.name for field in fields(TrainingConfig)}
    if set(config) - allowed or not {"rows", "contexts", "output"} <= set(config):
        raise ValueError("Training config requires rows/contexts/output and only documented GICO options.")
    for key in ("rows", "contexts", "calibration_rows", "output", "teacher_artifact"):
        if key in config:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else location.parent / value)
    return config


def load_selection_evaluator(evaluator, *, dry_run=False):
    if evaluator is not None and (
        not isinstance(evaluator, dict)
        or set(evaluator) != {"factory", "config"}
        or not isinstance(evaluator["factory"], str)
        or len(evaluator["factory"].split(":")) != 2
        or not all(part.isidentifier() for part in evaluator["factory"].replace(":", ".").split("."))
        or not isinstance(evaluator["config"], dict)
    ):
        raise ValueError("selection_evaluator requires a trusted module:factory and a config object.")
    if dry_run:
        return None
    if evaluator is None:
        raise ValueError("Training requires a held-out terminal-utility selection_evaluator factory.")
    module, name = evaluator["factory"].split(":")
    callback = getattr(importlib.import_module(module), name)(evaluator["config"])
    if not callable(callback):
        raise ValueError("Selection evaluator factory must return a callable.")
    return callback


def run_config(config: dict, *, dry_run: bool = False) -> dict:
    values = dict(config)
    evaluator = values.pop("selection_evaluator", None)
    callback = load_selection_evaluator(evaluator, dry_run=dry_run)
    if values.get("student_kind", "both") not in ("GICO-det-policy", "GICO-sto-policy", "both"):
        raise ValueError("student_kind must be GICO-det-policy, GICO-sto-policy, or both.")
    rows = read_rows(values.pop("rows"))
    contexts = load_context_embedding_table(values.pop("contexts"))
    calibration = read_rows(values.pop("calibration_rows")) if "calibration_rows" in values else None
    if dry_run:
        evidence = prepare_evidence(
            rows, contexts, calibration_rows=calibration, purpose=values.get("purpose", "research")
        )
        training = resolve_profile(
            evidence.task,
            **{key: value for key, value in values.items() if key in {field.name for field in fields(TrainingConfig)}},
        )
        if training.backbone is not None and training.backbone != evidence.backbone:
            raise ValueError("Fitting profile backbone differs from measurement evidence.")
        training = replace(training, backbone=evidence.backbone)
        if "teacher_artifact" in values:
            reuse_teacher(values["teacher_artifact"], evidence, training)
        return {
            "task": evidence.task,
            "backbone": evidence.backbone,
            "purpose": evidence.purpose,
            "paired_cells": len(evidence.cells),
            "condition_width": evidence.conditioning.width,
            "reward_calibrations": {key: c.to_payload() for key, c in evidence.calibrations.items()},
            "teacher_score_weight": training.teacher_score_weight,
            "fitting_profile": asdict(training),
            "dry_run": True,
            "selection_evaluator": evaluator,
        }
    return fit(rows, contexts, calibration_rows=calibration, selection_evaluator=callback, **values)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON configuration; paths are relative to this file.")
    parser.add_argument("--student-kind", choices=("GICO-det-policy", "GICO-sto-policy", "both"))
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
