"""Common configuration-based training CLI for every retained task."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from genode.gico.evidence import prepare_evidence
from genode.gico.policy import load_context_embedding_table
from genode.gico.profiles import SCORE_WEIGHTS, TrainingConfig, resolve_profile
from genode.gico.training import fit, reuse_teacher


def read_rows(path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".json":
        value = json.loads(text)
        return value["rows"] if isinstance(value, dict) else value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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
        "collection_manifest",
    } | {field.name for field in fields(TrainingConfig)}
    if set(config) - allowed or not {"rows", "contexts", "output"} <= set(config):
        raise ValueError("Training config requires rows/contexts/output and only documented GICO options.")
    for key in ("rows", "contexts", "calibration_rows", "output", "teacher_artifact", "collection_manifest"):
        if key in config:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else location.parent / value)
    return config


def run_config(config: dict, *, dry_run: bool = False) -> dict:
    values = dict(config)
    if values.get("student_kind", "GICO-det-policy") not in ("GICO-det-policy", "GICO-sto-policy", "both"):
        raise ValueError("student_kind must be GICO-det-policy, GICO-sto-policy, or both.")
    rows = read_rows(values.pop("rows"))
    contexts = load_context_embedding_table(values.pop("contexts"))
    calibration = read_rows(values.pop("calibration_rows")) if "calibration_rows" in values else None
    manifest = values.pop("collection_manifest", None)
    manifest = json.loads(Path(manifest).read_text(encoding="utf-8")) if manifest is not None else None
    if manifest is not None and "collection_manifest" in manifest:
        manifest = manifest["collection_manifest"]
    if dry_run:
        evidence = prepare_evidence(
            rows,
            contexts,
            calibration_rows=calibration,
            purpose=values.get("purpose", "research"),
            collection_manifest=manifest,
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
            "collection_manifest": evidence.collection_manifest,
        }
    return fit(rows, contexts, calibration_rows=calibration, collection_manifest=manifest, **values)


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
