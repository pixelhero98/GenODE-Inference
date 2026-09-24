"""Common configuration-based training CLI for every retained task."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

from genode.gico.evidence import prepare_evidence
from genode.gico.policy import load_context_embedding_table
from genode.gico.profiles import TrainingConfig, resolve_profile
from genode.gico.training import fit, reuse_utility_surrogate


def read_rows(path) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".json":
        value = json.loads(text)
        return value["rows"] if isinstance(value, dict) else value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def validate_config(config):
    allowed = {
        "rows",
        "contexts",
        "calibration_rows",
        "output",
        "policy_kind",
        "refinement_weight",
        "device",
        "purpose",
        "utility_surrogate_artifact",
        "collection_manifest",
    } | {field.name for field in fields(TrainingConfig)}
    if set(config) - allowed or not {"rows", "contexts", "output"} <= set(config):
        raise ValueError("Training config requires rows/contexts/output and only documented GICO options.")


def load_config(path) -> dict:
    location = Path(path).resolve()
    config = json.loads(location.read_text(encoding="utf-8"))
    validate_config(config)
    for key in ("rows", "contexts", "calibration_rows", "output", "utility_surrogate_artifact", "collection_manifest"):
        if key in config:
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else location.parent / value)
    return config


def run_config(config: dict, *, dry_run: bool = False) -> dict:
    validate_config(config)
    values = dict(config)
    if values.get("policy_kind", "deterministic") not in ("deterministic", "stochastic", "both", None):
        raise ValueError("policy_kind must be deterministic, stochastic, both, or null.")
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
        if "utility_surrogate_artifact" in values:
            reuse_utility_surrogate(values["utility_surrogate_artifact"], evidence, training)
        return {
            "task": evidence.task,
            "backbone": evidence.backbone,
            "purpose": evidence.purpose,
            "paired_cells": len(evidence.cells),
            "condition_width": evidence.conditioning.width,
            "reward_calibrations": {key: c.to_payload() for key, c in evidence.calibrations.items()},
            "refinement_weight": training.refinement_weight,
            "fitting_profile": asdict(training),
            "dry_run": True,
            "collection_manifest": evidence.collection_manifest,
        }
    return fit(rows, contexts, calibration_rows=calibration, collection_manifest=manifest, **values)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON configuration; paths are relative to this file.")
    parser.add_argument("--policy-kind", choices=("deterministic", "stochastic", "both"))
    parser.add_argument("--utility-surrogate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    if args.policy_kind is not None:
        config["policy_kind"] = args.policy_kind
    if args.utility_surrogate_only:
        if args.policy_kind is not None:
            raise ValueError("Choose either a policy kind or utility-surrogate-only fitting.")
        config["policy_kind"] = None
    print(json.dumps(run_config(config, dry_run=args.dry_run), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
