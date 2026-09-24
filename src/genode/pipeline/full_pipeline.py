"""Explicit forecast/molecule evidence stages and common GICO training."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from genode.backbone_manifest import validate_provided_backbone_manifest
from genode.canonical_experiment_layout import (
    CANONICAL_CHECKPOINT_STEPS,
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_SCENARIO_KEYS,
    CANONICAL_SEEN_NFES,
    CANONICAL_SUPERVISION_SCHEDULE_KEYS,
    CANONICAL_UNSEEN_NFES,
    SCENARIO_FAMILY_FORECAST,
    scenario_family_for_key,
)
from genode.data.molecule_xyz import (
    default_molecule_group_root,
    load_molecule_group_manifest,
    trainable_molecule_group_members,
)
from genode.data.otflow_monash_datasets import download_monash_dataset
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    project_outputs_root,
    project_paper_dataset_root,
    resolve_project_path,
)
from genode.evaluation import diffusion_flow_time_reparameterization as schedule_runner

PIPELINE_VERSION = "unified_gico_pipeline_v2"
DEFAULT_STAGES = ("backbone_training", "schedule_rows_seen", "schedule_rows_unseen")
PIPELINE_STAGE_ORDER = ("data_prep", *DEFAULT_STAGES, "gico_training")


@dataclass(frozen=True)
class StageCommand:
    stage: str
    commands: list[list[str]]
    manifest_name: str


def _python_module_command(module: str, args: list[object]) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in args)]


def _resolved_scenario_key(args: argparse.Namespace) -> str:
    return str(args.scenario_key)


def _effective_stage_names(args: argparse.Namespace) -> list[str]:
    stages = (
        [value.strip() for value in args.stages.split(",") if value.strip()] if args.stages else list(DEFAULT_STAGES)
    )
    unknown = set(stages) - set(PIPELINE_STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")
    if len(stages) != len(set(stages)):
        raise ValueError("Pipeline stages must be unique.")
    if "gico_training" in stages and not args.gico_config:
        raise ValueError("The gico_training stage requires --gico-config.")
    phases = args.split_phases.split(",")
    if (
        not phases
        or len(phases) != len(set(phases))
        or any(phase not in {"train_tuning", "validation_tuning", "locked_test"} for phase in phases)
    ):
        raise ValueError("Use unique train_tuning, validation_tuning or locked_test split phases.")
    if args.context_sample_count <= 0:
        raise ValueError("context_sample_count must be positive.")
    if args.use_provided_backbones and "backbone_training" in stages:
        raise ValueError("Provided backbones cannot be combined with backbone_training.")
    return [stage for stage in PIPELINE_STAGE_ORDER if stage in stages]


def _schedule_dataset_args(dataset: str) -> list[str]:
    forecast = scenario_family_for_key(dataset) == SCENARIO_FAMILY_FORECAST
    return ["--forecast_datasets", dataset if forecast else "", "--molecule_datasets", "" if forecast else dataset]


def _backbone_training_commands(args: argparse.Namespace, dataset: str, checkpoints: str) -> list[list[str]]:
    if scenario_family_for_key(dataset) == SCENARIO_FAMILY_FORECAST:
        return [
            _python_module_command(
                "genode.training.train_backbone",
                [
                    "--dataset",
                    dataset,
                    "--dataset_root",
                    args.dataset_root,
                    "--steps",
                    args.backbone_steps,
                    "--checkpoint_steps",
                    checkpoints,
                    "--checkpoint_export_mode",
                    "exact_budget",
                    "--device",
                    args.device,
                ],
            )
        ]
    group_root = resolve_project_path(args.molecule_group_root)
    manifest = load_molecule_group_manifest(dataset, group_root)
    members = trainable_molecule_group_members(manifest)
    if not members:
        raise ValueError(f"Molecule group {dataset!r} has no trainable members.")
    return [
        _python_module_command(
            "genode.training.train_molecule_backbone",
            [
                "--dataset_key",
                dataset,
                "--stratum",
                member["stratum"],
                "--processed_dir",
                group_root / dataset / member["processed_dir"],
                "--out_dir",
                args.molecule_backbone_root,
                "--backbone_manifest",
                args.backbone_manifest,
                "--molecule_group_root",
                group_root,
                "--steps",
                args.backbone_steps,
                "--budget_steps",
                checkpoints,
                "--device",
                args.device,
                "--no_prepare_data",
            ],
        )
        for member in members
    ]


def _build_stage_commands(args: argparse.Namespace, run_root: Path) -> list[StageCommand]:
    stages = _effective_stage_names(args)
    dataset = _resolved_scenario_key(args)
    checkpoint_steps = [int(value) for value in args.checkpoint_steps.split(",")]
    if not checkpoint_steps or any(value not in CANONICAL_CHECKPOINT_STEPS for value in checkpoint_steps):
        raise ValueError("Use canonical checkpoint steps.")
    entries = []
    for stage in stages:
        commands = []
        if stage == "data_prep":
            commands = [["internal", "prepare_data", dataset]]
        elif stage == "backbone_training":
            commands = _backbone_training_commands(args, dataset, args.checkpoint_steps)
        elif stage in {"schedule_rows_seen", "schedule_rows_unseen"}:
            role = stage.removeprefix("schedule_rows_")
            nfes = args.seen_nfes if role == "seen" else args.unseen_nfes
            for step in checkpoint_steps:
                for phase in args.split_phases.split(","):
                    commands.append(
                        _python_module_command(
                            "genode.evaluation.diffusion_flow_time_reparameterization",
                            [
                                *_schedule_dataset_args(dataset),
                                "--nfe_role",
                                role,
                                "--target_nfe_values",
                                nfes,
                                "--checkpoint_steps",
                                step,
                                "--baseline_scheduler_names",
                                args.schedule_keys,
                                "--split_phase",
                                phase,
                                "--train_tuning_context_sample_count",
                                args.context_sample_count,
                                "--molecule_group_root",
                                args.molecule_group_root,
                                "--backbone_manifest",
                                args.backbone_manifest,
                                "--dataset_root",
                                args.dataset_root,
                                "--shared_backbone_root",
                                args.shared_backbone_root,
                                "--write_context_rows",
                                "--allow_execute",
                                "--device",
                                args.device,
                                "--out_root",
                                run_root / "schedule_rows" / role / phase / f"{step}_steps",
                            ],
                        )
                    )
        elif stage == "gico_training":
            config_path = resolve_project_path(args.gico_config)
            if not config_path.is_file():
                raise FileNotFoundError(f"GICO configuration not found: {config_path}")
            commands = [
                _python_module_command(
                    "genode.gico.train_gico",
                    [
                        "--config",
                        config_path,
                        "--policy-kind",
                        args.policy_kind,
                    ],
                )
            ]
        entries.append(StageCommand(stage, commands, f"{stage}_manifest.json"))
    return entries


def run_full_pipeline(args: argparse.Namespace) -> dict[str, object]:
    scenario_family_for_key(args.scenario_key)
    stages = _effective_stage_names(args)
    if args.use_provided_backbones and not args.dry_run:
        validation = validate_provided_backbone_manifest(
            args.backbone_manifest,
            scenario_key=args.scenario_key,
            benchmark_family=scenario_family_for_key(args.scenario_key),
        )
        if validation["status"] != "complete":
            raise ValueError("Invalid provided backbone manifest: " + "; ".join(validation["errors"]))
    run_root = resolve_project_path(args.run_root or str(project_outputs_root() / "full_pipeline" / args.scenario_key))
    entries = _build_stage_commands(args, run_root)
    payload = {
        "version": PIPELINE_VERSION,
        "scenario_key": args.scenario_key,
        "stages": stages,
        "commands": [{"stage": entry.stage, "commands": entry.commands} for entry in entries],
    }
    if args.dry_run:
        return {**payload, "status": "dry_run"}
    run_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        manifest_path = run_root / entry.manifest_name
        command_hash = hashlib.sha256(json.dumps(entry.commands, sort_keys=True).encode()).hexdigest()
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if args.resume and previous.get("command_hash") == command_hash and previous.get("status") == "complete":
                if entry.stage.startswith("schedule_rows_"):
                    complete = True
                    for command in entry.commands:
                        parsed = schedule_runner.build_argparser().parse_args(command[3:])
                        state = schedule_runner.schedule_row_output_status(Path(parsed.out_root), parsed)
                        complete = complete and bool(state.get("complete", False))
                    if complete:
                        continue
                # Backbone and GICO stages validate their own checkpoint resume contracts.
            elif not args.overwrite:
                raise FileExistsError(f"Stage manifest already exists: {manifest_path}; use --resume or --overwrite.")
        record = {"stage": entry.stage, "commands": entry.commands, "command_hash": command_hash, "status": "running"}
        manifest_path.write_text(json.dumps(record, indent=2))
        try:
            for command in entry.commands:
                if command[:2] == ["internal", "prepare_data"]:
                    if scenario_family_for_key(args.scenario_key) == SCENARIO_FAMILY_FORECAST:
                        download_monash_dataset(resolve_project_path(args.dataset_root), args.scenario_key)
                    else:
                        load_molecule_group_manifest(args.scenario_key, resolve_project_path(args.molecule_group_root))
                else:
                    subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            manifest_path.write_text(json.dumps({**record, "status": "failed", "error": str(exc)}, indent=2))
            raise
        manifest_path.write_text(json.dumps({**record, "status": "complete"}, indent=2))
    summary = {**payload, "status": "complete"}
    (run_root / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario_key", choices=CANONICAL_SCENARIO_KEYS, required=True)
    parser.add_argument("--run_root", default="")
    parser.add_argument("--stages", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone_steps", type=int, default=20000)
    parser.add_argument("--checkpoint_steps", default=",".join(map(str, CANONICAL_CHECKPOINT_STEPS)))
    parser.add_argument("--seen_nfes", default=",".join(map(str, CANONICAL_SEEN_NFES)))
    parser.add_argument("--unseen_nfes", default=",".join(map(str, CANONICAL_UNSEEN_NFES)))
    parser.add_argument("--schedule_keys", default=",".join(CANONICAL_SUPERVISION_SCHEDULE_KEYS))
    parser.add_argument("--context_sample_count", type=int, default=CANONICAL_CONTEXT_SAMPLE_COUNT)
    parser.add_argument("--split_phases", default="train_tuning,validation_tuning")
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(project_outputs_root() / "otflow_backbones"))
    parser.add_argument("--backbone_manifest", default=str(default_backbone_manifest_path()))
    parser.add_argument("--molecule_group_root", default=str(default_molecule_group_root()))
    parser.add_argument("--molecule_backbone_root", default=str(project_outputs_root() / "molecule_3d_backbones"))
    parser.add_argument("--use_provided_backbones", action="store_true")
    parser.add_argument("--gico-config", default="")
    parser.add_argument("--policy-kind", choices=("deterministic", "stochastic", "both"), default="both")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run_full_pipeline(build_argparser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
