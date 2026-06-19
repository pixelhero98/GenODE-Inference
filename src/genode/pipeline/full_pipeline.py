from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from genode.canonical_experiment_layout import (
    CANONICAL_CHECKPOINT_STEPS,
    CANONICAL_CONTEXT_SAMPLE_COUNT,
    CANONICAL_PSEUDO_TARGET_WEIGHT,
    CANONICAL_SEEN_NFES,
    CANONICAL_SOLVER_KEYS,
    CANONICAL_SUPERVISION_SCHEDULE_KEYS,
    CANONICAL_UNSEEN_NFES,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO,
    scenario_family_for_key,
)
from genode.data.molecule_xyz import default_molecule_group_root, load_molecule_group_manifest, trainable_molecule_group_members
from genode.data.otflow_experiment_plan import experiment_plan_by_key
from genode.data.otflow_paths import (
    default_backbone_manifest_path,
    default_cryptos_data_path,
    default_lobster_synthetic_profile_path,
    default_long_term_st_data_path,
    project_outputs_root,
    project_paper_dataset_root,
    project_root,
    resolve_project_path,
)
from genode.backbone_packages import (
    apply_backbone_package_to_args,
    backbone_package_protocol_payload,
    validate_provided_backbone_manifest,
)
from genode.gipo.objectives import objective_specs_for_family
from genode.gipo.ablation_plan import (
    DEFAULT_GIPO_ABLATION_PRESET,
    GipoAblationArm,
    gipo_ablation_arms,
    gipo_ablation_preset_choices,
)
from genode.gipo.policy import (
    DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT,
    DEFAULT_STUDENT_TARGET_ELITE_FRACTION,
    DEFAULT_STUDENT_TARGET_ELITE_K,
    DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT,
    DEFAULT_STUDENT_TARGET_MIXTURE_MODE,
    DEFAULT_STUDENT_TEACHER_SCORE_CLIP,
    DEFAULT_STUDENT_TEACHER_SCORE_WEIGHT,
    DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION,
    STUDENT_TARGET_MIXTURE_MODES,
)

PIPELINE_VERSION = "canonical_multi_family_gipo_pipeline"
DEFAULT_STAGES = (
    "data_prep",
    "backbone_training",
    "ser_summaries",
    "schedule_rows_seen",
    "schedule_rows_unseen",
    "gipo_student_seen_only_zero_shot",
    "gipo_student_seen_plus_unseen_pseudo",
    "locked_test_reports",
)
GIPO_ABLATION_STUDENT_STAGE = "gipo_ablation_students"
GIPO_ABLATION_LOCKED_TEST_STAGE = "gipo_ablation_locked_test_reports"
DEFAULT_ABLATION_FIRST_STAGES = (
    "data_prep",
    "ser_summaries",
    "schedule_rows_seen",
    "schedule_rows_unseen",
    GIPO_ABLATION_STUDENT_STAGE,
    GIPO_ABLATION_LOCKED_TEST_STAGE,
)
PIPELINE_STAGE_ORDER = (*DEFAULT_STAGES, GIPO_ABLATION_STUDENT_STAGE, GIPO_ABLATION_LOCKED_TEST_STAGE)


@dataclass(frozen=True)
class StageCommand:
    stage: str
    commands: List[List[str]]
    manifest_name: str


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_int_csv(text: str, default: Sequence[int]) -> List[int]:
    raw = _parse_csv(text)
    return [int(part) for part in raw] if raw else [int(value) for value in default]


def _effective_stage_names(args: argparse.Namespace) -> List[str]:
    requested = _parse_csv(str(args.stages))
    if requested:
        requested_set = set(requested)
        unknown = sorted(requested_set - set(PIPELINE_STAGE_ORDER))
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {', '.join(unknown)}")
        return [stage for stage in PIPELINE_STAGE_ORDER if stage in requested_set]
    if bool(getattr(args, "ablation_first", False)):
        return list(DEFAULT_ABLATION_FIRST_STAGES)
    return list(DEFAULT_STAGES)


def _json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        parts = tuple(str(part) for part in resolved.parts)
        for marker in ("projects", "tmp"):
            if marker in parts:
                tail = parts[parts.index(marker) + 1 :]
                return "/".join(tail) if tail else resolved.name
        tail = parts[-min(8, len(parts)) :]
        if tail and (tail[0].endswith(":") or tail[0] == resolved.anchor):
            tail = tail[1:]
        return "/".join(str(part) for part in tail)


def _display_command(command: Sequence[str]) -> List[str]:
    out: List[str] = []
    for token in command:
        text = str(token)
        path = Path(text).expanduser()
        if path.is_absolute():
            out.append(_display_path(path))
        else:
            out.append(text)
    return out


def _display_stage(entry: StageCommand) -> Dict[str, Any]:
    return {
        "stage": entry.stage,
        "manifest_name": entry.manifest_name,
        "commands": [_display_command(command) for command in entry.commands],
    }


def _git_head_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _teacher_target_args_for_family(dataset: str) -> List[str]:
    specs = objective_specs_for_family(scenario_family_for_key(str(dataset)))
    target_keys = [str(spec.utility_key) for spec in specs]
    weights = [f"{spec.utility_key}={float(spec.weight):g}" for spec in specs]
    return [
        "--teacher_metric_target_keys",
        ",".join(target_keys),
        "--teacher_utility_weights",
        ",".join(weights),
    ]


def _student_objective_args(args: argparse.Namespace, override: GipoAblationArm | None = None) -> List[Any]:
    source: Any = override if override is not None else args
    values: List[Any] = [
        "--student_teacher_score_weight",
        float(source.student_teacher_score_weight),
        "--student_teacher_score_warmup_fraction",
        float(source.student_teacher_score_warmup_fraction),
        "--student_target_mixture_mode",
        str(source.student_target_mixture_mode),
        "--student_target_elite_fraction",
        float(source.student_target_elite_fraction),
        "--student_target_elite_k",
        int(source.student_target_elite_k),
        "--student_target_elite_min_count",
        int(source.student_target_elite_min_count),
        "--student_target_elite_blend_all_weight",
        float(source.student_target_elite_blend_all_weight),
    ]
    if bool(source.student_teacher_score_include_pseudo):
        values.append("--student_teacher_score_include_pseudo")
    return values


def _validate_inputs_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = str(args.scenario_key or args.dataset)
    family = scenario_family_for_key(dataset)
    if int(args.synthetic_length) <= 0:
        raise ValueError("--synthetic_length must be positive.")
    effective_stages = set(_effective_stage_names(args))
    includes_backbone_training = "backbone_training" in effective_stages
    if bool(getattr(args, "use_provided_backbones", False)) or str(getattr(args, "backbone_package_root", "") or "").strip():
        if includes_backbone_training:
            raise ValueError("Provided-backbone mode cannot include backbone_training; run downstream stages only.")
        manifest_path = resolve_project_path(str(args.backbone_manifest))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Provided-backbone mode requires an existing backbone manifest: {manifest_path}")
        provided_validation = validate_provided_backbone_manifest(
            manifest_path,
            scenario_key=str(args.scenario_key or args.dataset),
            benchmark_family=family,
        )
        if provided_validation["status"] != "complete":
            raise ValueError("Invalid provided backbone manifest:\n- " + "\n- ".join(provided_validation["errors"]))
    elif bool(getattr(args, "ablation_first", False)) and not includes_backbone_training and not bool(getattr(args, "dry_run", False)):
        manifest_path = resolve_project_path(str(args.backbone_manifest))
        provided_validation = validate_provided_backbone_manifest(
            manifest_path,
            scenario_key=str(args.scenario_key or args.dataset),
            benchmark_family=family,
        )
        if provided_validation["status"] != "complete":
            raise ValueError("Ablation-first mode requires ready provided backbones:\n- " + "\n- ".join(provided_validation["errors"]))
    if includes_backbone_training and family in {SCENARIO_FAMILY_FORECAST, SCENARIO_FAMILY_CONDITIONAL_GENERATION}:
        requested_manifest = resolve_project_path(str(args.backbone_manifest))
        default_manifest = default_backbone_manifest_path().resolve()
        if requested_manifest != default_manifest:
            raise ValueError(
                "Temporal full-pipeline backbone training materializes the canonical backbone manifest at "
                f"{_display_path(default_manifest)}; do not override --backbone_manifest for runs that include "
                "backbone_training."
            )
    return {
        "status": "complete",
        "scenario_key": dataset,
        "benchmark_family": family,
        "synthetic_length": int(args.synthetic_length),
    }


def _protocol_payload(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = str(args.scenario_key or args.dataset)
    plan = experiment_plan_by_key().get(dataset)
    effective_stages = _effective_stage_names(args)
    includes_ablations = any(stage in {GIPO_ABLATION_STUDENT_STAGE, GIPO_ABLATION_LOCKED_TEST_STAGE} for stage in effective_stages)
    ablation_preset = str(getattr(args, "gipo_ablation_preset", DEFAULT_GIPO_ABLATION_PRESET))
    return {
        "version": PIPELINE_VERSION,
        "scenario_key": dataset,
        "benchmark_family": "" if plan is None else str(plan.benchmark_family),
        "stages": effective_stages,
        "ablation_first": bool(getattr(args, "ablation_first", False)),
        "gipo_ablation_preset": ablation_preset if includes_ablations else "",
        "gipo_ablation_arms": [arm.manifest_record() for arm in gipo_ablation_arms(ablation_preset)] if includes_ablations else [],
        "backbone_steps": int(args.backbone_steps),
        "checkpoint_steps": _parse_int_csv(str(args.checkpoint_steps), CANONICAL_CHECKPOINT_STEPS),
        "seen_nfes": _parse_int_csv(str(args.seen_nfes), CANONICAL_SEEN_NFES),
        "unseen_nfes": _parse_int_csv(str(args.unseen_nfes), CANONICAL_UNSEEN_NFES),
        "schedules": _parse_csv(str(args.schedule_keys)) or list(CANONICAL_SUPERVISION_SCHEDULE_KEYS),
        "context_sample_count": int(args.context_sample_count),
        "student_teacher_score_weight": float(args.student_teacher_score_weight),
        "student_teacher_score_warmup_fraction": float(args.student_teacher_score_warmup_fraction),
        "student_teacher_score_clip": float(DEFAULT_STUDENT_TEACHER_SCORE_CLIP),
        "student_teacher_score_protocol": "late_ramped_per_cell_teacher_utility_z_score",
        "student_teacher_score_include_pseudo": bool(args.student_teacher_score_include_pseudo),
        "student_target_mixture_mode": str(args.student_target_mixture_mode),
        "student_target_elite_fraction": float(args.student_target_elite_fraction),
        "student_target_elite_k": int(args.student_target_elite_k),
        "student_target_elite_min_count": int(args.student_target_elite_min_count),
        "student_target_elite_blend_all_weight": float(args.student_target_elite_blend_all_weight),
        "synthetic_length": int(args.synthetic_length),
        "locked_test_rows": str(args.locked_test_rows),
        "dataset_root": _display_path(str(args.dataset_root)),
        "shared_backbone_root": _display_path(str(args.shared_backbone_root)),
        "backbone_manifest": _display_path(str(args.backbone_manifest)),
        "cryptos_path": _display_path(str(args.cryptos_path)),
        "lobster_synthetic_profile_path": _display_path(str(args.lobster_synthetic_profile_path)),
        "long_term_st_path": _display_path(str(args.long_term_st_path)),
        "molecule_group_root": _display_path(str(getattr(args, "molecule_group_root", "") or default_molecule_group_root())),
        "molecule_backbone_root": _display_path(str(getattr(args, "molecule_backbone_root", "") or (project_outputs_root() / "molecule_3d_backbones"))),
        "backbone_package": backbone_package_protocol_payload(args),
    }


def _has_ablation_stage(commands: Sequence[StageCommand]) -> bool:
    ablation_stages = {GIPO_ABLATION_STUDENT_STAGE, GIPO_ABLATION_LOCKED_TEST_STAGE}
    return any(entry.stage in ablation_stages for entry in commands)


def _ablation_root(run_root: Path, preset: str) -> Path:
    if run_root.name == str(preset) and run_root.parent.name == "gipo_ablations":
        return run_root
    return run_root / "gipo_ablations" / str(preset)


def _run_relative_path(run_root: Path, path: Path) -> str:
    rel = path.relative_to(run_root).as_posix()
    return "." if rel == "." else rel


def _resolve_run_root(args: argparse.Namespace) -> Path:
    explicit = str(args.run_root).strip()
    if explicit:
        return resolve_project_path(explicit)
    scenario_root = project_outputs_root() / "full_pipeline" / str(args.scenario_key or args.dataset)
    if bool(getattr(args, "ablation_first", False)):
        preset = str(getattr(args, "gipo_ablation_preset", DEFAULT_GIPO_ABLATION_PRESET))
        return scenario_root / "gipo_ablations" / preset
    return scenario_root


def _build_ablation_manifest(
    args: argparse.Namespace,
    run_root: Path,
    *,
    protocol_hash: str,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    dataset = str(args.scenario_key or args.dataset)
    family = scenario_family_for_key(dataset)
    preset = str(getattr(args, "gipo_ablation_preset", DEFAULT_GIPO_ABLATION_PRESET))
    root = _ablation_root(run_root, preset)
    arms = []
    for arm in gipo_ablation_arms(preset):
        train_root = root / arm.arm_id / "gipo" / arm.student_training_mode
        report_root = root / arm.arm_id / "locked_test_reports" / arm.student_training_mode
        arms.append(
            {
                **arm.manifest_record(),
                "outputs": {
                    "training_summary": _run_relative_path(run_root, train_root / "gipo_training_summary.json"),
                    "student_checkpoint": _run_relative_path(run_root, train_root / "gipo_student.pt"),
                    "locked_test_report_root": _run_relative_path(run_root, report_root),
                },
            }
        )
    manifest = {
        "artifact": "gipo_ablation_manifest",
        "schema_version": "genode_gipo_ablation_v1",
        "status": status,
        "ablation_set_id": preset,
        "scenario_key": dataset,
        "benchmark_family": family,
        "commit_sha": _git_head_commit(),
        "protocol_hash": protocol_hash,
        "ablation_root": _run_relative_path(run_root, root),
        "checkpoint_steps": _parse_int_csv(str(args.checkpoint_steps), CANONICAL_CHECKPOINT_STEPS),
        "seen_nfes": _parse_int_csv(str(args.seen_nfes), CANONICAL_SEEN_NFES),
        "unseen_nfes": _parse_int_csv(str(args.unseen_nfes), CANONICAL_UNSEEN_NFES),
        "schedule_keys": _parse_csv(str(args.schedule_keys)) or list(CANONICAL_SUPERVISION_SCHEDULE_KEYS),
        "arms": arms,
        "arm_count": int(len(arms)),
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def _status_path(run_root: Path) -> Path:
    return run_root / "status.json"


def _load_existing_status(run_root: Path) -> Dict[str, Any]:
    path = _status_path(run_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run_root(run_root: Path, protocol_hash: str, *, resume: bool, overwrite: bool) -> None:
    existing = _load_existing_status(run_root)
    if not existing:
        return
    existing_hash = str(existing.get("protocol_hash", "") or "")
    if existing_hash == str(protocol_hash):
        return
    if bool(overwrite):
        return
    if bool(resume):
        raise ValueError(
            f"Cannot resume {run_root} with different protocol_hash; existing={existing_hash}, new={protocol_hash}."
        )
    raise ValueError(f"Run root {run_root} already has status.json; pass --resume or --overwrite explicitly.")


def _python_module_command(module: str, args: Iterable[str]) -> List[str]:
    return [sys.executable, "-m", module, *[str(value) for value in args]]


def _fixed_schedule_keys_for_runner(schedule_keys: Sequence[str]) -> List[str]:
    return [str(key) for key in schedule_keys if "ser_ptg_local_defect" not in str(key)]


def _ser_schedule_keys_for_runner(schedule_keys: Sequence[str]) -> List[str]:
    return [str(key) for key in schedule_keys if "ser_ptg_local_defect" in str(key)]


def _schedule_keys_for_gipo(schedule_keys: Sequence[str]) -> str:
    return ",".join(str(key) for key in schedule_keys)


def _schedule_dataset_args(dataset: str) -> List[str]:
    family = scenario_family_for_key(str(dataset))
    if family == SCENARIO_FAMILY_FORECAST:
        return ["--forecast_datasets", str(dataset), "--conditional_generation_datasets", "", "--molecule_datasets", ""]
    if family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        return ["--forecast_datasets", "", "--conditional_generation_datasets", str(dataset), "--molecule_datasets", ""]
    if family == SCENARIO_FAMILY_MOLECULE:
        return ["--forecast_datasets", "", "--conditional_generation_datasets", "", "--molecule_datasets", str(dataset)]
    raise ValueError(f"Unsupported scenario family {family!r} for {dataset!r}.")


def _data_path_args(args: argparse.Namespace) -> List[str]:
    return [
        "--dataset_root",
        str(args.dataset_root),
        "--shared_backbone_root",
        str(args.shared_backbone_root),
        "--cryptos_path",
        str(args.cryptos_path),
        "--lobster_synthetic_profile_path",
        str(args.lobster_synthetic_profile_path),
        "--long_term_st_path",
        str(args.long_term_st_path),
    ]


def _temporal_training_data_path_args(args: argparse.Namespace) -> List[str]:
    return [
        "--dataset_root",
        str(args.dataset_root),
        "--cryptos_path",
        str(args.cryptos_path),
        "--lobster_synthetic_profile_path",
        str(args.lobster_synthetic_profile_path),
        "--long_term_st_path",
        str(args.long_term_st_path),
    ]


def _molecule_processed_dir_arg(group_root: Path, dataset: str, member: Mapping[str, Any]) -> Path:
    return group_root / str(dataset) / str(member["processed_dir"])


def _backbone_training_commands(args: argparse.Namespace, dataset: str, checkpoints: str) -> List[List[str]]:
    family = scenario_family_for_key(str(dataset))
    if family != SCENARIO_FAMILY_MOLECULE:
        return [
            _python_module_command(
                "genode.training.train_backbone",
                [
                    "--dataset",
                    dataset,
                    "--steps",
                    int(args.backbone_steps),
                    "--checkpoint_steps",
                    checkpoints,
                    "--checkpoint_export_mode",
                    "exact_budget",
                    "--synthetic_length",
                    int(args.synthetic_length),
                    *_temporal_training_data_path_args(args),
                    "--device",
                    str(args.device),
                ],
            )
        ]
    group_root = resolve_project_path(str(getattr(args, "molecule_group_root", "") or default_molecule_group_root()))
    manifest_path = group_root / str(dataset) / "group_manifest.json"
    if not manifest_path.exists() and bool(getattr(args, "dry_run", False)):
        return [["internal", "expand_molecule_member_backbones", f"--scenario_key={dataset}", f"--molecule_group_root={_display_path(group_root)}"]]
    manifest = load_molecule_group_manifest(str(dataset), group_root)
    commands: List[List[str]] = []
    for member in trainable_molecule_group_members(manifest):
        commands.append(
            _python_module_command(
                "genode.training.train_molecule_backbone",
                [
                    "--dataset_key",
                    dataset,
                    "--stratum",
                    str(member["stratum"]),
                    "--processed_dir",
                    _molecule_processed_dir_arg(group_root, str(dataset), member),
                    "--out_dir",
                    str(args.molecule_backbone_root),
                    "--molecule_group_root",
                    group_root,
                    "--steps",
                    int(args.backbone_steps),
                    "--budget_steps",
                    checkpoints,
                    "--device",
                    str(args.device),
                    "--no_prepare_data",
                ],
            )
        )
    if not commands:
        raise ValueError(f"Molecule group {dataset!r} has no trainable member strata.")
    return commands


def _build_stage_commands(args: argparse.Namespace, run_root: Path) -> List[StageCommand]:
    dataset = str(args.scenario_key or args.dataset)
    checkpoint_values = _parse_int_csv(str(args.checkpoint_steps), CANONICAL_CHECKPOINT_STEPS)
    checkpoints = ",".join(str(value) for value in checkpoint_values)
    seen_nfes = ",".join(str(value) for value in _parse_int_csv(str(args.seen_nfes), CANONICAL_SEEN_NFES))
    unseen_nfes = ",".join(str(value) for value in _parse_int_csv(str(args.unseen_nfes), CANONICAL_UNSEEN_NFES))
    schedules = _parse_csv(str(args.schedule_keys)) or list(CANONICAL_SUPERVISION_SCHEDULE_KEYS)
    fixed_schedules = ",".join(_fixed_schedule_keys_for_runner(schedules))
    ser_schedules = ",".join(_ser_schedule_keys_for_runner(schedules))
    support_schedules = _schedule_keys_for_gipo(schedules)
    rows_root = run_root / "schedule_rows"
    ser_root = run_root / "ser_summaries"
    gipo_root = run_root / "gipo"
    ablation_preset = str(getattr(args, "gipo_ablation_preset", DEFAULT_GIPO_ABLATION_PRESET))
    gipo_ablation_root = _ablation_root(run_root, ablation_preset)
    ablation_arms = gipo_ablation_arms(ablation_preset)
    split_phases = ("train_tuning", "validation_tuning", "locked_test")
    role_nfes = {"seen": seen_nfes, "unseen": unseen_nfes}

    def _rows_dir(role: str, phase: str, checkpoint_step: int) -> Path:
        return rows_root / role / phase / f"{int(checkpoint_step)}_steps"

    def _ser_summary_path(role: str, checkpoint_step: int) -> Path:
        return ser_root / role / f"{int(checkpoint_step)}_steps" / "ser_ptg_schedule_summary.json"

    def _csv_list(role: str, phase: str, name: str) -> str:
        return ",".join(str(_rows_dir(role, phase, int(step)) / name) for step in checkpoint_values)

    def _ser_summary_list(role: str) -> str:
        if not ser_schedules:
            return ""
        return ",".join(str(_ser_summary_path(role, int(step))) for step in checkpoint_values)

    def _gipo_train_command(mode: str, out_dir: Path, arm: GipoAblationArm | None = None) -> List[str]:
        command_args: List[Any] = [
            "--rows_csv",
            _csv_list("seen", "train_tuning", "context_rows.csv"),
            "--context_embeddings_npz",
            _csv_list("seen", "train_tuning", "context_embeddings.npz"),
            "--schedule_summary_json",
            _ser_summary_list("seen"),
            "--support_schedule_keys",
            support_schedules,
            *_teacher_target_args_for_family(dataset),
            *_student_objective_args(args, arm),
        ]
        if mode == STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO:
            pseudo_weight = CANONICAL_PSEUDO_TARGET_WEIGHT if arm is None else float(arm.student_pseudo_target_weight)
            command_args.extend(
                [
                    "--student_pseudo_rows_csv",
                    _csv_list("unseen", "train_tuning", "context_rows.csv"),
                    "--student_pseudo_context_embeddings_npz",
                    _csv_list("unseen", "train_tuning", "context_embeddings.npz"),
                    "--student_pseudo_schedule_summary_json",
                    _ser_summary_list("unseen"),
                    "--student_pseudo_target_weight",
                    f"{float(pseudo_weight):g}",
                ]
            )
        command_args.extend(
            [
                "--context_sample_count",
                int(args.context_sample_count),
                "--out_dir",
                out_dir,
            ]
        )
        return _python_module_command("genode.gipo.train_gipo", command_args)

    def _schedule_row_commands(role: str, nfe_values: str) -> List[List[str]]:
        commands: List[List[str]] = []
        for phase in split_phases:
            for checkpoint_step in checkpoint_values:
                summary_args: List[Any] = []
                if ser_schedules:
                    summary_args = [
                        "--schedule_summary_json",
                        _ser_summary_path(role, int(checkpoint_step)),
                        "--summary_scheduler_names",
                        ser_schedules,
                    ]
                commands.append(
                    _python_module_command(
                        "genode.evaluation.diffusion_flow_time_reparameterization",
                        [
                            *_schedule_dataset_args(dataset),
                            "--nfe_role",
                            role,
                            "--target_nfe_values",
                            nfe_values,
                            "--checkpoint_steps",
                            int(checkpoint_step),
                            "--baseline_scheduler_names",
                            fixed_schedules,
                            *summary_args,
                            "--split_phase",
                            phase,
                            "--context_sample_count",
                            int(args.context_sample_count),
                            "--molecule_group_root",
                            str(args.molecule_group_root),
                            "--backbone_manifest",
                            str(args.backbone_manifest),
                            *_data_path_args(args),
                            "--write_context_rows",
                            "--allow_execute",
                            "--out_root",
                            _rows_dir(role, phase, int(checkpoint_step)),
                            "--device",
                            str(args.device),
                        ],
                    )
                )
        return commands

    def _locked_test_report_commands_for_student(
        *,
        student_root: Path,
        report_root: Path,
    ) -> List[List[str]]:
        commands: List[List[str]] = []
        family = scenario_family_for_key(dataset)
        for role, nfe_values in role_nfes.items():
            for checkpoint_step in checkpoint_values:
                report_dir = report_root / role / f"{int(checkpoint_step)}_steps"
                locked_dir = _rows_dir(role, "locked_test", int(checkpoint_step))
                commands.append(
                    _python_module_command(
                        "genode.gipo.report_locked_test",
                        [
                            "--gipo_student_checkpoint",
                            student_root / "gipo_student.pt",
                            "--training_summary",
                            student_root / "gipo_training_summary.json",
                            "--context_rows",
                            locked_dir / "context_rows.csv",
                            "--context_embeddings_npz",
                            locked_dir / "context_embeddings.npz",
                            "--baseline_rows",
                            locked_dir / "context_rows.csv",
                            "--comparator_rows",
                            locked_dir / "context_rows.csv",
                            "--benchmark_family",
                            family,
                            "--dataset",
                            dataset,
                            "--split_phase",
                            "locked_test",
                            "--solver_names",
                            ",".join(str(key) for key in CANONICAL_SOLVER_KEYS),
                            "--target_nfe_values",
                            nfe_values,
                            "--otflow_train_steps",
                            int(checkpoint_step),
                            "--steps",
                            int(checkpoint_step),
                            "--molecule_group_root",
                            str(args.molecule_group_root),
                            "--backbone_manifest",
                            str(args.backbone_manifest),
                            *_data_path_args(args),
                            "--out_dir",
                            report_dir,
                            "--device",
                            str(args.device),
                        ],
                    )
                )
        return commands

    def _locked_test_report_commands() -> List[List[str]]:
        commands: List[List[str]] = []
        for mode in (STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO):
            commands.extend(
                _locked_test_report_commands_for_student(
                    student_root=gipo_root / mode,
                    report_root=run_root / "locked_test_reports" / mode,
                )
            )
        return commands

    def _gipo_ablation_student_commands() -> List[List[str]]:
        return [
            _gipo_train_command(
                arm.student_training_mode,
                gipo_ablation_root / arm.arm_id / "gipo" / arm.student_training_mode,
                arm,
            )
            for arm in ablation_arms
        ]

    def _gipo_ablation_locked_test_commands() -> List[List[str]]:
        commands: List[List[str]] = []
        for arm in ablation_arms:
            commands.extend(
                _locked_test_report_commands_for_student(
                    student_root=gipo_ablation_root / arm.arm_id / "gipo" / arm.student_training_mode,
                    report_root=gipo_ablation_root / arm.arm_id / "locked_test_reports" / arm.student_training_mode,
                )
            )
        return commands

    ser_commands = [
        _python_module_command(
            "genode.gipo.ser_ptg_reference",
            [
                "--dataset",
                dataset,
                "--solver_names",
                ",".join(str(key) for key in CANONICAL_SOLVER_KEYS),
                "--target_nfe_values",
                role_nfes[role],
                "--reference_split",
                "train_tuning",
                "--otflow_train_steps",
                int(checkpoint_step),
                "--steps",
                int(checkpoint_step),
                "--context_sample_count",
                int(args.context_sample_count),
                *_data_path_args(args),
                "--backbone_manifest",
                str(args.backbone_manifest),
                "--molecule_group_root",
                str(args.molecule_group_root),
                "--out_dir",
                ser_root / role / f"{int(checkpoint_step)}_steps",
                "--device",
                str(args.device),
            ],
        )
        for role in ("seen", "unseen")
        for checkpoint_step in checkpoint_values
        if ser_schedules
    ]
    commands = [
        StageCommand(
            "data_prep",
            [
                [
                    "internal",
                    "validate_inputs",
                    f"--scenario_key={dataset}",
                    f"--synthetic_length={int(args.synthetic_length)}",
                ]
            ] if bool(args.dry_run) else [],
            "data_prep_manifest.json",
        ),
        StageCommand(
            "backbone_training",
            _backbone_training_commands(args, dataset, checkpoints),
            "backbone_training_manifest.json",
        ),
        StageCommand(
            "ser_summaries",
            ser_commands,
            "ser_summaries_manifest.json",
        ),
        StageCommand(
            "schedule_rows_seen",
            _schedule_row_commands("seen", seen_nfes),
            "schedule_rows_seen_manifest.json",
        ),
        StageCommand(
            "schedule_rows_unseen",
            _schedule_row_commands("unseen", unseen_nfes),
            "schedule_rows_unseen_manifest.json",
        ),
        StageCommand(
            "gipo_student_seen_only_zero_shot",
            [_gipo_train_command(STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT, gipo_root / STUDENT_TRAINING_MODE_SEEN_ONLY_ZERO_SHOT)],
            "gipo_seen_only_manifest.json",
        ),
        StageCommand(
            "gipo_student_seen_plus_unseen_pseudo",
            [_gipo_train_command(STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO, gipo_root / STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_PSEUDO)],
            "gipo_seen_plus_pseudo_manifest.json",
        ),
        StageCommand(
            "locked_test_reports",
            _locked_test_report_commands(),
            "locked_test_reports_manifest.json",
        ),
        StageCommand(
            GIPO_ABLATION_STUDENT_STAGE,
            _gipo_ablation_student_commands(),
            "gipo_ablation_students_manifest.json",
        ),
        StageCommand(
            GIPO_ABLATION_LOCKED_TEST_STAGE,
            _gipo_ablation_locked_test_commands(),
            "gipo_ablation_locked_test_reports_manifest.json",
        ),
    ]
    stage_filter = set(_effective_stage_names(args))
    return [entry for entry in commands if entry.stage in stage_filter]


def run_full_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    args = apply_backbone_package_to_args(args)
    run_root = _resolve_run_root(args)
    run_root.mkdir(parents=True, exist_ok=True)
    preflight = _validate_inputs_preflight(args)
    protocol = _protocol_payload(args)
    protocol_hash = _json_hash(protocol)
    _validate_run_root(run_root, protocol_hash, resume=bool(args.resume), overwrite=bool(args.overwrite))
    commands = _build_stage_commands(args, run_root)
    has_ablation_stage = _has_ablation_stage(commands)
    if has_ablation_stage:
        ablation_manifest = _build_ablation_manifest(
            args,
            run_root,
            protocol_hash=protocol_hash,
            status="dry_run" if bool(args.dry_run) else "running",
        )
        _write_json(_ablation_root(run_root, str(args.gipo_ablation_preset)) / "ablation_manifest.json", ablation_manifest)
    _write_json(run_root / "protocol.json", {**protocol, "protocol_hash": protocol_hash})
    _write_json(
        _status_path(run_root),
        {
            "version": PIPELINE_VERSION,
            "status": "running" if not bool(args.dry_run) else "dry_run",
            "protocol_hash": protocol_hash,
            "run_root": _display_path(run_root),
            "stages": [entry.stage for entry in commands],
            "completed_stages": [],
            "dry_run": bool(args.dry_run),
            "preflight": preflight,
        },
    )
    completed: List[str] = []
    for entry in commands:
        manifest = {
            "stage": entry.stage,
            "protocol_hash": protocol_hash,
            "commands": [_display_command(command) for command in entry.commands],
            "dry_run": bool(args.dry_run),
            "command_results": [],
        }
        if not bool(args.dry_run):
            for command_idx, command in enumerate(entry.commands):
                if command and command[0] == "internal":
                    raise RuntimeError(f"Unresolved internal pipeline command in non-dry-run: {_display_command(command)}")
                log_path = run_root / "logs" / f"{entry.stage}_{command_idx}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as log_fh:
                    result = subprocess.run(command, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
                command_result = {
                    "command_index": int(command_idx),
                    "returncode": int(result.returncode),
                    "log_path": str(log_path.relative_to(run_root)),
                }
                manifest["command_results"].append(command_result)
                if int(result.returncode) != 0:
                    manifest["status"] = "failed"
                    _write_json(run_root / entry.manifest_name, manifest)
                    if has_ablation_stage:
                        failed_ablation_manifest = _build_ablation_manifest(
                            args,
                            run_root,
                            protocol_hash=protocol_hash,
                            status="failed",
                            extra={
                                "failed_stage": entry.stage,
                                "failed_command_index": int(command_idx),
                                "failed_log_path": str(log_path.relative_to(run_root)),
                            },
                        )
                        _write_json(
                            _ablation_root(run_root, str(args.gipo_ablation_preset)) / "ablation_manifest.json",
                            failed_ablation_manifest,
                        )
                    _write_json(
                        _status_path(run_root),
                        {
                            "version": PIPELINE_VERSION,
                            "status": "failed",
                            "failed_stage": entry.stage,
                            "failed_command_index": int(command_idx),
                            "protocol_hash": protocol_hash,
                            "completed_stages": completed,
                        },
                    )
                    raise RuntimeError(f"Pipeline stage {entry.stage!r} failed; see {log_path}.")
        manifest["status"] = "planned" if bool(args.dry_run) else "complete"
        _write_json(run_root / entry.manifest_name, manifest)
        completed.append(entry.stage)
        _write_json(
            _status_path(run_root),
            {
                "version": PIPELINE_VERSION,
                "status": "dry_run" if bool(args.dry_run) else "running",
                "protocol_hash": protocol_hash,
                "completed_stages": completed,
                "remaining_stages": [cmd.stage for cmd in commands if cmd.stage not in set(completed)],
            },
        )
    summary = {
        "version": PIPELINE_VERSION,
        "status": "dry_run" if bool(args.dry_run) else "complete",
        "protocol_hash": protocol_hash,
        "stage_count": int(len(commands)),
        "stages": [_display_stage(entry) for entry in commands],
        "run_root": _display_path(run_root),
    }
    _write_json(run_root / "pipeline_summary.json", summary)
    _write_json(_status_path(run_root), summary)
    if has_ablation_stage:
        final_ablation_manifest = _build_ablation_manifest(
            args,
            run_root,
            protocol_hash=protocol_hash,
            status="dry_run" if bool(args.dry_run) else "complete",
        )
        _write_json(_ablation_root(run_root, str(args.gipo_ablation_preset)) / "ablation_manifest.json", final_ablation_manifest)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical multi-family GIPO pipeline with restartable status.")
    parser.add_argument("--scenario_key", default="lobster_synthetic")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--run_root", default="")
    parser.add_argument("--stages", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone_steps", type=int, default=20_000)
    parser.add_argument("--checkpoint_steps", default=",".join(str(value) for value in CANONICAL_CHECKPOINT_STEPS))
    parser.add_argument("--seen_nfes", default=",".join(str(value) for value in CANONICAL_SEEN_NFES))
    parser.add_argument("--unseen_nfes", default=",".join(str(value) for value in CANONICAL_UNSEEN_NFES))
    parser.add_argument("--schedule_keys", default=",".join(CANONICAL_SUPERVISION_SCHEDULE_KEYS))
    parser.add_argument("--context_sample_count", type=int, default=CANONICAL_CONTEXT_SAMPLE_COUNT)
    parser.add_argument("--student_teacher_score_weight", type=float, default=DEFAULT_STUDENT_TEACHER_SCORE_WEIGHT)
    parser.add_argument("--student_teacher_score_warmup_fraction", type=float, default=DEFAULT_STUDENT_TEACHER_SCORE_WARMUP_FRACTION)
    parser.add_argument("--student_teacher_score_include_pseudo", action="store_true", default=False)
    parser.add_argument("--student_target_mixture_mode", choices=STUDENT_TARGET_MIXTURE_MODES, default=DEFAULT_STUDENT_TARGET_MIXTURE_MODE)
    parser.add_argument("--student_target_elite_fraction", type=float, default=DEFAULT_STUDENT_TARGET_ELITE_FRACTION)
    parser.add_argument("--student_target_elite_k", type=int, default=DEFAULT_STUDENT_TARGET_ELITE_K)
    parser.add_argument("--student_target_elite_min_count", type=int, default=DEFAULT_STUDENT_TARGET_ELITE_MIN_COUNT)
    parser.add_argument("--student_target_elite_blend_all_weight", type=float, default=DEFAULT_STUDENT_TARGET_ELITE_BLEND_ALL_WEIGHT)
    parser.add_argument("--synthetic_length", type=int, default=2_000_000)
    parser.add_argument("--locked_test_rows", default="")
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(project_outputs_root() / "otflow_backbones"))
    parser.add_argument("--backbone_manifest", default=str(default_backbone_manifest_path()))
    parser.add_argument("--cryptos_path", default=str(default_cryptos_data_path()))
    parser.add_argument("--lobster_synthetic_profile_path", default=str(default_lobster_synthetic_profile_path()))
    parser.add_argument("--long_term_st_path", default=str(default_long_term_st_data_path()))
    parser.add_argument("--molecule_group_root", default=str(default_molecule_group_root()))
    parser.add_argument("--molecule_backbone_root", default=str(project_outputs_root() / "molecule_3d_backbones"))
    parser.add_argument("--backbone_package_root", default="", help="Portable backbone package root to use for downstream-only GIPO stages.")
    parser.add_argument("--use_provided_backbones", action="store_true", default=False, help="Require existing packaged/provided backbones and refuse backbone_training stages.")
    parser.add_argument("--ablation_first", action="store_true", default=False, help="Run prerequisite artifacts plus GIPO ablations before stock GIPO stages.")
    parser.add_argument("--gipo_ablation_preset", choices=gipo_ablation_preset_choices(), default=DEFAULT_GIPO_ABLATION_PRESET)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def main() -> None:
    summary = run_full_pipeline(build_argparser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
