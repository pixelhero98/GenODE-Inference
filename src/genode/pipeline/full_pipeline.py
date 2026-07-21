from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from genode.cli import parse_csv, parse_int_csv
from genode.experiment_layout import (
    REFERENCE_CHECKPOINT_STEPS,
    REFERENCE_SEEN_NFES,
    REFERENCE_SUPERVISION_SCHEDULE_KEYS,
    REFERENCE_UNSEEN_NFES,
    REFERENCE_UNSEEN_TARGET_WEIGHT,
    TRAIN_TUNING_CONTEXT_SAMPLE_COUNT,
    LOCKED_TEST_PREVIEW_CONTEXTS,
    SCENARIO_FAMILY_CONDITIONAL_GENERATION,
    SCENARIO_FAMILY_FORECAST,
    SCENARIO_FAMILY_MOLECULE,
    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
    scenario_family_for_key,
)
from genode.data.molecule_xyz import molecule_group_root, load_molecule_group_manifest, trainable_molecule_group_members
from genode.data.otflow_experiment_plan import experiment_plan_by_key
from genode.data.otflow_paths import (
    backbone_manifest_path,
    cryptos_data_path,
    lobster_synthetic_profile_path,
    long_term_st_data_path,
    display_project_path,
    project_outputs_root,
    project_dataset_root,
    project_root,
    resolve_project_path,
)
from genode.solver_protocol import SUPPORTED_SOLVER_KEYS
from genode.backbone_packages import (
    apply_backbone_package_to_args,
    backbone_package_protocol_payload,
    load_portable_backbone_manifest,
    validate_backbone_artifact_checkpoint,
    validate_provided_backbone_manifest,
)
from genode.evaluation import diffusion_flow_time_reparameterization as schedule_runner
from genode.evaluation.diffusion_flow_time_reparameterization import SCHEDULE_CONTEXT_SELECTION_PROTOCOL
from genode.evaluation.fm_backbone_registry import (
    BACKBONE_NAME_OTFLOW,
    BACKBONE_NAME_OTFLOW_MOLECULE,
)
from genode.distillation.artifacts import DEMONSTRATION_MANIFEST_NAME
from genode.provenance import file_sha256
from genode.gipo.objectives import teacher_metric_profile_for_scenario, teacher_objective_specs_for_scenario
from genode.gipo.ser_ptg_reference import SER_PTG_EXAMPLE_SELECTION_PROTOCOL, SER_PTG_LOCAL_DEFECT_PROXY_PROTOCOL
from genode.gipo.evaluate_schedule_summary import SER_REFERENCE_SCHEDULE_KEYS
from genode.gipo.ablation_plan import (
    ABLATION_PRESET_ALL,
    GIPO_POLICY_KEY,
    GIPOStudentPolicy,
    ablation_preset_keys,
    ablation_student_policies,
    gipo_policy,
)
from genode.gipo.policy import (
    DEFAULT_STUDENT_TEACHER_SCORE_CLIP,
    STUDENT_TARGET_MIXTURE_MODES,
)
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS

PIPELINE_VERSION = "multi_family_gipo_pipeline"
INPUT_PREFLIGHT_STAGE = "input_preflight"
REFERENCE_PIPELINE_STAGES = (
    INPUT_PREFLIGHT_STAGE,
    "backbone_training",
    "ser_summaries",
    "schedule_rows_seen",
    "schedule_rows_unseen",
    "train_gipo",
    "report_gipo_locked_test",
)
ABLATION_STUDENT_STAGE = "train_ablation_students"
ABLATION_LOCKED_TEST_STAGE = "report_ablation_locked_test"
FLOW_MAP_COLLECTION_STAGE = "collect_flow_map_demonstrations"
FLOW_MAP_TRAINING_STAGE = "train_flow_map"
FLOW_MAP_EVALUATION_STAGE = "evaluate_flow_map"
FLOW_MAP_PIPELINE_STAGES = (
    FLOW_MAP_COLLECTION_STAGE,
    FLOW_MAP_TRAINING_STAGE,
    FLOW_MAP_EVALUATION_STAGE,
)
PIPELINE_STAGE_ORDER = (
    *REFERENCE_PIPELINE_STAGES,
    "train_unseen_target_student",
    ABLATION_STUDENT_STAGE,
    ABLATION_LOCKED_TEST_STAGE,
    *FLOW_MAP_PIPELINE_STAGES,
)
DEFAULT_GIPO_TEACHER_STEPS = 500
DEFAULT_GIPO_STUDENT_STEPS = 500
SCHEDULE_ROW_SPLIT_PHASES = ("train_tuning", "locked_test")
GIPO_POLICY_ROOT = Path("policies") / "gipo"
STUDENT_OBJECTIVE_CLI_FIELDS = (
    "student_teacher_score_weight",
    "student_teacher_score_warmup_fraction",
    "student_teacher_score_include_unseen_targets",
    "student_target_mixture_mode",
    "student_target_elite_fraction",
    "student_target_elite_k",
    "student_target_elite_min_count",
    "student_target_elite_blend_all_weight",
)


@dataclass(frozen=True)
class StageCommand:
    stage: str
    commands: List[List[str]]
    manifest_name: str


def _resolved_scenario_key(args: argparse.Namespace) -> str:
    return str(args.scenario_key)


def _locked_test_settings(args: argparse.Namespace) -> Dict[str, Any]:
    preview_enabled = bool(getattr(args, "locked_test_preview", False))
    requested_contexts = getattr(args, "locked_test_preview_contexts", None)
    if requested_contexts is not None and not preview_enabled:
        raise ValueError("--locked_test_preview_contexts requires --locked_test_preview.")
    if not preview_enabled:
        return {"mode": "full", "context_limit": None, "context_limit_scope": "none"}
    context_limit = (
        int(LOCKED_TEST_PREVIEW_CONTEXTS)
        if requested_contexts is None
        else int(requested_contexts)
    )
    if context_limit <= 0:
        raise ValueError("--locked_test_preview_contexts must be positive.")
    return {"mode": "preview", "context_limit": int(context_limit), "context_limit_scope": "per_seed"}


def _selected_stage_names(args: argparse.Namespace) -> List[str]:
    requested = parse_csv(args.stages)
    if requested:
        selection_flags = [
            flag
            for flag in ("include_ablations", "include_flow_map")
            if bool(getattr(args, flag, False))
        ]
        if selection_flags:
            options = ", ".join(f"--{flag}" for flag in selection_flags)
            raise ValueError(f"{options} cannot be combined with an explicit --stages list.")
        requested_set = set(requested)
        unknown = sorted(requested_set - set(PIPELINE_STAGE_ORDER))
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {', '.join(unknown)}")
        return [stage for stage in PIPELINE_STAGE_ORDER if stage in requested_set]
    selected = list(REFERENCE_PIPELINE_STAGES)
    if bool(getattr(args, "include_ablations", False)):
        selected.extend((ABLATION_STUDENT_STAGE, ABLATION_LOCKED_TEST_STAGE))
    if bool(getattr(args, "include_flow_map", False)):
        selected.extend(FLOW_MAP_PIPELINE_STAGES)
    return selected


def _json_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _display_path(path: str | Path, *, path_base: Path | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    if path_base is not None:
        try:
            return resolved.relative_to(Path(path_base).expanduser().resolve()).as_posix()
        except ValueError:
            pass
    return display_project_path(path)


def _display_optional_path(path: Any) -> str:
    text = str(path or "").strip()
    return _display_path(text) if text else ""


def _display_command(command: Sequence[str], *, path_base: Path | None = None) -> List[str]:
    out: List[str] = []
    for token in command:
        text = str(token)
        path = Path(text).expanduser()
        if path.is_absolute():
            out.append(_display_path(path, path_base=path_base))
        else:
            out.append(text)
    return out


def _display_stage(entry: StageCommand, *, path_base: Path | None = None) -> Dict[str, Any]:
    return {
        "stage": entry.stage,
        "manifest_name": entry.manifest_name,
        "commands": [_display_command(command, path_base=path_base) for command in entry.commands],
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


def _teacher_target_args_for_scenario(dataset: str) -> List[str]:
    specs = teacher_objective_specs_for_scenario(str(dataset))
    target_keys = [str(spec.utility_key) for spec in specs]
    weights = [f"{spec.utility_key}={float(spec.weight):g}" for spec in specs]
    return [
        "--teacher_metric_target_keys",
        ",".join(target_keys),
        "--teacher_utility_weights",
        ",".join(weights),
    ]


def _student_objective_args(args: argparse.Namespace, override: GIPOStudentPolicy | None = None) -> List[Any]:
    settings = _student_objective_settings_payload(override if override is not None else args)
    values: List[Any] = [
        "--student_teacher_score_weight",
        float(settings["student_teacher_score_weight"]),
        "--student_teacher_score_warmup_fraction",
        float(settings["student_teacher_score_warmup_fraction"]),
        "--student_target_mixture_mode",
        str(settings["student_target_mixture_mode"]),
        "--student_target_elite_fraction",
        float(settings["student_target_elite_fraction"]),
        "--student_target_elite_k",
        int(settings["student_target_elite_k"]),
        "--student_target_elite_min_count",
        int(settings["student_target_elite_min_count"]),
        "--student_target_elite_blend_all_weight",
        float(settings["student_target_elite_blend_all_weight"]),
    ]
    if bool(settings["student_teacher_score_include_unseen_targets"]):
        values.append("--student_teacher_score_include_unseen_targets")
    return values


def _student_objective_settings_payload(source: Any) -> Dict[str, Any]:
    defaults = gipo_policy()
    raw = {}
    for field in STUDENT_OBJECTIVE_CLI_FIELDS:
        value = getattr(source, field, None)
        raw[field] = getattr(defaults, field) if value is None else value
    return {
        "student_teacher_score_weight": float(raw["student_teacher_score_weight"]),
        "student_teacher_score_warmup_fraction": float(raw["student_teacher_score_warmup_fraction"]),
        "student_teacher_score_include_unseen_targets": bool(
            raw["student_teacher_score_include_unseen_targets"]
        ),
        "student_target_mixture_mode": str(raw["student_target_mixture_mode"]),
        "student_target_elite_fraction": float(raw["student_target_elite_fraction"]),
        "student_target_elite_k": int(raw["student_target_elite_k"]),
        "student_target_elite_min_count": int(raw["student_target_elite_min_count"]),
        "student_target_elite_blend_all_weight": float(raw["student_target_elite_blend_all_weight"]),
    }


def _student_objective_cli_overrides(args: argparse.Namespace) -> List[str]:
    return [field for field in STUDENT_OBJECTIVE_CLI_FIELDS if getattr(args, field) is not None]


def _validate_inputs_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = _resolved_scenario_key(args)
    family = scenario_family_for_key(dataset)
    effective_stages = _selected_stage_names(args)
    objective_overrides = _student_objective_cli_overrides(args)
    if objective_overrides and "train_unseen_target_student" not in effective_stages:
        raise ValueError(
            "Full-pipeline student objective flags only apply to the explicit "
            "train_unseen_target_student stage. The reference policy and ablation policies use fixed "
            "objective recipes; add that stage to --stages "
            f"or remove: {', '.join('--' + name for name in objective_overrides)}."
        )
    if int(args.synthetic_length) <= 0:
        raise ValueError("--synthetic_length must be positive.")
    if int(args.context_sample_count) <= 0 or int(args.context_sample_count) > TRAIN_TUNING_CONTEXT_SAMPLE_COUNT:
        raise ValueError(
            "--context_sample_count must be in "
            f"[1, {TRAIN_TUNING_CONTEXT_SAMPLE_COUNT}], got {int(args.context_sample_count)!r}."
        )
    if int(args.ser_calibration_batch_size) <= 0:
        raise ValueError("--ser_calibration_batch_size must be positive.")
    if int(args.ser_val_windows) < 0:
        raise ValueError("--ser_val_windows must be nonnegative.")
    if int(args.ser_train_tuning_max_examples) < 0:
        raise ValueError("--ser_train_tuning_max_examples must be nonnegative.")
    flow_map_stages = set(effective_stages) & set(FLOW_MAP_PIPELINE_STAGES)
    required_flow_map_paths: list[str] = []
    if flow_map_stages:
        required_flow_map_paths.append("flow_map_backbone_checkpoint")
    if FLOW_MAP_COLLECTION_STAGE in flow_map_stages:
        required_flow_map_paths.append("flow_map_contexts_npz")
    if (
        FLOW_MAP_EVALUATION_STAGE in flow_map_stages
        and FLOW_MAP_TRAINING_STAGE not in flow_map_stages
    ):
        required_flow_map_paths.append("flow_map_checkpoint")
    missing = [
        name
        for name in required_flow_map_paths
        if not str(getattr(args, name, "") or "").strip()
    ]
    if missing:
        raise ValueError(
            "The selected flow-map stages require: "
            + ", ".join(f"--{name}" for name in missing)
        )
    if (
        FLOW_MAP_TRAINING_STAGE in flow_map_stages
        and str(getattr(args, "flow_map_checkpoint", "") or "").strip()
    ):
        raise ValueError(
            "--flow_map_checkpoint is only an input for evaluation-only runs; "
            "training writes its checkpoint below the run root."
        )
    if FLOW_MAP_TRAINING_STAGE in flow_map_stages and FLOW_MAP_COLLECTION_STAGE not in flow_map_stages:
        demonstration_manifest = str(getattr(args, "flow_map_demonstration_manifest", "") or "").strip()
        if not demonstration_manifest:
            raise ValueError(
                "An explicit train_flow_map stage without collection requires "
                "--flow_map_demonstration_manifest."
            )
        required_flow_map_paths.append("flow_map_demonstration_manifest")
    explicit_demonstration_manifest = str(
        getattr(args, "flow_map_demonstration_manifest", "") or ""
    ).strip()
    if FLOW_MAP_COLLECTION_STAGE in flow_map_stages and explicit_demonstration_manifest:
        if Path(explicit_demonstration_manifest).name != DEMONSTRATION_MANIFEST_NAME:
            raise ValueError(
                f"--flow_map_demonstration_manifest must end in {DEMONSTRATION_MANIFEST_NAME!r}."
            )
    gipo_checkpoint = str(getattr(args, "flow_map_gipo_checkpoint", "") or "").strip()
    if (
        flow_map_stages
        and "train_gipo" not in effective_stages
        and not gipo_checkpoint
    ):
        raise ValueError(
            "Flow-map stages without the train_gipo stage require "
            "--flow_map_gipo_checkpoint."
        )
    if flow_map_stages and gipo_checkpoint:
        required_flow_map_paths.append("flow_map_gipo_checkpoint")
    quality_rows = str(getattr(args, "flow_map_quality_rows_csv", "") or "").strip()
    if flow_map_stages and quality_rows:
        required_flow_map_paths.append("flow_map_quality_rows_csv")
    if not bool(getattr(args, "dry_run", False)):
        for name in dict.fromkeys(required_flow_map_paths):
            path = resolve_project_path(str(getattr(args, name)))
            if not path.is_file():
                raise FileNotFoundError(f"--{name} does not exist: {path}")
    if flow_map_stages:
        for name in ("flow_map_rollouts_per_context", "flow_map_steps", "flow_map_batch_size"):
            if int(getattr(args, name)) <= 0:
                raise ValueError(f"--{name} must be positive.")
        if int(args.flow_map_bootstrap_samples) < 1_000:
            raise ValueError("--flow_map_bootstrap_samples must be at least 1000.")
        if not 0.0 < float(args.flow_map_familywise_alpha) < 1.0:
            raise ValueError("--flow_map_familywise_alpha must be between zero and one.")
    _locked_test_settings(args)
    effective_stages = set(_selected_stage_names(args))
    if "report_gipo_locked_test" in effective_stages:
        requested_schedules = set(
            parse_csv(args.schedule_keys) or REFERENCE_SUPERVISION_SCHEDULE_KEYS
        )
        missing_baselines = sorted(set(BASELINE_SCHEDULE_KEYS) - requested_schedules)
        missing_comparators = sorted(set(SER_REFERENCE_SCHEDULE_KEYS) - requested_schedules)
        if missing_baselines or missing_comparators:
            raise ValueError(
                "Reference locked-test reporting requires complete fixed and SER comparator sets; "
                f"missing fixed keys: {missing_baselines}; missing SER keys: {missing_comparators}."
            )
    includes_backbone_training = "backbone_training" in effective_stages
    provided_validation: Dict[str, Any] | None = None
    if bool(getattr(args, "use_provided_backbones", False)) or str(getattr(args, "backbone_package_root", "") or "").strip():
        if includes_backbone_training:
            raise ValueError("Provided-backbone mode cannot include backbone_training; run downstream stages only.")
        manifest_path = resolve_project_path(str(args.backbone_manifest))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Provided-backbone mode requires an existing backbone manifest: {manifest_path}")
        provided_validation = validate_provided_backbone_manifest(
            manifest_path,
            scenario_key=dataset,
            benchmark_family=family,
        )
        if provided_validation["status"] != "complete":
            raise ValueError("Invalid provided backbone manifest:\n- " + "\n- ".join(provided_validation["errors"]))
    if includes_backbone_training and family in {SCENARIO_FAMILY_FORECAST, SCENARIO_FAMILY_CONDITIONAL_GENERATION}:
        requested_manifest = resolve_project_path(str(args.backbone_manifest))
        required_manifest = backbone_manifest_path().resolve()
        if requested_manifest != required_manifest:
            raise ValueError(
                "Temporal full-pipeline backbone training materializes the reference backbone manifest at "
                f"{_display_path(required_manifest)}; do not override --backbone_manifest for runs that include "
                "backbone_training."
            )
    report = {
        "status": "complete",
        "scenario_key": dataset,
        "benchmark_family": family,
        "synthetic_length": int(args.synthetic_length),
    }
    if provided_validation is not None:
        report["provided_backbone_validation"] = provided_validation
    return report


def _protocol_payload(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = _resolved_scenario_key(args)
    plan = experiment_plan_by_key().get(dataset)
    locked_test_settings = _locked_test_settings(args)
    effective_stages = _selected_stage_names(args)
    includes_ablations = any(stage in {ABLATION_STUDENT_STAGE, ABLATION_LOCKED_TEST_STAGE} for stage in effective_stages)
    gipo_active = any(stage in {"train_gipo", "report_gipo_locked_test"} for stage in effective_stages)
    unseen_target_student_active = "train_unseen_target_student" in effective_stages
    ablation_preset = str(getattr(args, "ablation_preset", ABLATION_PRESET_ALL))
    policy = gipo_policy()
    student_source: Any = policy if gipo_active else args
    student_settings = _student_objective_settings_payload(student_source)
    flow_map_active = bool(set(effective_stages) & set(FLOW_MAP_PIPELINE_STAGES))
    return {
        "version": PIPELINE_VERSION,
        "scenario_key": dataset,
        "benchmark_family": "" if plan is None else str(plan.benchmark_family),
        "stages": effective_stages,
        "include_ablations": bool(includes_ablations),
        "include_flow_map": bool(flow_map_active),
        "gipo_policy_root": GIPO_POLICY_ROOT.as_posix() if gipo_active else "",
        "gipo_policy_key": GIPO_POLICY_KEY if gipo_active else "",
        "gipo_policy": policy.manifest_record() if gipo_active else {},
        "unseen_target_student_objective_settings": (
            _student_objective_settings_payload(args) if unseen_target_student_active else {}
        ),
        "ablation_preset": ablation_preset if includes_ablations else "",
        "ablation_student_policies": [policy.manifest_record() for policy in ablation_student_policies(ablation_preset)] if includes_ablations else [],
        "backbone_steps": int(args.backbone_steps),
        "checkpoint_steps": parse_int_csv(args.checkpoint_steps, default=REFERENCE_CHECKPOINT_STEPS),
        "seen_nfes": parse_int_csv(args.seen_nfes, default=REFERENCE_SEEN_NFES),
        "unseen_nfes": parse_int_csv(args.unseen_nfes, default=REFERENCE_UNSEEN_NFES),
        "schedules": parse_csv(args.schedule_keys) or list(REFERENCE_SUPERVISION_SCHEDULE_KEYS),
        "context_sample_count": int(args.context_sample_count),
        "schedule_row_split_phases": list(SCHEDULE_ROW_SPLIT_PHASES),
        "locked_test_mode": str(locked_test_settings["mode"]),
        "locked_test_context_limit": locked_test_settings["context_limit"],
        "locked_test_context_limit_scope": str(locked_test_settings["context_limit_scope"]),
        "ser_calibration_batch_size": int(args.ser_calibration_batch_size),
        "ser_val_windows": int(args.ser_val_windows),
        "ser_train_tuning_max_examples": int(args.ser_train_tuning_max_examples),
        "ser_train_tuning_effective_max_examples": _effective_ser_train_tuning_max_examples(args),
        "ser_example_selection_protocol": SER_PTG_EXAMPLE_SELECTION_PROTOCOL,
        "ser_local_defect_proxy_protocol": SER_PTG_LOCAL_DEFECT_PROXY_PROTOCOL,
        "schedule_context_selection_protocol": SCHEDULE_CONTEXT_SELECTION_PROTOCOL,
        "gipo_teacher_steps": int(args.gipo_teacher_steps),
        "gipo_student_steps": int(args.gipo_student_steps),
        "student_teacher_score_weight": float(student_settings["student_teacher_score_weight"]),
        "student_teacher_score_warmup_fraction": float(student_settings["student_teacher_score_warmup_fraction"]),
        "student_teacher_score_clip": float(DEFAULT_STUDENT_TEACHER_SCORE_CLIP),
        "student_teacher_score_protocol": "late_ramped_per_cell_teacher_utility_z_score",
        "student_teacher_score_include_unseen_targets": bool(
            student_settings["student_teacher_score_include_unseen_targets"]
        ),
        "student_target_mixture_mode": str(student_settings["student_target_mixture_mode"]),
        "student_target_elite_fraction": float(student_settings["student_target_elite_fraction"]),
        "student_target_elite_k": int(student_settings["student_target_elite_k"]),
        "student_target_elite_min_count": int(student_settings["student_target_elite_min_count"]),
        "student_target_elite_blend_all_weight": float(student_settings["student_target_elite_blend_all_weight"]),
        "teacher_metric_profile": teacher_metric_profile_for_scenario(dataset),
        "synthetic_length": int(args.synthetic_length),
        "dataset_root": _display_path(str(args.dataset_root)),
        "shared_backbone_root": _display_path(str(args.shared_backbone_root)),
        "backbone_manifest": _display_path(str(args.backbone_manifest)),
        "cryptos_path": _display_path(str(args.cryptos_path)),
        "lobster_synthetic_profile_path": _display_path(str(args.lobster_synthetic_profile_path)),
        "long_term_st_path": _display_path(str(args.long_term_st_path)),
        "molecule_group_root": _display_path(str(getattr(args, "molecule_group_root", "") or molecule_group_root())),
        "molecule_backbone_root": _display_path(str(getattr(args, "molecule_backbone_root", "") or (project_outputs_root() / "molecule_3d_backbones"))),
        "backbone_package": backbone_package_protocol_payload(args),
        "flow_map": (
            {
                "quality_status": (
                    "pending"
                    if str(args.flow_map_quality_rows_csv).strip()
                    else "not_evaluated"
                ),
                "performance_claim": False,
                "backbone_checkpoint": _display_optional_path(args.flow_map_backbone_checkpoint),
                "gipo_checkpoint": _display_optional_path(args.flow_map_gipo_checkpoint),
                "flow_map_checkpoint": _display_optional_path(
                    getattr(args, "flow_map_checkpoint", "")
                ),
                "contexts_npz": _display_optional_path(args.flow_map_contexts_npz),
                "demonstration_manifest": _display_optional_path(args.flow_map_demonstration_manifest),
                "settings": str(args.flow_map_settings),
                "rollouts_per_context": int(args.flow_map_rollouts_per_context),
                "training_steps": int(args.flow_map_steps),
                "training_batch_size": int(args.flow_map_batch_size),
                "quality_rows_csv": _display_optional_path(args.flow_map_quality_rows_csv),
                "bootstrap_samples": int(args.flow_map_bootstrap_samples),
                "familywise_alpha": float(args.flow_map_familywise_alpha),
            }
            if flow_map_active
            else {}
        ),
    }


def _effective_ser_train_tuning_max_examples(args: argparse.Namespace) -> int:
    explicit = int(getattr(args, "ser_train_tuning_max_examples", 0))
    if explicit > 0:
        return int(explicit)
    return int(getattr(args, "context_sample_count", TRAIN_TUNING_CONTEXT_SAMPLE_COUNT))


def _effective_ser_train_tuning_max_examples_source(args: argparse.Namespace) -> str:
    if int(getattr(args, "ser_train_tuning_max_examples", 0)) > 0:
        return "train_tuning_max_examples"
    return "context_sample_count"


def _has_ablation_stage(commands: Sequence[StageCommand]) -> bool:
    ablation_stages = {ABLATION_STUDENT_STAGE, ABLATION_LOCKED_TEST_STAGE}
    return any(entry.stage in ablation_stages for entry in commands)


def _ablation_root(run_root: Path, preset: str) -> Path:
    return run_root / "gipo_ablations" / str(preset)


def _run_relative_path(run_root: Path, path: Path) -> str:
    rel = path.relative_to(run_root).as_posix()
    return "." if rel == "." else rel


def _resolve_run_root(args: argparse.Namespace) -> Path:
    explicit = str(args.run_root).strip()
    if explicit:
        return resolve_project_path(explicit)
    return project_outputs_root() / "full_pipeline" / _resolved_scenario_key(args)


def _build_ablation_manifest(
    args: argparse.Namespace,
    run_root: Path,
    *,
    protocol_hash: str,
    status: str,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    dataset = _resolved_scenario_key(args)
    family = scenario_family_for_key(dataset)
    preset = str(getattr(args, "ablation_preset", ABLATION_PRESET_ALL))
    root = _ablation_root(run_root, preset)
    policies = []
    for policy in ablation_student_policies(preset):
        train_root = root / policy.policy_key / "gipo" / policy.student_training_mode
        report_root = root / policy.policy_key / "locked_test_reports" / policy.student_training_mode
        policies.append(
            {
                **policy.manifest_record(),
                "outputs": {
                    "training_summary": _run_relative_path(run_root, train_root / "gipo_training_summary.json"),
                    "student_checkpoint": _run_relative_path(run_root, train_root / "gipo_student.pt"),
                    "locked_test_report_root": _run_relative_path(run_root, report_root),
                },
            }
        )
    manifest = {
        "artifact": "gipo_ablation_manifest",
        "schema_version": "genode_gipo_ablation",
        "status": status,
        "ablation_set_id": preset,
        "scenario_key": dataset,
        "benchmark_family": family,
        "commit_sha": _git_head_commit(),
        "protocol_hash": protocol_hash,
        "ablation_root": _run_relative_path(run_root, root),
        "checkpoint_steps": parse_int_csv(args.checkpoint_steps, default=REFERENCE_CHECKPOINT_STEPS),
        "seen_nfes": parse_int_csv(args.seen_nfes, default=REFERENCE_SEEN_NFES),
        "unseen_nfes": parse_int_csv(args.unseen_nfes, default=REFERENCE_UNSEEN_NFES),
        "gipo_teacher_steps": int(args.gipo_teacher_steps),
        "gipo_student_steps": int(args.gipo_student_steps),
        "schedule_keys": parse_csv(args.schedule_keys) or list(REFERENCE_SUPERVISION_SCHEDULE_KEYS),
        "student_policies": policies,
        "student_policy_count": int(len(policies)),
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
        if bool(resume) or bool(overwrite):
            return
        raise ValueError(f"Run root {run_root} already has status.json; pass --resume or --overwrite explicitly.")
    if bool(overwrite):
        return
    if bool(resume):
        raise ValueError(
            f"Cannot resume {run_root} with different protocol_hash; existing={existing_hash}, new={protocol_hash}."
        )
    raise ValueError(f"Run root {run_root} already has status.json; pass --resume or --overwrite explicitly.")


def _python_module_command(module: str, args: Iterable[str]) -> List[str]:
    return [sys.executable, "-m", module, *[str(value) for value in args]]


def _command_hash(command: Sequence[str]) -> str:
    input_args_by_module = {
        "genode.distillation.demonstrations": (
            "--backbone-checkpoint",
            "--gipo-checkpoint",
            "--contexts-npz",
        ),
        "genode.distillation.training": (
            "--demonstration-manifest",
            "--backbone-checkpoint",
            "--gipo-checkpoint",
        ),
        "genode.distillation.evaluation": (
            "--rows-csv",
            "--flow-map-checkpoint",
            "--backbone-checkpoint",
            "--gipo-checkpoint",
        ),
    }
    input_hashes: dict[str, str] = {}
    for module, path_args in input_args_by_module.items():
        if _command_module_args(command, module) is None:
            continue
        for path_arg in path_args:
            value = _command_arg_value(command, path_arg)
            if not value:
                continue
            path = resolve_project_path(value)
            input_hashes[path_arg] = file_sha256(path) if path.is_file() else "missing"
    encoded = json.dumps(
        {
            "argv": [str(part) for part in command],
            "input_sha256": input_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _command_module_args(command: Sequence[str], module: str) -> List[str] | None:
    parts = [str(part) for part in command]
    if module not in parts:
        return None
    idx = parts.index(module)
    if idx < 1 or parts[idx - 1] != "-m":
        return None
    return parts[idx + 1 :]


def _command_arg_value(command: Sequence[str], name: str) -> str:
    parts = [str(part) for part in command]
    if str(name) not in parts:
        return ""
    idx = parts.index(str(name))
    if idx + 1 >= len(parts):
        raise ValueError(f"{name} is missing a value in command: {' '.join(parts)}")
    return parts[idx + 1]


def _schedule_row_command_status(command: Sequence[str]) -> Dict[str, Any] | None:
    runner_args = _command_module_args(command, "genode.evaluation.diffusion_flow_time_reparameterization")
    if runner_args is None:
        return None
    parsed = schedule_runner.build_argparser().parse_args(runner_args)
    return schedule_runner.schedule_row_output_status(resolve_project_path(str(parsed.out_root)), parsed)


def _command_json_output_exists(command: Sequence[str], *, module: str, out_arg: str, relative_path: str) -> Tuple[bool, str]:
    if _command_module_args(command, module) is None:
        return True, ""
    out_dir = _command_arg_value(command, out_arg)
    if not out_dir:
        return False, f"missing {out_arg} in {module} command"
    path = resolve_project_path(out_dir) / relative_path
    if not path.exists():
        return False, f"missing expected artifact: {_display_path(path)}"
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON artifact {path}: {exc}"
    return True, ""


def _command_path_exists(command: Sequence[str], *, module: str, out_arg: str, relative_path: str) -> Tuple[bool, str]:
    if _command_module_args(command, module) is None:
        return True, ""
    out_dir = _command_arg_value(command, out_arg)
    if not out_dir:
        return False, f"missing {out_arg} in {module} command"
    path = resolve_project_path(out_dir) / relative_path
    if not path.exists():
        return False, f"missing expected artifact: {path}"
    return True, ""


def _command_output_file_exists(
    command: Sequence[str],
    *,
    module: str,
    path_arg: str,
    require_json: bool = False,
) -> Tuple[bool, str]:
    if _command_module_args(command, module) is None:
        return True, ""
    value = _command_arg_value(command, path_arg)
    if not value:
        return False, f"missing {path_arg} in {module} command"
    path = resolve_project_path(value)
    if not path.is_file():
        return False, f"missing expected artifact: {_display_path(path)}"
    if require_json:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON artifact {path}: {exc}"
    return True, ""


def _backbone_manifest_artifacts_complete(
    manifest_path: Path,
    *,
    backbone_name: str,
    benchmark_family: str,
    scenario_key: str,
    checkpoint_steps: Sequence[int],
    stratum: str = "",
) -> Tuple[bool, str]:
    if not manifest_path.exists():
        return False, f"missing required backbone manifest: {_display_path(manifest_path)}"
    try:
        manifest = load_portable_backbone_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            False,
            "invalid required backbone manifest "
            f"{_display_path(manifest_path)} ({type(exc).__name__})",
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False, f"required backbone manifest has invalid artifacts: {_display_path(manifest_path)}"

    requested_steps = [int(step) for step in checkpoint_steps]
    if not requested_steps or len(set(requested_steps)) != len(requested_steps):
        return False, f"invalid requested backbone checkpoint steps: {requested_steps}"
    for checkpoint_step in requested_steps:
        matches: List[Mapping[str, Any]] = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            try:
                artifact_steps = int(artifact.get("train_steps", -1))
            except (TypeError, ValueError):
                continue
            if (
                str(artifact.get("status", "")) == "ready"
                and str(artifact.get("backbone_name", "")) == str(backbone_name)
                and str(artifact.get("benchmark_family", "")) == str(benchmark_family)
                and str(artifact.get("dataset_key", "")) == str(scenario_key)
                and int(artifact_steps) == int(checkpoint_step)
                and (not stratum or str(artifact.get("stratum", "")) == str(stratum))
            ):
                matches.append(artifact)
        if len(matches) != 1:
            member_label = f", stratum={stratum!r}" if stratum else ""
            return (
                False,
                "required backbone manifest has "
                f"{len(matches)} ready artifacts for {backbone_name}/{benchmark_family}/{scenario_key} "
                f"at train_steps={int(checkpoint_step)}{member_label}; expected 1",
            )
        artifact = matches[0]
        checkpoint_value = str(artifact.get("checkpoint_path", "") or "")
        if not checkpoint_value:
            return False, f"required backbone artifact is missing checkpoint_path: {artifact.get('checkpoint_id', '')}"
        checkpoint_path = Path(checkpoint_value)
        try:
            errors = validate_backbone_artifact_checkpoint(artifact, checkpoint_path)
        except OSError as exc:
            return (
                False,
                "unable to validate required backbone artifact "
                f"{_display_path(checkpoint_path)} ({type(exc).__name__})",
            )
        if errors:
            return (
                False,
                "required backbone artifact failed integrity validation: "
                f"{artifact.get('checkpoint_id', '')}",
            )
        for field in ("summary_path", "metadata_path"):
            value = str(artifact.get(field, "") or "")
            if value and not Path(value).is_file():
                return False, f"required backbone artifact is missing {field}: {_display_path(value)}"
    return True, ""


def _backbone_training_command_outputs_complete(command: Sequence[str]) -> Tuple[bool, str]:
    if _command_module_args(command, "genode.training.train_backbone") is not None:
        scenario_key = _command_arg_value(command, "--scenario_key")
        checkpoint_values = _command_arg_value(command, "--checkpoint_steps")
        if not scenario_key or not checkpoint_values:
            return False, "backbone training command is missing --scenario_key or --checkpoint_steps"
        try:
            benchmark_family = scenario_family_for_key(scenario_key)
            checkpoint_steps = parse_int_csv(checkpoint_values)
        except ValueError as exc:
            return False, f"invalid backbone training command: {exc}"
        if benchmark_family not in {SCENARIO_FAMILY_FORECAST, SCENARIO_FAMILY_CONDITIONAL_GENERATION}:
            return False, f"temporal backbone training command has unsupported family: {benchmark_family!r}"
        manifest_value = _command_arg_value(command, "--backbone_manifest")
        manifest_path = resolve_project_path(manifest_value) if manifest_value else backbone_manifest_path().resolve()
        return _backbone_manifest_artifacts_complete(
            manifest_path,
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=benchmark_family,
            scenario_key=scenario_key,
            checkpoint_steps=checkpoint_steps,
        )

    if _command_module_args(command, "genode.training.train_molecule_backbone") is None:
        return True, ""
    scenario_key = _command_arg_value(command, "--scenario_key")
    stratum = _command_arg_value(command, "--stratum")
    manifest_value = _command_arg_value(command, "--backbone_manifest")
    checkpoint_values = _command_arg_value(command, "--budget_steps")
    if not scenario_key or not stratum or not manifest_value or not checkpoint_values:
        return False, "molecule backbone training command is missing required manifest or artifact arguments"
    try:
        checkpoint_steps = parse_int_csv(checkpoint_values)
    except ValueError as exc:
        return False, f"invalid molecule backbone training command: {exc}"
    return _backbone_manifest_artifacts_complete(
        resolve_project_path(manifest_value),
        backbone_name=BACKBONE_NAME_OTFLOW_MOLECULE,
        benchmark_family=SCENARIO_FAMILY_MOLECULE,
        scenario_key=scenario_key,
        checkpoint_steps=checkpoint_steps,
        stratum=stratum,
    )


def _flow_map_command_outputs_complete(command: Sequence[str]) -> Tuple[bool, str]:
    demonstration_args = _command_module_args(command, "genode.distillation.demonstrations")
    if demonstration_args is not None:
        from genode.distillation.artifacts import load_demonstration_manifest

        output_dir = _command_arg_value(command, "--output-dir")
        try:
            load_demonstration_manifest(
                resolve_project_path(output_dir) / DEMONSTRATION_MANIFEST_NAME
            )
        except (OSError, TypeError, ValueError) as exc:
            return False, f"invalid flow-map demonstration artifact: {exc}"

    training_args = _command_module_args(command, "genode.distillation.training")
    if training_args is not None:
        from genode.distillation.artifacts import load_demonstration_manifest
        from genode.distillation.checkpoint import load_flow_map_checkpoint

        checkpoint_path = resolve_project_path(_command_arg_value(command, "--output-checkpoint"))
        backbone_path = resolve_project_path(_command_arg_value(command, "--backbone-checkpoint"))
        gipo_path = resolve_project_path(_command_arg_value(command, "--gipo-checkpoint"))
        manifest_path = resolve_project_path(_command_arg_value(command, "--demonstration-manifest"))
        summary_path = resolve_project_path(_command_arg_value(command, "--summary-json"))
        try:
            load_demonstration_manifest(manifest_path)
            _, payload = load_flow_map_checkpoint(
                checkpoint_path,
                backbone_checkpoint=backbone_path,
                gipo_checkpoint=gipo_path,
            )
            if str(payload.get("demonstration_manifest_sha256", "")) != file_sha256(
                manifest_path
            ):
                return False, "flow-map checkpoint demonstration hash does not match"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if str(summary.get("checkpoint_sha256", "")) != file_sha256(checkpoint_path):
                return False, "flow-map training summary checkpoint hash does not match"
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"invalid flow-map training artifact: {exc}"

    evaluation_args = _command_module_args(command, "genode.distillation.evaluation")
    if evaluation_args is not None:
        from genode.distillation.checkpoint import load_flow_map_checkpoint
        from genode.distillation.evaluation import (
            QualityGateConfig,
            evaluate_quality_gate,
            metric_specs_for_scenario,
            not_evaluated_report,
            read_quality_rows,
        )

        output_path = resolve_project_path(_command_arg_value(command, "--output-json"))
        flow_map_path = resolve_project_path(_command_arg_value(command, "--flow-map-checkpoint"))
        backbone_path = resolve_project_path(_command_arg_value(command, "--backbone-checkpoint"))
        gipo_path = resolve_project_path(_command_arg_value(command, "--gipo-checkpoint"))
        scenario_key = _command_arg_value(command, "--scenario-key")
        try:
            _, payload = load_flow_map_checkpoint(
                flow_map_path,
                backbone_checkpoint=backbone_path,
                gipo_checkpoint=gipo_path,
            )
            binding = {
                "scenario_key": scenario_key,
                "flow_map_checkpoint_sha256": file_sha256(flow_map_path),
                "backbone_checkpoint_sha256": str(payload["backbone_checkpoint_sha256"]),
                "gipo_checkpoint_sha256": str(payload["gipo_checkpoint_sha256"]),
            }
            rows_path_value = _command_arg_value(command, "--rows-csv")
            if rows_path_value:
                rows_path = resolve_project_path(rows_path_value)
                expected = evaluate_quality_gate(
                    read_quality_rows(rows_path),
                    metric_specs=metric_specs_for_scenario(scenario_key),
                    artifact_binding=binding,
                    config=QualityGateConfig(
                        bootstrap_samples=int(_command_arg_value(command, "--bootstrap-samples")),
                        familywise_alpha=float(_command_arg_value(command, "--familywise-alpha")),
                        margin=0.0,
                        seed=int(_command_arg_value(command, "--seed")),
                    ),
                )
                expected["rows_sha256"] = file_sha256(rows_path)
            else:
                expected = not_evaluated_report(
                    reason=_command_arg_value(command, "--not-evaluated-reason"),
                    artifact_binding=binding,
                )
            actual = json.loads(output_path.read_text(encoding="utf-8"))
            if actual != expected:
                return False, "flow-map quality report does not match its bound inputs"
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"invalid flow-map quality artifact: {exc}"
    return True, ""


def _stage_outputs_complete(entry: StageCommand) -> Tuple[bool, str]:
    if entry.stage == "backbone_training":
        for command in entry.commands:
            ok, reason = _backbone_training_command_outputs_complete(command)
            if not ok:
                return False, reason
    for command in entry.commands:
        ok, reason = _flow_map_command_outputs_complete(command)
        if not ok:
            return False, reason
        schedule_status = _schedule_row_command_status(command)
        if schedule_status is not None and not bool(schedule_status.get("complete", False)):
            return False, str(schedule_status.get("reason", "schedule-row output is incomplete"))
        ok, reason = _command_json_output_exists(
            command,
            module="genode.gipo.ser_ptg_reference",
            out_arg="--out_dir",
            relative_path="ser_ptg_schedule_summary.json",
        )
        if not ok:
            return False, reason
        ok, reason = _command_json_output_exists(
            command,
            module="genode.gipo.train_gipo",
            out_arg="--out_dir",
            relative_path="gipo_training_summary.json",
        )
        if not ok:
            return False, reason
        ok, reason = _command_path_exists(
            command,
            module="genode.gipo.train_gipo",
            out_arg="--out_dir",
            relative_path="gipo_student.pt",
        )
        if not ok:
            return False, reason
        ok, reason = _command_json_output_exists(
            command,
            module="genode.gipo.report_locked_test",
            out_arg="--out_dir",
            relative_path="locked_test_gipo_policy_summary.json",
        )
        if not ok:
            return False, reason
        ok, reason = _command_json_output_exists(
            command,
            module="genode.gipo.report_locked_test",
            out_arg="--out_dir",
            relative_path="locked_test_gipo_comparison_summary.json",
        )
        if not ok:
            return False, reason
        for relative_path in (
            "locked_test_gipo_rows.csv",
            "locked_test_gipo_aggregate_rows.csv",
            "locked_test_gipo_decisions.csv",
        ):
            ok, reason = _command_path_exists(
                command,
                module="genode.gipo.report_locked_test",
                out_arg="--out_dir",
                relative_path=relative_path,
            )
            if not ok:
                return False, reason
        ok, reason = _command_json_output_exists(
            command,
            module="genode.distillation.demonstrations",
            out_arg="--output-dir",
            relative_path=DEMONSTRATION_MANIFEST_NAME,
        )
        if not ok:
            return False, reason
        for path_arg, require_json in (
            ("--output-checkpoint", False),
            ("--summary-json", True),
        ):
            ok, reason = _command_output_file_exists(
                command,
                module="genode.distillation.training",
                path_arg=path_arg,
                require_json=require_json,
            )
            if not ok:
                return False, reason
        ok, reason = _command_output_file_exists(
            command,
            module="genode.distillation.evaluation",
            path_arg="--output-json",
            require_json=True,
        )
        if not ok:
            return False, reason
    return True, ""


def _stage_manifest_complete(run_root: Path, entry: StageCommand, *, protocol_hash: str) -> Tuple[bool, str]:
    path = run_root / entry.manifest_name
    if not path.exists():
        return False, f"missing stage manifest: {path}"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid stage manifest {path}: {exc}"
    if str(manifest.get("status", "")) != "complete":
        return False, f"stage manifest status is {manifest.get('status')!r}"
    if str(manifest.get("stage", "")) != str(entry.stage):
        return False, f"stage manifest is for {manifest.get('stage')!r}, expected {entry.stage!r}"
    if str(manifest.get("protocol_hash", "")) != str(protocol_hash):
        return False, "stage manifest protocol hash does not match"
    expected_hashes = [_command_hash(command) for command in entry.commands]
    manifest_hashes = manifest.get("command_hashes")
    if not isinstance(manifest_hashes, list):
        return False, "stage manifest is missing command hashes"
    if list(manifest_hashes) != expected_hashes:
        return False, "stage command hashes do not match"
    results = list(manifest.get("command_results", []))
    if len(results) != len(entry.commands):
        return False, "stage command_results are incomplete"
    for idx, result in enumerate(results):
        if int(result.get("command_index", -1)) != int(idx):
            return False, f"stage command_result index mismatch at {idx}"
        if int(result.get("returncode", -1)) != 0:
            return False, f"stage command {idx} did not complete successfully"
    outputs_complete, reason = _stage_outputs_complete(entry)
    if not outputs_complete:
        return False, reason
    return True, ""


def _resume_completed_prefix(run_root: Path, commands: Sequence[StageCommand], *, protocol_hash: str) -> List[StageCommand]:
    completed: List[StageCommand] = []
    for entry in commands:
        is_complete, _reason = _stage_manifest_complete(run_root, entry, protocol_hash=protocol_hash)
        if not is_complete:
            break
        completed.append(entry)
    return completed


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
                    "--scenario_key",
                    dataset,
                    "--steps",
                    int(args.backbone_steps),
                    "--checkpoint_steps",
                    checkpoints,
                    "--synthetic_length",
                    int(args.synthetic_length),
                    *_temporal_training_data_path_args(args),
                    "--device",
                    str(args.device),
                ],
            )
        ]
    group_root = resolve_project_path(str(getattr(args, "molecule_group_root", "") or molecule_group_root()))
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
                    "--scenario_key",
                    dataset,
                    "--stratum",
                    str(member["stratum"]),
                    "--processed_dir",
                    _molecule_processed_dir_arg(group_root, str(dataset), member),
                    "--out_dir",
                    str(args.molecule_backbone_root),
                    "--backbone_manifest",
                    str(args.backbone_manifest),
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
    dataset = _resolved_scenario_key(args)
    checkpoint_values = parse_int_csv(args.checkpoint_steps, default=REFERENCE_CHECKPOINT_STEPS)
    checkpoints = ",".join(str(value) for value in checkpoint_values)
    seen_nfes = ",".join(
        str(value) for value in parse_int_csv(args.seen_nfes, default=REFERENCE_SEEN_NFES)
    )
    unseen_nfes = ",".join(
        str(value) for value in parse_int_csv(args.unseen_nfes, default=REFERENCE_UNSEEN_NFES)
    )
    schedules = parse_csv(args.schedule_keys) or list(REFERENCE_SUPERVISION_SCHEDULE_KEYS)
    fixed_schedules = ",".join(_fixed_schedule_keys_for_runner(schedules))
    ser_schedules = ",".join(_ser_schedule_keys_for_runner(schedules))
    support_schedules = _schedule_keys_for_gipo(schedules)
    rows_root = run_root / "schedule_rows"
    ser_root = run_root / "ser_summaries"
    gipo_root = run_root / "gipo"
    ablation_preset = str(getattr(args, "ablation_preset", ABLATION_PRESET_ALL))
    gipo_ablation_root = _ablation_root(run_root, ablation_preset)
    ablation_policies = ablation_student_policies(ablation_preset)
    policy = gipo_policy()
    gipo_policy_root = run_root / GIPO_POLICY_ROOT
    gipo_training_root = gipo_policy_root / "training"
    gipo_report_root = gipo_policy_root / "locked_test_reports"
    flow_map_root = run_root / "flow_map"
    default_demonstration_root = flow_map_root / "demonstrations"
    explicit_demonstration_manifest = str(
        getattr(args, "flow_map_demonstration_manifest", "") or ""
    ).strip()
    demonstration_manifest = (
        resolve_project_path(explicit_demonstration_manifest)
        if explicit_demonstration_manifest
        else default_demonstration_root / DEMONSTRATION_MANIFEST_NAME
    )
    demonstration_root = demonstration_manifest.parent
    explicit_flow_map_checkpoint = str(
        getattr(args, "flow_map_checkpoint", "") or ""
    ).strip()
    flow_map_checkpoint = (
        resolve_project_path(explicit_flow_map_checkpoint)
        if explicit_flow_map_checkpoint
        else flow_map_root / "endpoint_flow_map.pt"
    )
    flow_map_training_summary = flow_map_root / "training_summary.json"
    flow_map_quality_report = flow_map_root / "quality_report.json"
    flow_map_gipo_checkpoint = str(getattr(args, "flow_map_gipo_checkpoint", "") or "").strip()
    if not flow_map_gipo_checkpoint:
        flow_map_gipo_checkpoint = str(gipo_training_root / "gipo_student.pt")
    split_phases = SCHEDULE_ROW_SPLIT_PHASES
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

    def _schedule_row_phase_budget_args(phase: str) -> List[Any]:
        if str(phase) == "train_tuning":
            return ["--context_sample_count", int(args.context_sample_count)]
        if str(phase) == "locked_test":
            settings = _locked_test_settings(args)
            if settings["mode"] == "full":
                return []
            values: List[Any] = ["--locked_test_preview"]
            if getattr(args, "locked_test_preview_contexts", None) is not None:
                values.extend(["--locked_test_preview_contexts", int(settings["context_limit"])])
            return values
        raise ValueError(f"Unsupported split phase {phase!r}.")

    def _gipo_train_command(mode: str, out_dir: Path, policy: GIPOStudentPolicy | None = None) -> List[str]:
        command_args: List[Any] = [
            "--rows_csv",
            _csv_list("seen", "train_tuning", "context_rows.csv"),
            "--context_embeddings_npz",
            _csv_list("seen", "train_tuning", "context_embeddings.npz"),
            "--schedule_summary_json",
            _ser_summary_list("seen"),
            "--support_schedule_keys",
            support_schedules,
            "--teacher_steps",
            int(args.gipo_teacher_steps),
            "--student_steps",
            int(args.gipo_student_steps),
            "--seen_target_nfe_values",
            seen_nfes,
            "--unseen_target_nfe_values",
            unseen_nfes,
            "--student_policy_key",
            "custom_gipo" if policy is None else policy.policy_key,
            *_teacher_target_args_for_scenario(dataset),
            *_student_objective_args(args, policy),
        ]
        if mode == STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET:
            unseen_target_weight = (
                REFERENCE_UNSEEN_TARGET_WEIGHT
                if policy is None
                else float(policy.student_unseen_target_weight)
            )
            command_args.extend(
                [
                    "--student_unseen_target_rows_csv",
                    _csv_list("unseen", "train_tuning", "context_rows.csv"),
                    "--student_unseen_target_context_embeddings_npz",
                    _csv_list("unseen", "train_tuning", "context_embeddings.npz"),
                    "--student_unseen_target_schedule_summary_json",
                    _ser_summary_list("unseen"),
                    "--student_unseen_target_weight",
                    f"{float(unseen_target_weight):g}",
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
                            *_schedule_row_phase_budget_args(phase),
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
                            "--scenario_key",
                            dataset,
                            "--split_phase",
                            "locked_test",
                            "--solver_names",
                            ",".join(str(key) for key in SUPPORTED_SOLVER_KEYS),
                            "--target_nfe_values",
                            nfe_values,
                            "--checkpoint_step",
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
        return _locked_test_report_commands_for_student(
            student_root=gipo_training_root,
            report_root=gipo_report_root,
        )

    def _ablation_student_commands() -> List[List[str]]:
        return [
            _gipo_train_command(
                policy.student_training_mode,
                gipo_ablation_root / policy.policy_key / "gipo" / policy.student_training_mode,
                policy,
            )
            for policy in ablation_policies
        ]

    def _ablation_locked_test_commands() -> List[List[str]]:
        commands: List[List[str]] = []
        for policy in ablation_policies:
            commands.extend(
                _locked_test_report_commands_for_student(
                    student_root=gipo_ablation_root / policy.policy_key / "gipo" / policy.student_training_mode,
                    report_root=gipo_ablation_root / policy.policy_key / "locked_test_reports" / policy.student_training_mode,
                )
            )
        return commands

    def _flow_map_collection_command() -> List[str]:
        command_args: List[Any] = [
            "--backbone-checkpoint",
            str(args.flow_map_backbone_checkpoint),
            "--gipo-checkpoint",
            flow_map_gipo_checkpoint,
            "--contexts-npz",
            str(args.flow_map_contexts_npz),
            "--output-dir",
            demonstration_root,
            "--split-phase",
            "train_tuning",
            "--scenario-key",
            dataset,
            "--benchmark-family",
            scenario_family_for_key(dataset),
            "--rollouts-per-context",
            int(args.flow_map_rollouts_per_context),
            "--seed",
            int(args.flow_map_seed),
            "--device",
            str(args.device),
        ]
        if str(args.flow_map_settings).strip():
            command_args.extend(["--settings", str(args.flow_map_settings)])
        if bool(args.overwrite):
            command_args.append("--overwrite")
        return _python_module_command("genode.distillation.demonstrations", command_args)

    def _flow_map_training_command() -> List[str]:
        command_args: List[Any] = [
            "--demonstration-manifest",
            demonstration_manifest,
            "--backbone-checkpoint",
            str(args.flow_map_backbone_checkpoint),
            "--gipo-checkpoint",
            flow_map_gipo_checkpoint,
            "--output-checkpoint",
            flow_map_checkpoint,
            "--summary-json",
            flow_map_training_summary,
            "--steps",
            int(args.flow_map_steps),
            "--batch-size",
            int(args.flow_map_batch_size),
            "--seed",
            int(args.flow_map_seed),
            "--device",
            str(args.device),
        ]
        if bool(args.overwrite):
            command_args.append("--overwrite")
        return _python_module_command(
            "genode.distillation.training",
            command_args,
        )

    def _flow_map_evaluation_command() -> List[str]:
        command_args: List[Any] = [
            "--output-json",
            flow_map_quality_report,
            "--scenario-key",
            dataset,
            "--flow-map-checkpoint",
            flow_map_checkpoint,
            "--backbone-checkpoint",
            str(args.flow_map_backbone_checkpoint),
            "--gipo-checkpoint",
            flow_map_gipo_checkpoint,
            "--bootstrap-samples",
            int(args.flow_map_bootstrap_samples),
            "--familywise-alpha",
            float(args.flow_map_familywise_alpha),
            "--seed",
            int(args.flow_map_seed),
        ]
        quality_rows = str(args.flow_map_quality_rows_csv).strip()
        if quality_rows:
            command_args.extend(["--rows-csv", quality_rows])
        else:
            command_args.extend(
                [
                    "--not-evaluated-reason",
                    "No benchmark rows were supplied; this artifact has no performance claim.",
                ]
            )
        return _python_module_command("genode.distillation.evaluation", command_args)

    ser_commands = [
        _python_module_command(
            "genode.gipo.ser_ptg_reference",
            [
                "--scenario_key",
                dataset,
                "--solver_names",
                ",".join(str(key) for key in SUPPORTED_SOLVER_KEYS),
                "--target_nfe_values",
                role_nfes[role],
                "--reference_split",
                "train_tuning",
                "--checkpoint_step",
                int(checkpoint_step),
                "--context_sample_count",
                int(args.context_sample_count),
                "--calibration_batch_size",
                int(args.ser_calibration_batch_size),
                "--val_windows",
                int(args.ser_val_windows),
                "--train_tuning_max_examples",
                _effective_ser_train_tuning_max_examples(args),
                "--train_tuning_max_examples_source",
                _effective_ser_train_tuning_max_examples_source(args),
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
            INPUT_PREFLIGHT_STAGE,
            [],
            "input_preflight_manifest.json",
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
            "train_gipo",
            [_gipo_train_command(policy.student_training_mode, gipo_training_root, policy)],
            "gipo_training_manifest.json",
        ),
        StageCommand(
            "report_gipo_locked_test",
            _locked_test_report_commands(),
            "gipo_locked_test_manifest.json",
        ),
        StageCommand(
            "train_unseen_target_student",
            [
                _gipo_train_command(
                    STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
                    gipo_root / STUDENT_TRAINING_MODE_SEEN_PLUS_UNSEEN_TARGET,
                )
            ],
            "unseen_target_student_manifest.json",
        ),
        StageCommand(
            ABLATION_STUDENT_STAGE,
            _ablation_student_commands(),
            "ablation_students_manifest.json",
        ),
        StageCommand(
            ABLATION_LOCKED_TEST_STAGE,
            _ablation_locked_test_commands(),
            "ablation_locked_test_manifest.json",
        ),
        StageCommand(
            FLOW_MAP_COLLECTION_STAGE,
            [_flow_map_collection_command()],
            "flow_map_demonstrations_manifest.json",
        ),
        StageCommand(
            FLOW_MAP_TRAINING_STAGE,
            [_flow_map_training_command()],
            "flow_map_training_manifest.json",
        ),
        StageCommand(
            FLOW_MAP_EVALUATION_STAGE,
            [_flow_map_evaluation_command()],
            "flow_map_evaluation_manifest.json",
        ),
    ]
    stage_filter = set(_selected_stage_names(args))
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
    skipped_entries = _resume_completed_prefix(run_root, commands, protocol_hash=protocol_hash) if bool(args.resume) and not bool(args.overwrite) else []
    skipped_stage_names = [entry.stage for entry in skipped_entries]
    commands_to_run = list(commands[len(skipped_entries) :])
    should_write_ablation_manifest = has_ablation_stage and (
        not (bool(args.resume) and not bool(args.overwrite))
        or _has_ablation_stage(commands_to_run)
        or bool(args.dry_run)
    )
    if should_write_ablation_manifest:
        ablation_manifest = _build_ablation_manifest(
            args,
            run_root,
            protocol_hash=protocol_hash,
            status="dry_run" if bool(args.dry_run) else "running",
        )
        _write_json(_ablation_root(run_root, str(args.ablation_preset)) / "ablation_manifest.json", ablation_manifest)
    _write_json(run_root / "protocol.json", {**protocol, "protocol_hash": protocol_hash})
    _write_json(
        _status_path(run_root),
        {
            "version": PIPELINE_VERSION,
            "status": "running" if not bool(args.dry_run) else "dry_run",
            "protocol_hash": protocol_hash,
            "run_root": _display_path(run_root),
            "stages": [entry.stage for entry in commands],
            "completed_stages": list(skipped_stage_names),
            "skipped_stages": list(skipped_stage_names),
            "dry_run": bool(args.dry_run),
            "preflight": preflight,
        },
    )
    completed: List[str] = list(skipped_stage_names)
    for entry in commands_to_run:
        manifest = {
            "stage": entry.stage,
            "protocol_hash": protocol_hash,
            "commands": [_display_command(command, path_base=run_root) for command in entry.commands],
            "command_hashes": [_command_hash(command) for command in entry.commands],
            "dry_run": bool(args.dry_run),
            "command_results": [],
        }
        if entry.stage == INPUT_PREFLIGHT_STAGE:
            manifest["preflight"] = preflight
        if not bool(args.dry_run):
            for command_idx, command in enumerate(entry.commands):
                if command and command[0] == "internal":
                    raise RuntimeError(
                        "Unresolved internal pipeline command in non-dry-run: "
                        f"{_display_command(command, path_base=run_root)}"
                    )
                log_path = run_root / "logs" / f"{entry.stage}_{command_idx}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                schedule_status = _schedule_row_command_status(command)
                if schedule_status is not None and bool(schedule_status.get("complete", False)):
                    command_result = {
                        "command_index": int(command_idx),
                        "returncode": 0,
                        "log_path": str(log_path.relative_to(run_root)),
                        "skipped": True,
                        "skip_reason": "schedule-row output already complete",
                    }
                    manifest["command_results"].append(command_result)
                    continue
                if schedule_status is not None and bool(schedule_status.get("protocol_mismatch", False)) and bool(args.resume):
                    manifest["command_results"].append(
                        {
                            "command_index": int(command_idx),
                            "returncode": 2,
                            "log_path": str(log_path.relative_to(run_root)),
                            "error": str(schedule_status.get("reason", "protocol mismatch")),
                        }
                    )
                    manifest["status"] = "failed"
                    _write_json(run_root / entry.manifest_name, manifest)
                    _write_json(
                        _status_path(run_root),
                        {
                            "version": PIPELINE_VERSION,
                            "status": "failed",
                            "failed_stage": entry.stage,
                            "failed_command_index": int(command_idx),
                            "protocol_hash": protocol_hash,
                            "completed_stages": completed,
                            "skipped_stages": list(skipped_stage_names),
                            "failure_reason": str(schedule_status.get("reason", "protocol mismatch")),
                        },
                    )
                    raise RuntimeError(
                        "Cannot resume schedule-row command with incompatible row-level protocol; "
                        f"{schedule_status.get('reason', 'protocol mismatch')}"
                    )
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
                            _ablation_root(run_root, str(args.ablation_preset)) / "ablation_manifest.json",
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
                            "skipped_stages": list(skipped_stage_names),
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
                "skipped_stages": list(skipped_stage_names),
                "remaining_stages": [cmd.stage for cmd in commands if cmd.stage not in set(completed)],
            },
        )
    summary = {
        "version": PIPELINE_VERSION,
        "status": "dry_run" if bool(args.dry_run) else "complete",
        "protocol_hash": protocol_hash,
        "stage_count": int(len(commands)),
        "stages": [_display_stage(entry, path_base=run_root) for entry in commands],
        "completed_stages": [entry.stage for entry in commands],
        "skipped_stages": list(skipped_stage_names),
        "executed_stages": [entry.stage for entry in commands_to_run],
        "run_root": _display_path(run_root),
        "preflight": preflight,
    }
    evaluation_entry = next(
        (entry for entry in commands if entry.stage == FLOW_MAP_EVALUATION_STAGE),
        None,
    )
    if evaluation_entry is not None and not bool(args.dry_run):
        evaluation_command = evaluation_entry.commands[0]
        quality_report_path = resolve_project_path(
            _command_arg_value(evaluation_command, "--output-json")
        )
        quality_report = json.loads(quality_report_path.read_text(encoding="utf-8"))
        summary["flow_map_quality"] = {
            "status": str(quality_report.get("status", "")),
            "performance_claim": bool(quality_report.get("performance_claim", False)),
            "report_path": _display_path(quality_report_path, path_base=run_root),
            "report_sha256": file_sha256(quality_report_path),
        }
    _write_json(run_root / "pipeline_summary.json", summary)
    _write_json(_status_path(run_root), summary)
    if should_write_ablation_manifest:
        final_ablation_manifest = _build_ablation_manifest(
            args,
            run_root,
            protocol_hash=protocol_hash,
            status="dry_run" if bool(args.dry_run) else "complete",
        )
        _write_json(_ablation_root(run_root, str(args.ablation_preset)) / "ablation_manifest.json", final_ablation_manifest)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-family GIPO reference workflow. It trains and reports the "
            f"{GIPO_POLICY_KEY} policy; --include_ablations adds the opt-in ablation grid."
        )
    )
    parser.add_argument("--scenario_key", required=True)
    parser.add_argument("--run_root", default="")
    parser.add_argument(
        "--stages",
        default="",
        help="Comma-separated explicit stage list; omitted runs the reference GIPO workflow.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone_steps", type=int, default=20_000)
    parser.add_argument(
        "--checkpoint_steps",
        default=",".join(str(value) for value in REFERENCE_CHECKPOINT_STEPS),
    )
    parser.add_argument("--seen_nfes", default=",".join(str(value) for value in REFERENCE_SEEN_NFES))
    parser.add_argument("--unseen_nfes", default=",".join(str(value) for value in REFERENCE_UNSEEN_NFES))
    parser.add_argument("--schedule_keys", default=",".join(REFERENCE_SUPERVISION_SCHEDULE_KEYS))
    parser.add_argument(
        "--context_sample_count",
        type=int,
        default=TRAIN_TUNING_CONTEXT_SAMPLE_COUNT,
        help="Train-tuning context budget for GIPO teacher/student supervision. Locked-test evaluation is controlled separately.",
    )
    parser.add_argument(
        "--locked_test_preview",
        action="store_true",
        help=(
            "Evaluate a deterministic per-seed preview instead of the full locked test. "
            f"Without an explicit count the preview uses {LOCKED_TEST_PREVIEW_CONTEXTS} contexts."
        ),
    )
    parser.add_argument(
        "--locked_test_preview_contexts",
        type=int,
        default=None,
        help=(
            "Per-seed context limit for --locked_test_preview. This option is invalid unless preview mode is enabled."
        ),
    )
    parser.add_argument("--ser_calibration_batch_size", type=int, default=64)
    parser.add_argument("--ser_val_windows", type=int, default=0)
    parser.add_argument("--ser_train_tuning_max_examples", type=int, default=0)
    parser.add_argument("--gipo_teacher_steps", type=int, default=DEFAULT_GIPO_TEACHER_STEPS)
    parser.add_argument("--gipo_student_steps", type=int, default=DEFAULT_GIPO_STUDENT_STEPS)
    unseen_target_help = (
        "Override only the explicit train_unseen_target_student stage; "
        "reference and ablation recipes are fixed."
    )
    parser.add_argument("--student_teacher_score_weight", type=float, default=None, help=unseen_target_help)
    parser.add_argument(
        "--student_teacher_score_warmup_fraction",
        type=float,
        default=None,
        help=unseen_target_help,
    )
    parser.add_argument(
        "--student_teacher_score_include_unseen_targets",
        action="store_true",
        default=None,
        help=unseen_target_help,
    )
    parser.add_argument(
        "--student_target_mixture_mode",
        choices=STUDENT_TARGET_MIXTURE_MODES,
        default=None,
        help=unseen_target_help,
    )
    parser.add_argument("--student_target_elite_fraction", type=float, default=None, help=unseen_target_help)
    parser.add_argument("--student_target_elite_k", type=int, default=None, help=unseen_target_help)
    parser.add_argument("--student_target_elite_min_count", type=int, default=None, help=unseen_target_help)
    parser.add_argument(
        "--student_target_elite_blend_all_weight",
        type=float,
        default=None,
        help=unseen_target_help,
    )
    parser.add_argument("--synthetic_length", type=int, default=2_000_000)
    parser.add_argument("--dataset_root", default=str(project_dataset_root()))
    parser.add_argument("--shared_backbone_root", default=str(project_outputs_root() / "otflow_backbones"))
    parser.add_argument("--backbone_manifest", default=str(backbone_manifest_path()))
    parser.add_argument("--cryptos_path", default=str(cryptos_data_path()))
    parser.add_argument("--lobster_synthetic_profile_path", default=str(lobster_synthetic_profile_path()))
    parser.add_argument("--long_term_st_path", default=str(long_term_st_data_path()))
    parser.add_argument("--molecule_group_root", default=str(molecule_group_root()))
    parser.add_argument("--molecule_backbone_root", default=str(project_outputs_root() / "molecule_3d_backbones"))
    parser.add_argument("--backbone_package_root", default="", help="Portable backbone package root to use for downstream-only GIPO stages.")
    parser.add_argument("--use_provided_backbones", action="store_true", default=False, help="Require existing packaged/provided backbones and refuse backbone_training stages.")
    parser.add_argument(
        "--include_ablations",
        action="store_true",
        default=False,
        help="Add ablation-student training and locked-test reports after the reference GIPO workflow.",
    )
    parser.add_argument("--ablation_preset", choices=ablation_preset_keys(), default=ABLATION_PRESET_ALL)
    parser.add_argument(
        "--include_flow_map",
        action="store_true",
        default=False,
        help=(
            "Add opt-in GIPO-guided demonstration collection, endpoint flow-map training, "
            "and quality reporting. This does not imply a performance claim."
        ),
    )
    parser.add_argument(
        "--flow_map_backbone_checkpoint",
        default="",
        help="Frozen OTFlow checkpoint used by every flow-map stage.",
    )
    parser.add_argument(
        "--flow_map_gipo_checkpoint",
        default="",
        help="Frozen GIPO checkpoint; omitted uses the checkpoint produced by train_gipo.",
    )
    parser.add_argument(
        "--flow_map_checkpoint",
        default="",
        help="Existing endpoint flow-map checkpoint for an evaluation-only stage.",
    )
    parser.add_argument(
        "--flow_map_contexts_npz",
        default="",
        help="Training/tuning contexts NPZ containing context_ids, histories, and optional conditions.",
    )
    parser.add_argument(
        "--flow_map_demonstration_manifest",
        default="",
        help=(
            "Existing demonstration manifest for an explicit train_flow_map stage, or an alternate "
            f"collection destination ending in {DEMONSTRATION_MANIFEST_NAME}."
        ),
    )
    parser.add_argument(
        "--flow_map_settings",
        default="",
        help="Comma-separated solver_key:target_nfe settings; omitted uses every supported setting.",
    )
    parser.add_argument("--flow_map_rollouts_per_context", type=int, default=4)
    parser.add_argument("--flow_map_steps", type=int, default=50_000)
    parser.add_argument("--flow_map_batch_size", type=int, default=256)
    parser.add_argument("--flow_map_seed", type=int, default=0)
    parser.add_argument(
        "--flow_map_quality_rows_csv",
        default="",
        help=(
            "Optional paired benchmark rows for the quality gate. Omitted writes status=not_evaluated "
            "and makes no performance claim."
        ),
    )
    parser.add_argument("--flow_map_bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--flow_map_familywise_alpha", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def main() -> None:
    summary = run_full_pipeline(build_argparser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
