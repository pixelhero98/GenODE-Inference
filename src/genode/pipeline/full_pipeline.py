from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
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
from genode.data.molecule_xyz import (
    load_molecule_group_manifest,
    molecule_group_manifest_path,
    molecule_group_root,
    trainable_molecule_group_members,
)
from genode.data.otflow_medical_constants import long_term_st_manifest_path
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
    context_embedding_table_manifest_path,
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
QUALITY_GATE_REJECTION_RESULT = "failed_without_performance_claim"
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
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _display_path(path: str | Path, *, path_base: Path | None = None) -> str:
    resolved = Path(path).expanduser().resolve()
    if path_base is not None:
        try:
            return resolved.relative_to(Path(path_base).expanduser().resolve()).as_posix()
        except ValueError:
            pass
    return display_project_path(path)


def _display_optional_project_path(path: Any) -> str:
    text = str(path or "").strip()
    return _display_path(resolve_project_path(text)) if text else ""


def _resolve_optional_project_path(path: Any) -> str:
    text = str(path or "").strip()
    return str(resolve_project_path(text)) if text else ""


def _optional_project_file_sha256(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    resolved = resolve_project_path(text)
    return file_sha256(resolved) if resolved.is_file() else ""


def _display_command(command: Sequence[str], *, path_base: Path | None = None) -> List[str]:
    out: List[str] = []
    for token in command:
        text = str(token)
        if "," in text:
            components = text.split(",")
            rendered = [
                _display_path(component, path_base=path_base)
                if Path(component).expanduser().is_absolute()
                else component
                for component in components
            ]
            out.append(",".join(rendered))
            continue
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
    quality_argument_names = (
        "flow_map_quality_rows_csv",
        "flow_map_quality_candidate_catalog",
        "flow_map_quality_contexts_npz",
        "flow_map_quality_sample_panel_npz",
        "flow_map_quality_measurement_protocol",
    )
    quality_arguments = {
        name: str(getattr(args, name, "") or "").strip()
        for name in quality_argument_names
    }
    supplied_quality_arguments = [
        name for name, value in quality_arguments.items() if value
    ]
    quality_rows = quality_arguments["flow_map_quality_rows_csv"]
    if supplied_quality_arguments:
        if FLOW_MAP_EVALUATION_STAGE not in flow_map_stages:
            raise ValueError(
                "Flow-map quality artifacts require the evaluate_flow_map stage; "
                "unused quality artifact arguments are not accepted."
            )
        conflicting = sorted(
            flow_map_stages
            & {FLOW_MAP_COLLECTION_STAGE, FLOW_MAP_TRAINING_STAGE}
        )
        if conflicting:
            raise ValueError(
                "Paired flow-map quality rows require a separate evaluation run "
                "without collection or training stages; conflicting stages: "
                + ", ".join(conflicting)
            )
        if not quality_rows:
            raise ValueError(
                "--flow_map_quality_rows_csv is required when any flow-map quality "
                "artifact argument is supplied."
            )
        required_flow_map_paths.append("flow_map_quality_rows_csv")
        candidate_catalog = quality_arguments["flow_map_quality_candidate_catalog"]
        if not candidate_catalog:
            raise ValueError(
                "--flow_map_quality_candidate_catalog is required when paired "
                "quality rows are supplied."
            )
        required_flow_map_paths.append("flow_map_quality_candidate_catalog")
        for name in (
            "flow_map_quality_contexts_npz",
            "flow_map_quality_sample_panel_npz",
            "flow_map_quality_measurement_protocol",
        ):
            if not quality_arguments[name]:
                raise ValueError(
                    f"--{name} is required when paired quality rows are supplied."
                )
            required_flow_map_paths.append(name)
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
        if int(args.flow_map_seed) < 0:
            raise ValueError("--flow_map_seed must be nonnegative.")
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


def _flow_map_quality_protocol_fields(args: argparse.Namespace) -> Dict[str, str]:
    rows_value = str(getattr(args, "flow_map_quality_rows_csv", "") or "").strip()
    catalog_value = str(
        getattr(args, "flow_map_quality_candidate_catalog", "") or ""
    ).strip()
    contexts_value = str(
        getattr(args, "flow_map_quality_contexts_npz", "") or ""
    ).strip()
    sample_panel_value = str(
        getattr(args, "flow_map_quality_sample_panel_npz", "") or ""
    ).strip()
    measurement_protocol_value = str(
        getattr(args, "flow_map_quality_measurement_protocol", "") or ""
    ).strip()
    if not rows_value:
        return {
            "quality_candidate_catalog_sha256": "",
            "quality_rows_sha256": "",
            "quality_contexts_sha256": "",
            "quality_sample_panel_sha256": "",
            "quality_measurement_protocol_sha256": "",
        }
    if (
        not catalog_value
        or not contexts_value
        or not sample_panel_value
        or not measurement_protocol_value
    ):
        raise ValueError(
            "The candidate catalog, quality contexts, quality sample panel, and "
            "measurement protocol are required when paired quality rows are supplied."
        )

    from genode.distillation.evaluation import (
        candidate_catalog_sha256,
        metric_specs_for_scenario,
        read_candidate_catalog,
    )
    from genode.distillation.measurement_protocol import (
        read_quality_measurement_protocol,
    )

    rows_path = resolve_project_path(rows_value)
    catalog_path = resolve_project_path(catalog_value)
    contexts_path = resolve_project_path(contexts_value)
    sample_panel_path = resolve_project_path(sample_panel_value)
    measurement_protocol_path = resolve_project_path(measurement_protocol_value)
    candidates = read_candidate_catalog(catalog_path)
    catalog_hash = candidate_catalog_sha256(candidates)
    contexts_hash = file_sha256(contexts_path)
    sample_panel_hash = file_sha256(sample_panel_path)
    artifact_binding = {
        "flow_map_checkpoint_sha256": file_sha256(
            resolve_project_path(str(args.flow_map_checkpoint))
        ),
        "backbone_checkpoint_sha256": file_sha256(
            resolve_project_path(str(args.flow_map_backbone_checkpoint))
        ),
        "gipo_checkpoint_sha256": file_sha256(
            resolve_project_path(str(args.flow_map_gipo_checkpoint))
        ),
    }
    scenario_key = _resolved_scenario_key(args)
    metric_payloads = [
        {
            "name": spec.name,
            "direction": spec.direction,
            "weight": float(spec.weight),
            "applicable_key": spec.applicable_key,
        }
        for spec in metric_specs_for_scenario(scenario_key)
    ]
    _, measurement_protocol_hash = read_quality_measurement_protocol(
        measurement_protocol_path,
        scenario_key=scenario_key,
        candidate_catalog_sha256=catalog_hash,
        quality_contexts_sha256=contexts_hash,
        quality_sample_panel_sha256=sample_panel_hash,
        artifact_binding=artifact_binding,
        primary_metrics=metric_payloads,
        bootstrap_samples=int(args.flow_map_bootstrap_samples),
        bootstrap_seed=int(args.flow_map_seed),
        familywise_alpha=float(args.flow_map_familywise_alpha),
    )
    return {
        "quality_candidate_catalog_sha256": catalog_hash,
        "quality_rows_sha256": file_sha256(rows_path),
        "quality_contexts_sha256": contexts_hash,
        "quality_sample_panel_sha256": sample_panel_hash,
        "quality_measurement_protocol_sha256": measurement_protocol_hash,
    }


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
    flow_map_quality_fields = (
        _flow_map_quality_protocol_fields(args) if flow_map_active else {}
    )
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
                **flow_map_quality_fields,
                "quality_status": (
                    "pending"
                    if str(args.flow_map_quality_rows_csv).strip()
                    else "not_evaluated"
                ),
                "performance_claim": False,
                "backbone_checkpoint": _display_optional_project_path(
                    args.flow_map_backbone_checkpoint
                ),
                "gipo_checkpoint": _display_optional_project_path(
                    args.flow_map_gipo_checkpoint
                ),
                "flow_map_checkpoint": _display_optional_project_path(
                    getattr(args, "flow_map_checkpoint", "")
                ),
                "contexts_npz": _display_optional_project_path(
                    args.flow_map_contexts_npz
                ),
                "contexts_source_sha256": _optional_project_file_sha256(
                    args.flow_map_contexts_npz
                ),
                "demonstration_manifest": _display_optional_project_path(
                    args.flow_map_demonstration_manifest
                ),
                "settings": str(args.flow_map_settings),
                "rollouts_per_context": int(args.flow_map_rollouts_per_context),
                "training_steps": int(args.flow_map_steps),
                "training_batch_size": int(args.flow_map_batch_size),
                "seed": int(args.flow_map_seed),
                "quality_rows_csv": _display_optional_project_path(
                    args.flow_map_quality_rows_csv
                ),
                "quality_candidate_catalog": _display_optional_project_path(
                    args.flow_map_quality_candidate_catalog
                ),
                "quality_contexts_npz": _display_optional_project_path(
                    args.flow_map_quality_contexts_npz
                ),
                "quality_sample_panel_npz": _display_optional_project_path(
                    args.flow_map_quality_sample_panel_npz
                ),
                "quality_measurement_protocol": _display_optional_project_path(
                    args.flow_map_quality_measurement_protocol
                ),
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
    input_args_by_module: dict[str, tuple[tuple[str, bool, bool], ...]] = {
        "genode.distillation.demonstrations": (
            ("--backbone-checkpoint", False, False),
            ("--gipo-checkpoint", False, False),
            ("--contexts-npz", False, False),
        ),
        "genode.distillation.training": (
            ("--demonstration-manifest", False, False),
            ("--backbone-checkpoint", False, False),
            ("--gipo-checkpoint", False, False),
        ),
        "genode.distillation.evaluation": (
            ("--rows-csv", False, False),
            ("--candidate-catalog", False, False),
            ("--quality-contexts-npz", False, False),
            ("--quality-sample-panel-npz", False, False),
            ("--measurement-protocol-json", False, False),
            ("--quality-protocol-json", False, False),
            ("--flow-map-checkpoint", False, False),
            ("--backbone-checkpoint", False, False),
            ("--gipo-checkpoint", False, False),
            ("--demonstration-manifest", False, False),
        ),
        "genode.gipo.train_gipo": (
            ("--rows_csv", True, False),
            ("--context_embeddings_npz", True, True),
            ("--schedule_summary_json", True, False),
            ("--teacher_unseen_selection_rows_csv", True, False),
            ("--teacher_unseen_selection_context_embeddings_npz", True, True),
            ("--teacher_unseen_selection_schedule_summary_json", True, False),
            ("--student_unseen_target_rows_csv", True, False),
            ("--student_unseen_target_context_embeddings_npz", True, True),
            ("--student_unseen_target_schedule_summary_json", True, False),
        ),
        "genode.gipo.report_locked_test": (
            ("--gipo_student_checkpoint", False, False),
            ("--training_summary", False, False),
            ("--context_rows", True, False),
            ("--context_embeddings_npz", True, True),
            ("--baseline_rows", True, False),
            ("--comparator_rows", True, False),
            ("--backbone_manifest", False, False),
        ),
        "genode.gipo.ser_ptg_reference": (
            ("--backbone_manifest", False, False),
        ),
    }
    input_hashes: dict[str, str] = {}
    for module, path_specs in input_args_by_module.items():
        if _command_module_args(command, module) is None:
            continue
        for path_arg, comma_separated, include_table_manifest in path_specs:
            value = _command_arg_value(command, path_arg)
            if not value:
                continue
            path_values = parse_csv(value) if comma_separated else [value]
            for index, path_value in enumerate(path_values):
                path = resolve_project_path(path_value)
                input_key = f"{path_arg}[{index}]"
                input_hashes[input_key] = (
                    file_sha256(path) if path.is_file() else "missing"
                )
                if include_table_manifest:
                    manifest_path = context_embedding_table_manifest_path(path)
                    input_hashes[f"{input_key}:manifest"] = (
                        file_sha256(manifest_path)
                        if manifest_path.is_file()
                        else "missing"
                    )
    input_hashes.update(_ser_reference_input_hashes(command))
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


def _ser_reference_selected_artifacts(
    command: Sequence[str],
) -> List[Mapping[str, Any]]:
    if _command_module_args(command, "genode.gipo.ser_ptg_reference") is None:
        return []
    manifest_value = _command_arg_value(command, "--backbone_manifest")
    scenario_key = _command_arg_value(command, "--scenario_key")
    checkpoint_step = _command_arg_value(command, "--checkpoint_step")
    if not manifest_value or not scenario_key or not checkpoint_step:
        return []
    manifest_path = resolve_project_path(manifest_value)
    if not manifest_path.is_file():
        return []
    try:
        benchmark_family = scenario_family_for_key(scenario_key)
        expected_backbone = (
            BACKBONE_NAME_OTFLOW_MOLECULE
            if benchmark_family == SCENARIO_FAMILY_MOLECULE
            else BACKBONE_NAME_OTFLOW
        )
        manifest = load_portable_backbone_manifest(manifest_path)
        selected = [
            artifact
            for artifact in manifest.get("artifacts", [])
            if isinstance(artifact, Mapping)
            and str(artifact.get("status", "")) == "ready"
            and str(artifact.get("backbone_name", "")) == expected_backbone
            and str(artifact.get("benchmark_family", "")) == benchmark_family
            and str(artifact.get("dataset_key", "")) == scenario_key
            and int(artifact.get("train_steps", -1)) == int(checkpoint_step)
        ]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    return sorted(
        selected,
        key=lambda artifact: (
            str(artifact.get("checkpoint_id", "")),
            str(artifact.get("checkpoint_path", "")),
        ),
    )


def _ser_reference_input_hashes(command: Sequence[str]) -> Dict[str, str]:
    if _command_module_args(command, "genode.gipo.ser_ptg_reference") is None:
        return {}
    input_hashes: Dict[str, str] = {}
    for artifact_index, artifact in enumerate(
        _ser_reference_selected_artifacts(command)
    ):
        identity = str(artifact.get("checkpoint_id", "") or artifact_index)
        for field in ("checkpoint_path", "metadata_path", "summary_path"):
            value = str(artifact.get(field, "") or "")
            path = Path(value) if value else None
            input_hashes[f"ser_artifact[{identity}]:{field}"] = (
                file_sha256(path) if path is not None and path.is_file() else "missing"
            )

    scenario_key = _command_arg_value(command, "--scenario_key")
    try:
        benchmark_family = scenario_family_for_key(scenario_key)
    except (KeyError, ValueError):
        benchmark_family = ""
    auxiliary_paths: List[Tuple[str, Path]] = []
    if benchmark_family == SCENARIO_FAMILY_FORECAST:
        dataset_root = _command_arg_value(command, "--dataset_root")
        if dataset_root:
            auxiliary_paths.append(
                (
                    "ser_dataset_manifest",
                    resolve_project_path(dataset_root)
                    / "monash"
                    / scenario_key
                    / "manifest.json",
                )
            )
    elif benchmark_family == SCENARIO_FAMILY_CONDITIONAL_GENERATION:
        option_by_scenario = {
            "cryptos": "--cryptos_path",
            "lobster_synthetic": "--lobster_synthetic_profile_path",
            "long_term_st": "--long_term_st_path",
        }
        option = option_by_scenario.get(scenario_key, "")
        value = _command_arg_value(command, option) if option else ""
        if value:
            path = resolve_project_path(value)
            auxiliary_paths.append(
                (
                    "ser_dataset_input",
                    long_term_st_manifest_path(path)
                    if scenario_key == "long_term_st"
                    else path,
                )
            )
    elif benchmark_family == SCENARIO_FAMILY_MOLECULE:
        group_root = _command_arg_value(command, "--molecule_group_root")
        if group_root:
            auxiliary_paths.append(
                (
                    "ser_molecule_group_manifest",
                    molecule_group_manifest_path(scenario_key, group_root),
                )
            )
    for key, path in auxiliary_paths:
        input_hashes[key] = file_sha256(path) if path.is_file() else "missing"
    return input_hashes


def _schedule_row_command_status(command: Sequence[str]) -> Dict[str, Any] | None:
    runner_args = _command_module_args(command, "genode.evaluation.diffusion_flow_time_reparameterization")
    if runner_args is None:
        return None
    parsed = schedule_runner.build_argparser().parse_args(runner_args)
    return schedule_runner.schedule_row_output_status(resolve_project_path(str(parsed.out_root)), parsed)


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


def _read_json_object(path: Path, *, artifact_label: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_label} must contain a JSON object")
    return payload


def _gipo_training_command_outputs_complete(
    command: Sequence[str],
) -> Tuple[bool, str]:
    module_args = _command_module_args(command, "genode.gipo.train_gipo")
    if module_args is None:
        return True, ""
    out_dir_value = _command_arg_value(command, "--out_dir")
    if not out_dir_value:
        return False, "GIPO training command is missing --out_dir"
    out_dir = resolve_project_path(out_dir_value)
    summary_path = out_dir / "gipo_training_summary.json"
    try:
        from genode.gipo import train_gipo as train_gipo_module

        parsed = train_gipo_module.build_argparser().parse_args(module_args)
        train_gipo_module.validate_gipo_training_bundle(out_dir)
        summary = _read_json_object(
            summary_path,
            artifact_label="GIPO training summary",
        )
        if str(summary.get("status", "")) != "completed":
            return False, "GIPO training summary status is not 'completed'"
        for checkpoint_kind in ("teacher", "student"):
            name_field = f"gipo_{checkpoint_kind}_checkpoint"
            hash_field = f"{name_field}_sha256"
            expected_name = f"gipo_{checkpoint_kind}.pt"
            if str(summary.get(name_field, "")) != expected_name:
                return False, f"GIPO training summary has invalid {name_field}"
            checkpoint_path = out_dir / expected_name
            if not checkpoint_path.is_file():
                return False, f"missing expected GIPO checkpoint: {checkpoint_path}"
            expected_hash = str(summary.get(hash_field, ""))
            if expected_hash != file_sha256(checkpoint_path):
                return False, f"GIPO training summary {hash_field} does not match"
        requested_policy = _command_arg_value(command, "--student_policy_key")
        if requested_policy and str(summary.get("student_policy_key", "")) != requested_policy:
            return False, "GIPO training summary student_policy_key does not match"
        input_names = (
            "rows_csv",
            "context_embeddings_npz",
            "schedule_summary_json",
            "teacher_unseen_selection_rows_csv",
            "teacher_unseen_selection_context_embeddings_npz",
            "teacher_unseen_selection_schedule_summary_json",
            "student_unseen_target_rows_csv",
            "student_unseen_target_context_embeddings_npz",
            "student_unseen_target_schedule_summary_json",
        )
        expected_inputs = {
            name: train_gipo_module._artifact_input_summary(
                str(getattr(parsed, name))
            )
            for name in input_names
        }
        if summary.get("training_inputs") != expected_inputs:
            return False, "GIPO training summary input fingerprints do not match"
        if int(summary.get("gipo_step_budget", -1)) != int(parsed.student_steps):
            return False, "GIPO training summary student step budget does not match"
        expected_seen_nfes = parse_int_csv(str(parsed.seen_target_nfe_values))
        expected_unseen_nfes = parse_int_csv(str(parsed.unseen_target_nfe_values))
        if list(summary.get("seen_target_nfe_values", [])) != expected_seen_nfes:
            return False, "GIPO training summary seen target NFEs do not match"
        if list(summary.get("unseen_target_nfe_values", [])) != expected_unseen_nfes:
            return False, "GIPO training summary unseen target NFEs do not match"
        if int(summary.get("context_sample_count", -1)) != int(
            parsed.context_sample_count
        ):
            return False, "GIPO training summary context sample count does not match"
        expected_support = parse_csv(str(parsed.support_schedule_keys))
        if expected_support and list(summary.get("support_schedule_keys", [])) != expected_support:
            return False, "GIPO training summary support schedules do not match"
        if not str(summary.get("policy_id", "")).strip():
            return False, "GIPO training summary is missing policy_id"
    except (OSError, TypeError, ValueError, json.JSONDecodeError, SystemExit) as exc:
        return False, f"invalid GIPO training artifact: {exc}"
    return True, ""


def _gipo_report_command_outputs_complete(
    command: Sequence[str],
) -> Tuple[bool, str]:
    if _command_module_args(command, "genode.gipo.report_locked_test") is None:
        return True, ""

    from genode.gipo.report_locked_test import (
        SELECTION_MODE_REPORTING,
        _output_prefix,
        _report_input_fingerprints,
    )

    out_dir_value = _command_arg_value(command, "--out_dir")
    checkpoint_value = _command_arg_value(command, "--gipo_student_checkpoint")
    if not out_dir_value or not checkpoint_value:
        return False, "GIPO report command is missing --out_dir or --gipo_student_checkpoint"
    out_dir = resolve_project_path(out_dir_value)
    split_phase = _command_arg_value(command, "--split_phase") or "locked_test"
    selection_mode = (
        _command_arg_value(command, "--selection_mode") or SELECTION_MODE_REPORTING
    )
    report_label = _command_arg_value(command, "--report_label")
    output_prefix = _output_prefix(split_phase, selection_mode, report_label)
    policy_summary_path = out_dir / f"{output_prefix}_policy_summary.json"
    comparison_summary_path = out_dir / f"{output_prefix}_comparison_summary.json"
    try:
        policy_summary = _read_json_object(
            policy_summary_path,
            artifact_label="GIPO policy summary",
        )
        comparison_summary = _read_json_object(
            comparison_summary_path,
            artifact_label="GIPO comparison summary",
        )
        checkpoint_path = resolve_project_path(checkpoint_value)
        checkpoint_hash = file_sha256(checkpoint_path)
        for label, payload in (
            ("policy", policy_summary),
            ("comparison", comparison_summary),
        ):
            if str(payload.get("gipo_student_checkpoint_sha256", "")) != checkpoint_hash:
                return False, f"GIPO {label} summary checkpoint hash does not match"

        input_args = {
            "gipo_student_checkpoint": "--gipo_student_checkpoint",
            "training_summary": "--training_summary",
            "context_rows": "--context_rows",
            "context_embeddings_npz": "--context_embeddings_npz",
            "baseline_rows": "--baseline_rows",
            "comparator_rows": "--comparator_rows",
        }
        expected_inputs = {
            name: _report_input_fingerprints(_command_arg_value(command, option))
            for name, option in input_args.items()
        }
        for label, payload in (
            ("policy", policy_summary),
            ("comparison", comparison_summary),
        ):
            if payload.get("report_inputs") != expected_inputs:
                return False, f"GIPO {label} summary input fingerprints do not match"

        expected_scalars = {
            "scenario_key": _command_arg_value(command, "--scenario_key"),
            "benchmark_family": _command_arg_value(command, "--benchmark_family"),
            "split_phase": split_phase,
        }
        for field, expected in expected_scalars.items():
            if expected and str(policy_summary.get(field, "")) != expected:
                return False, f"GIPO policy summary {field} does not match"
            if expected and str(comparison_summary.get(field, "")) != expected:
                return False, f"GIPO comparison summary {field} does not match"
        checkpoint_step = _command_arg_value(command, "--checkpoint_step")
        if checkpoint_step:
            expected_step = int(checkpoint_step)
            if int(policy_summary.get("checkpoint_step", -1)) != expected_step:
                return False, "GIPO policy summary checkpoint_step does not match"
            if int(comparison_summary.get("checkpoint_step", -1)) != expected_step:
                return False, "GIPO comparison summary checkpoint_step does not match"
        target_nfes = _command_arg_value(command, "--target_nfe_values")
        if target_nfes:
            expected_nfes = parse_int_csv(target_nfes)
            if list(policy_summary.get("target_nfe_values", [])) != expected_nfes:
                return False, "GIPO policy summary target_nfe_values do not match"
            if list(comparison_summary.get("target_nfe_values", [])) != expected_nfes:
                return False, "GIPO comparison summary target_nfe_values do not match"
        if str(policy_summary.get("comparison_summary_path", "")) != comparison_summary_path.name:
            return False, "GIPO policy summary comparison path does not match"
        for suffix in ("rows.csv", "aggregate_rows.csv", "decisions.csv"):
            output_path = out_dir / f"{output_prefix}_{suffix}"
            if not output_path.is_file():
                return False, f"missing expected GIPO report artifact: {output_path}"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid GIPO report artifact: {exc}"
    return True, ""


def _ser_reference_command_outputs_complete(
    command: Sequence[str],
) -> Tuple[bool, str]:
    module_args = _command_module_args(command, "genode.gipo.ser_ptg_reference")
    if module_args is None:
        return True, ""
    from genode.gipo import ser_ptg_reference

    try:
        parsed = ser_ptg_reference.build_argparser().parse_args(module_args)
        out_dir = resolve_project_path(str(parsed.out_dir))
        summary_path = out_dir / "ser_ptg_schedule_summary.json"
        summary = _read_json_object(
            summary_path,
            artifact_label="SER-PTG schedule summary",
        )
        solvers = parse_csv(str(parsed.solver_names)) or list(SUPPORTED_SOLVER_KEYS)
        target_nfes = parse_int_csv(
            str(parsed.target_nfe_values), default=REFERENCE_SEEN_NFES
        )
        seeds = parse_int_csv(str(parsed.seeds), default=(0, 1, 2))
        if bool(parsed.smoke):
            solvers = solvers[:1]
            target_nfes = target_nfes[:1]

        expected_scalars: Dict[str, Any] = {
            "status": "ready",
            "artifact": "ser_ptg_schedule_summary",
            "scenario_key": str(parsed.scenario_key),
            "example_selection_protocol": SER_PTG_EXAMPLE_SELECTION_PROTOCOL,
            "local_defect_trace_protocol": SER_PTG_LOCAL_DEFECT_PROXY_PROTOCOL,
            "reference_split": str(parsed.reference_split),
            "reference_split_key": (
                "train" if str(parsed.reference_split) == "train_tuning" else "val"
            ),
            "checkpoint_step": int(parsed.checkpoint_step),
            "context_sample_count": int(parsed.context_sample_count),
            "calibration_trace_samples": int(parsed.calibration_trace_samples),
            "density_floor_eta": float(parsed.density_floor_eta),
            "reference_macro_factor": float(parsed.reference_macro_factor),
        }
        for field, expected in expected_scalars.items():
            if summary.get(field) != expected:
                return False, f"SER-PTG schedule summary {field} does not match"
        expected_lists = {
            "solver_names": [str(value) for value in solvers],
            "target_nfe_values": [int(value) for value in target_nfes],
            "seeds": [int(value) for value in seeds],
        }
        for field, expected in expected_lists.items():
            if list(summary.get(field, [])) != expected:
                return False, f"SER-PTG schedule summary {field} does not match"

        selected_artifacts = _ser_reference_selected_artifacts(command)
        family = scenario_family_for_key(str(parsed.scenario_key))
        if family == SCENARIO_FAMILY_MOLECULE:
            if not selected_artifacts:
                return False, "SER-PTG command has no selected molecule backbone artifacts"
        elif len(selected_artifacts) != 1:
            return False, (
                "SER-PTG command must select exactly one temporal backbone artifact; "
                f"found {len(selected_artifacts)}"
            )
        expected_checkpoint_ids = sorted(
            str(artifact.get("checkpoint_id", ""))
            for artifact in selected_artifacts
        )
        if list(summary.get("checkpoint_ids", [])) != expected_checkpoint_ids:
            return False, "SER-PTG schedule summary checkpoint_ids do not match"

        predictions = summary.get("predictions")
        if not isinstance(predictions, list):
            return False, "SER-PTG schedule summary predictions must be a list"
        expected_pairs = {
            (str(solver), int(target_nfe))
            for solver in solvers
            for target_nfe in target_nfes
        }
        observed_pairs: set[Tuple[str, int]] = set()
        for prediction in predictions:
            if not isinstance(prediction, Mapping):
                return False, "SER-PTG schedule summary contains an invalid prediction"
            pair = (
                str(prediction.get("solver_key", "")),
                int(prediction.get("target_nfe", -1)),
            )
            if pair in observed_pairs:
                return False, "SER-PTG schedule summary contains duplicate predictions"
            observed_pairs.add(pair)
            if int(prediction.get("checkpoint_step", -1)) != int(
                parsed.checkpoint_step
            ):
                return False, "SER-PTG prediction checkpoint_step does not match"
            if str(prediction.get("reference_split", "")) != str(
                parsed.reference_split
            ):
                return False, "SER-PTG prediction reference_split does not match"
        if observed_pairs != expected_pairs:
            return False, "SER-PTG schedule summary prediction matrix does not match"
        if int(summary.get("prediction_count", -1)) != len(predictions):
            return False, "SER-PTG schedule summary prediction_count does not match"
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        SystemExit,
    ) as exc:
        return False, f"invalid SER-PTG schedule artifact: {exc}"
    return True, ""


def _flow_map_command_outputs_complete(command: Sequence[str]) -> Tuple[bool, str]:
    demonstration_args = _command_module_args(command, "genode.distillation.demonstrations")
    if demonstration_args is not None:
        from genode.distillation.artifacts import (
            context_binding,
            load_demonstration_manifest,
            validate_context_binding,
        )
        from genode.distillation.demonstrations import (
            load_distillation_contexts,
            parse_distillation_settings,
        )

        output_dir = _command_arg_value(command, "--output-dir")
        try:
            manifest = load_demonstration_manifest(
                resolve_project_path(output_dir) / DEMONSTRATION_MANIFEST_NAME
            )
            metadata = dict(manifest["metadata"])
            source_paths = {
                "backbone_checkpoint_sha256": _command_arg_value(
                    command, "--backbone-checkpoint"
                ),
                "gipo_checkpoint_sha256": _command_arg_value(
                    command, "--gipo-checkpoint"
                ),
                "contexts_source_sha256": _command_arg_value(
                    command, "--contexts-npz"
                ),
            }
            if str(metadata.get("contexts_source_kind", "")) != "npz":
                return False, "flow-map demonstration source is not the commanded NPZ"
            for field, source_path in source_paths.items():
                if not source_path or str(metadata.get(field, "")) != file_sha256(
                    resolve_project_path(source_path)
                ):
                    return False, f"flow-map demonstration {field} does not match"
            expected_scalars = {
                "split_phase": _command_arg_value(command, "--split-phase"),
                "scenario_key": _command_arg_value(command, "--scenario-key"),
                "benchmark_family": _command_arg_value(
                    command, "--benchmark-family"
                ),
            }
            for field, expected in expected_scalars.items():
                if expected and str(metadata.get(field, "")) != expected:
                    return False, f"flow-map demonstration {field} does not match"
            expected_settings = [
                {
                    "solver_key": setting.solver_key,
                    "target_nfe": int(setting.target_nfe),
                }
                for setting in parse_distillation_settings(
                    _command_arg_value(command, "--settings")
                )
            ]
            if metadata.get("settings") != expected_settings:
                return False, "flow-map demonstration settings do not match"
            expected_rollouts = int(
                _command_arg_value(command, "--rollouts-per-context") or 4
            )
            if int(metadata.get("rollouts_per_context", -1)) != expected_rollouts:
                return False, "flow-map demonstration rollout count does not match"
            expected_seed = int(_command_arg_value(command, "--seed") or 0)
            if int(metadata.get("collection_seed", -1)) != expected_seed:
                return False, "flow-map demonstration collection seed does not match"
            source_contexts = load_distillation_contexts(
                resolve_project_path(_command_arg_value(command, "--contexts-npz"))
            )
            source_binding = context_binding(
                source_contexts.content_fingerprints()
            )
            if validate_context_binding(
                metadata.get("context_binding", {})
            ) != source_binding:
                return False, "flow-map demonstration context binding does not match source NPZ"
        except (OSError, TypeError, ValueError) as exc:
            return False, f"invalid flow-map demonstration artifact: {exc}"

    training_args = _command_module_args(command, "genode.distillation.training")
    if training_args is not None:
        from genode.distillation.artifacts import load_demonstration_manifest
        from genode.distillation.checkpoint import load_flow_map_checkpoint
        from genode.distillation.training import (
            build_argparser as build_flow_map_training_argparser,
            validate_flow_map_bundle,
        )

        try:
            parsed = build_flow_map_training_argparser().parse_args(training_args)
            checkpoint_path = resolve_project_path(parsed.output_checkpoint)
            backbone_path = resolve_project_path(parsed.backbone_checkpoint)
            gipo_path = resolve_project_path(parsed.gipo_checkpoint)
            manifest_path = resolve_project_path(parsed.demonstration_manifest)
            if not str(parsed.summary_json).strip():
                return False, "flow-map training command is missing --summary-json"
            summary_path = resolve_project_path(parsed.summary_json)
            validate_flow_map_bundle(checkpoint_path, summary_path)
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
            if not isinstance(summary, dict):
                return False, "flow-map training summary must be an object"
            if str(summary.get("status", "")) != "completed":
                return False, "flow-map training summary status is not 'completed'"
            if str(summary.get("checkpoint_name", "")) != checkpoint_path.name:
                return False, "flow-map training summary checkpoint name does not match"
            if str(summary.get("checkpoint_sha256", "")) != file_sha256(checkpoint_path):
                return False, "flow-map training summary checkpoint hash does not match"
            embedded_summary = payload.get("training_summary")
            if not isinstance(embedded_summary, Mapping):
                return False, "flow-map checkpoint is missing its training summary"
            report_summary = {
                key: value
                for key, value in summary.items()
                if key not in {"status", "checkpoint_name", "checkpoint_sha256"}
            }
            if report_summary != dict(embedded_summary):
                return False, "flow-map file and checkpoint training summaries disagree"
            expected_summary_fields = {
                "steps": int(parsed.steps),
                "batch_size": int(parsed.batch_size),
                "learning_rate": float(parsed.learning_rate),
                "weight_decay": float(parsed.weight_decay),
                "grad_clip": float(parsed.grad_clip),
                "seed": int(parsed.seed),
                "split_seed": int(parsed.seed),
                "validation_fraction": float(parsed.validation_fraction),
                "validation_interval": int(parsed.validation_interval),
                "validation_shards": int(parsed.validation_shards),
                "validation_seed": int(parsed.seed) + 1_000_003,
                "batches_per_shard": int(parsed.batches_per_shard),
                "initialized_from_backbone_field": not bool(
                    parsed.no_backbone_initialization
                ),
            }
            for field, expected in expected_summary_fields.items():
                if summary.get(field) != expected:
                    return False, f"flow-map training summary {field} does not match"
        except (OSError, RuntimeError, SystemExit, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"invalid flow-map training artifact: {exc}"

    evaluation_args = _command_module_args(command, "genode.distillation.evaluation")
    if evaluation_args is not None:
        from genode.distillation.artifacts import (
            load_demonstration_manifest,
            validate_context_binding,
        )
        from genode.distillation.checkpoint import load_flow_map_checkpoint
        from genode.distillation.evaluation import (
            QualityGateConfig,
            candidate_catalog_sha256,
            evaluate_quality_gate,
            metric_specs_for_scenario,
            not_evaluated_report,
            read_candidate_catalog,
            read_quality_contexts,
            read_quality_protocol,
            read_quality_rows,
            read_quality_sample_panel,
        )
        from genode.distillation.measurement_protocol import (
            read_quality_measurement_protocol,
        )

        output_path = resolve_project_path(_command_arg_value(command, "--output-json"))
        flow_map_path = resolve_project_path(_command_arg_value(command, "--flow-map-checkpoint"))
        backbone_path = resolve_project_path(_command_arg_value(command, "--backbone-checkpoint"))
        gipo_path = resolve_project_path(_command_arg_value(command, "--gipo-checkpoint"))
        scenario_key = _command_arg_value(command, "--scenario-key")
        manifest_value = _command_arg_value(command, "--demonstration-manifest")
        try:
            quality_protocol_path = resolve_project_path(
                _command_arg_value(command, "--quality-protocol-json")
            )
            quality_protocol, quality_protocol_hash = read_quality_protocol(
                quality_protocol_path,
                scenario_key=scenario_key,
            )
            _, payload = load_flow_map_checkpoint(
                flow_map_path,
                backbone_checkpoint=backbone_path,
                gipo_checkpoint=gipo_path,
            )
            checkpoint_scenario = str(payload.get("scenario_key", "")).strip()
            if checkpoint_scenario and checkpoint_scenario != scenario_key:
                return False, "flow-map quality checkpoint scenario does not match"
            checkpoint_family = str(payload.get("benchmark_family", "")).strip()
            try:
                expected_family = scenario_family_for_key(scenario_key)
            except KeyError:
                expected_family = ""
            if (
                expected_family
                and checkpoint_family
                and checkpoint_family != expected_family
            ):
                return False, "flow-map quality checkpoint family does not match"
            binding = {
                "scenario_key": scenario_key,
                "flow_map_checkpoint_sha256": file_sha256(flow_map_path),
                "backbone_checkpoint_sha256": str(payload["backbone_checkpoint_sha256"]),
                "gipo_checkpoint_sha256": str(payload["gipo_checkpoint_sha256"]),
            }
            rows_path_value = _command_arg_value(command, "--rows-csv")
            if rows_path_value:
                if manifest_value:
                    manifest_path = resolve_project_path(manifest_value)
                    if file_sha256(manifest_path) != str(
                        payload["demonstration_manifest_sha256"]
                    ):
                        return False, "flow-map quality manifest hash does not match"
                    manifest = load_demonstration_manifest(manifest_path)
                    manifest_metadata = dict(manifest["metadata"])
                    if str(manifest_metadata.get("scenario_key", "")).strip() != scenario_key:
                        return False, "flow-map quality manifest scenario does not match"
                    manifest_family = str(
                        manifest_metadata.get("benchmark_family", "")
                    ).strip()
                    if checkpoint_family and manifest_family != checkpoint_family:
                        return False, "flow-map quality manifest family does not match checkpoint"
                    if expected_family and manifest_family != expected_family:
                        return False, "flow-map quality manifest family does not match scenario"
                    context_binding = validate_context_binding(
                        manifest_metadata["context_binding"]
                    )
                    checkpoint_context_binding = payload.get(
                        "demonstration_context_binding"
                    )
                    if checkpoint_context_binding is not None and validate_context_binding(
                        checkpoint_context_binding
                    ) != context_binding:
                        return False, "flow-map quality manifest binding does not match checkpoint"
                else:
                    context_binding = validate_context_binding(
                        payload.get("demonstration_context_binding", {})
                    )
                rows_path = resolve_project_path(rows_path_value)
                catalog_path = resolve_project_path(
                    _command_arg_value(command, "--candidate-catalog")
                )
                quality_contexts_path = resolve_project_path(
                    _command_arg_value(command, "--quality-contexts-npz")
                )
                sample_panel_path = resolve_project_path(
                    _command_arg_value(command, "--quality-sample-panel-npz")
                )
                measurement_protocol_path = resolve_project_path(
                    _command_arg_value(command, "--measurement-protocol-json")
                )
                catalog = read_candidate_catalog(catalog_path)
                quality_contexts = read_quality_contexts(quality_contexts_path)
                quality_sample_panel = read_quality_sample_panel(
                    sample_panel_path,
                    quality_context_binding=quality_contexts,
                )
                rows_hash = file_sha256(rows_path)
                metric_specs = metric_specs_for_scenario(scenario_key)
                metric_payloads = [
                    {
                        "name": spec.name,
                        "direction": spec.direction,
                        "weight": float(spec.weight),
                        "applicable_key": spec.applicable_key,
                    }
                    for spec in metric_specs
                ]
                measurement_protocol, measurement_protocol_hash = (
                    read_quality_measurement_protocol(
                        measurement_protocol_path,
                        scenario_key=scenario_key,
                        candidate_catalog_sha256=candidate_catalog_sha256(catalog),
                        quality_contexts_sha256=quality_contexts["artifact_sha256"],
                        quality_sample_panel_sha256=quality_sample_panel[
                            "artifact_sha256"
                        ],
                        artifact_binding={
                            "flow_map_checkpoint_sha256": binding[
                                "flow_map_checkpoint_sha256"
                            ],
                            "backbone_checkpoint_sha256": binding[
                                "backbone_checkpoint_sha256"
                            ],
                            "gipo_checkpoint_sha256": binding[
                                "gipo_checkpoint_sha256"
                            ],
                        },
                        primary_metrics=metric_payloads,
                        bootstrap_samples=int(
                            _command_arg_value(command, "--bootstrap-samples")
                        ),
                        bootstrap_seed=int(_command_arg_value(command, "--seed")),
                        familywise_alpha=float(
                            _command_arg_value(command, "--familywise-alpha")
                        ),
                    )
                )
                expected = evaluate_quality_gate(
                    read_quality_rows(rows_path),
                    metric_specs=metric_specs,
                    candidate_catalog=catalog,
                    artifact_binding=binding,
                    demonstration_context_binding=context_binding,
                    quality_context_binding=quality_contexts,
                    quality_sample_panel_binding=quality_sample_panel,
                    quality_protocol=quality_protocol,
                    quality_rows_sha256=rows_hash,
                    measurement_protocol=measurement_protocol,
                    measurement_protocol_sha256=measurement_protocol_hash,
                    config=QualityGateConfig(
                        bootstrap_samples=int(_command_arg_value(command, "--bootstrap-samples")),
                        familywise_alpha=float(_command_arg_value(command, "--familywise-alpha")),
                        margin=0.0,
                        seed=int(_command_arg_value(command, "--seed")),
                    ),
                )
                expected["rows_sha256"] = rows_hash
                expected["candidate_catalog_file_sha256"] = file_sha256(catalog_path)
            else:
                expected = not_evaluated_report(
                    reason=_command_arg_value(command, "--not-evaluated-reason"),
                    artifact_binding=binding,
                )
                expected["quality_protocol_hash"] = quality_protocol_hash
            actual = json.loads(output_path.read_text(encoding="utf-8"))
            if actual != expected:
                return False, "flow-map quality report does not match its bound inputs"
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return False, f"invalid flow-map quality artifact: {exc}"
    return True, ""


def _accepted_flow_map_quality_rejection(
    command: Sequence[str],
) -> Tuple[bool, str]:
    if _command_module_args(command, "genode.distillation.evaluation") is None:
        return False, "command is not a flow-map evaluation"
    if not _command_arg_value(command, "--rows-csv"):
        return False, "a not-evaluated report cannot be accepted as a quality-gate rejection"
    outputs_complete, reason = _flow_map_command_outputs_complete(command)
    if not outputs_complete:
        return False, reason
    output_path = resolve_project_path(_command_arg_value(command, "--output-json"))
    try:
        report = _read_json_object(
            output_path,
            artifact_label="flow-map quality report",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid flow-map quality rejection report: {exc}"
    if str(report.get("status", "")) != "failed":
        return False, "quality-gate rejection report status is not 'failed'"
    if report.get("performance_claim") is not False:
        return False, "quality-gate rejection report must deny the performance claim"
    if report.get("locked_test_used_for_selection") is not False:
        return False, "quality-gate rejection report must preserve locked-test isolation"
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
        ok, reason = _gipo_training_command_outputs_complete(command)
        if not ok:
            return False, reason
        ok, reason = _gipo_report_command_outputs_complete(command)
        if not ok:
            return False, reason
        schedule_status = _schedule_row_command_status(command)
        if schedule_status is not None and not bool(schedule_status.get("complete", False)):
            return False, str(schedule_status.get("reason", "schedule-row output is incomplete"))
        ok, reason = _ser_reference_command_outputs_complete(command)
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
        returncode = int(result.get("returncode", -1))
        if returncode == 0:
            continue
        accepted_rejection = (
            returncode == 2
            and entry.stage == FLOW_MAP_EVALUATION_STAGE
            and result.get("accepted_quality_gate_result")
            == QUALITY_GATE_REJECTION_RESULT
        )
        if not accepted_rejection:
            return False, f"stage command {idx} did not complete successfully"
        rejection_valid, reason = _accepted_flow_map_quality_rejection(
            entry.commands[idx]
        )
        if not rejection_valid:
            return False, f"stage command {idx} has an invalid accepted rejection: {reason}"
    outputs_complete, reason = _stage_outputs_complete(entry)
    if not outputs_complete:
        return False, reason
    return True, ""


def _interrupted_stage_manifest_matches(
    run_root: Path,
    entry: StageCommand,
    *,
    protocol_hash: str,
) -> Tuple[bool, str]:
    path = run_root / entry.manifest_name
    if not path.is_file():
        return False, f"missing interrupted stage manifest: {path}"
    try:
        manifest = _read_json_object(
            path,
            artifact_label="interrupted stage manifest",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid interrupted stage manifest: {exc}"
    if str(manifest.get("status", "")) != "running":
        return False, "stage manifest is not an interrupted running stage"
    if manifest.get("dry_run") is not False:
        return False, "interrupted stage manifest is not a non-dry-run manifest"
    if str(manifest.get("stage", "")) != str(entry.stage):
        return False, "interrupted stage manifest stage does not match"
    if str(manifest.get("protocol_hash", "")) != str(protocol_hash):
        return False, "interrupted stage manifest protocol hash does not match"
    expected_commands = [
        _display_command(command, path_base=run_root) for command in entry.commands
    ]
    if manifest.get("commands") != expected_commands:
        return False, "interrupted stage manifest commands do not match"
    expected_hashes = [_command_hash(command) for command in entry.commands]
    if manifest.get("command_hashes") != expected_hashes:
        return False, "interrupted stage manifest command hashes do not match"
    results = manifest.get("command_results")
    if not isinstance(results, list) or len(results) > len(entry.commands):
        return False, "interrupted stage manifest command_results are invalid"
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            return False, "interrupted stage manifest contains an invalid command result"
        try:
            result_index = int(result.get("command_index", -1))
            returncode = int(result.get("returncode", -1))
        except (TypeError, ValueError):
            return False, "interrupted stage manifest contains an invalid command result"
        if result_index != index or returncode != 0:
            return False, "interrupted stage manifest command_results do not match"
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
    stage_filter = set(_selected_stage_names(args))
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
    flow_map_backbone_checkpoint = _resolve_optional_project_path(
        args.flow_map_backbone_checkpoint
    )
    flow_map_contexts_npz = _resolve_optional_project_path(args.flow_map_contexts_npz)
    flow_map_quality_rows = _resolve_optional_project_path(
        args.flow_map_quality_rows_csv
    )
    flow_map_quality_catalog = _resolve_optional_project_path(
        args.flow_map_quality_candidate_catalog
    )
    flow_map_quality_contexts = _resolve_optional_project_path(
        args.flow_map_quality_contexts_npz
    )
    flow_map_quality_sample_panel = _resolve_optional_project_path(
        args.flow_map_quality_sample_panel_npz
    )
    flow_map_quality_measurement_protocol = _resolve_optional_project_path(
        args.flow_map_quality_measurement_protocol
    )
    flow_map_gipo_checkpoint = str(getattr(args, "flow_map_gipo_checkpoint", "") or "").strip()
    if flow_map_gipo_checkpoint:
        flow_map_gipo_checkpoint = str(resolve_project_path(flow_map_gipo_checkpoint))
    else:
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
        if bool(args.overwrite):
            command_args.append("--overwrite")
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
            flow_map_backbone_checkpoint,
            "--gipo-checkpoint",
            flow_map_gipo_checkpoint,
            "--contexts-npz",
            flow_map_contexts_npz,
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
            flow_map_backbone_checkpoint,
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
            flow_map_backbone_checkpoint,
            "--gipo-checkpoint",
            flow_map_gipo_checkpoint,
            "--bootstrap-samples",
            int(args.flow_map_bootstrap_samples),
            "--familywise-alpha",
            float(args.flow_map_familywise_alpha),
            "--seed",
            int(args.flow_map_seed),
            "--quality-protocol-json",
            run_root / "protocol.json",
        ]
        if flow_map_quality_rows:
            command_args.extend(["--rows-csv", flow_map_quality_rows])
            command_args.extend(
                [
                    "--candidate-catalog",
                    flow_map_quality_catalog,
                    "--quality-contexts-npz",
                    flow_map_quality_contexts,
                    "--quality-sample-panel-npz",
                    flow_map_quality_sample_panel,
                    "--measurement-protocol-json",
                    flow_map_quality_measurement_protocol,
                ]
            )
        else:
            command_args.extend(
                [
                    "--not-evaluated-reason",
                    "No benchmark rows were supplied; this artifact has no performance claim.",
                ]
            )
        if (
            explicit_demonstration_manifest
            or FLOW_MAP_COLLECTION_STAGE in stage_filter
            or FLOW_MAP_TRAINING_STAGE in stage_filter
        ):
            command_args.extend(
                ["--demonstration-manifest", demonstration_manifest]
            )
        if bool(args.overwrite):
            command_args.append("--overwrite")
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
    adopted_stage_names: List[str] = []
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
        interrupted_manifest_matches = False
        if bool(args.resume) and not bool(args.overwrite):
            interrupted_manifest_matches, _ = (
                _interrupted_stage_manifest_matches(
                    run_root,
                    entry,
                    protocol_hash=protocol_hash,
                )
            )
        manifest = {
            "stage": entry.stage,
            "protocol_hash": protocol_hash,
            "commands": [_display_command(command, path_base=run_root) for command in entry.commands],
            "command_hashes": [_command_hash(command) for command in entry.commands],
            "dry_run": bool(args.dry_run),
            "command_results": [],
            "status": "planned" if bool(args.dry_run) else "running",
        }
        if entry.stage == INPUT_PREFLIGHT_STAGE:
            manifest["preflight"] = preflight
        _write_json(run_root / entry.manifest_name, manifest)
        if not bool(args.dry_run):
            for command_idx, command in enumerate(entry.commands):
                if command and command[0] == "internal":
                    raise RuntimeError(
                        "Unresolved internal pipeline command in non-dry-run: "
                        f"{_display_command(command, path_base=run_root)}"
                    )
                log_path = run_root / "logs" / f"{entry.stage}_{command_idx}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if interrupted_manifest_matches:
                    gipo_validator = None
                    if (
                        _command_module_args(command, "genode.gipo.train_gipo")
                        is not None
                    ):
                        gipo_validator = _gipo_training_command_outputs_complete
                    elif (
                        _command_module_args(
                            command, "genode.gipo.report_locked_test"
                        )
                        is not None
                    ):
                        gipo_validator = _gipo_report_command_outputs_complete
                    if gipo_validator is not None:
                        gipo_complete, _ = gipo_validator(command)
                        if gipo_complete:
                            manifest["command_results"].append(
                                {
                                    "command_index": int(command_idx),
                                    "returncode": 0,
                                    "log_path": str(log_path.relative_to(run_root)),
                                    "skipped": True,
                                    "skip_reason": (
                                        "verified GIPO output already complete after "
                                        "an exact interrupted-stage match"
                                    ),
                                }
                            )
                            _write_json(run_root / entry.manifest_name, manifest)
                            continue
                if (
                    bool(args.resume)
                    and not bool(args.overwrite)
                    and interrupted_manifest_matches
                    and entry.stage
                    in {
                        FLOW_MAP_COLLECTION_STAGE,
                        FLOW_MAP_TRAINING_STAGE,
                        FLOW_MAP_EVALUATION_STAGE,
                    }
                ):
                    flow_map_complete, _ = _flow_map_command_outputs_complete(
                        command
                    )
                    if flow_map_complete:
                        accepted_rejection, _ = (
                            _accepted_flow_map_quality_rejection(command)
                            if entry.stage == FLOW_MAP_EVALUATION_STAGE
                            else (False, "")
                        )
                        command_result = {
                            "command_index": int(command_idx),
                            "returncode": 2 if accepted_rejection else 0,
                            "log_path": str(log_path.relative_to(run_root)),
                            "skipped": True,
                            "skip_reason": (
                                "verified flow-map output already complete"
                            ),
                        }
                        if accepted_rejection:
                            command_result["accepted_quality_gate_result"] = (
                                QUALITY_GATE_REJECTION_RESULT
                            )
                        manifest["command_results"].append(command_result)
                        _write_json(run_root / entry.manifest_name, manifest)
                        continue
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
                            "skipped_stages": [
                                *skipped_stage_names,
                                *adopted_stage_names,
                            ],
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
                    accepted_rejection = False
                    rejection_reason = ""
                    if (
                        int(result.returncode) == 2
                        and entry.stage == FLOW_MAP_EVALUATION_STAGE
                    ):
                        accepted_rejection, rejection_reason = (
                            _accepted_flow_map_quality_rejection(command)
                        )
                    if accepted_rejection:
                        command_result["accepted_quality_gate_result"] = (
                            QUALITY_GATE_REJECTION_RESULT
                        )
                        _write_json(run_root / entry.manifest_name, manifest)
                        continue
                    if rejection_reason:
                        command_result["error"] = rejection_reason
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
                            "skipped_stages": [
                                *skipped_stage_names,
                                *adopted_stage_names,
                            ],
                        },
                    )
                    raise RuntimeError(f"Pipeline stage {entry.stage!r} failed; see {log_path}.")
            outputs_complete, output_validation_error = _stage_outputs_complete(
                entry
            )
            if not outputs_complete:
                manifest["status"] = "failed"
                manifest["output_validation_error"] = output_validation_error
                _write_json(run_root / entry.manifest_name, manifest)
                if has_ablation_stage:
                    failed_ablation_manifest = _build_ablation_manifest(
                        args,
                        run_root,
                        protocol_hash=protocol_hash,
                        status="failed",
                        extra={
                            "failed_stage": entry.stage,
                            "failure_reason": output_validation_error,
                        },
                    )
                    _write_json(
                        _ablation_root(run_root, str(args.ablation_preset))
                        / "ablation_manifest.json",
                        failed_ablation_manifest,
                    )
                _write_json(
                    _status_path(run_root),
                    {
                        "version": PIPELINE_VERSION,
                        "status": "failed",
                        "failed_stage": entry.stage,
                        "protocol_hash": protocol_hash,
                        "completed_stages": completed,
                        "skipped_stages": [
                            *skipped_stage_names,
                            *adopted_stage_names,
                        ],
                        "failure_reason": output_validation_error,
                    },
                )
                raise RuntimeError(
                    f"Pipeline stage {entry.stage!r} produced invalid or incomplete "
                    f"outputs: {output_validation_error}"
                )
        manifest["status"] = "planned" if bool(args.dry_run) else "complete"
        _write_json(run_root / entry.manifest_name, manifest)
        if (
            not bool(args.dry_run)
            and len(manifest["command_results"]) == len(entry.commands)
            and bool(entry.commands)
            and all(
                result.get("skipped") is True
                for result in manifest["command_results"]
            )
            and any(
                str(result.get("skip_reason", "")).startswith(
                    ("verified flow-map output", "verified GIPO output")
                )
                for result in manifest["command_results"]
            )
        ):
            adopted_stage_names.append(entry.stage)
        completed.append(entry.stage)
        _write_json(
            _status_path(run_root),
            {
                "version": PIPELINE_VERSION,
                "status": "dry_run" if bool(args.dry_run) else "running",
                "protocol_hash": protocol_hash,
                "completed_stages": completed,
                "skipped_stages": [
                    *skipped_stage_names,
                    *adopted_stage_names,
                ],
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
        "skipped_stages": [*skipped_stage_names, *adopted_stage_names],
        "executed_stages": [
            entry.stage
            for entry in commands_to_run
            if entry.stage not in set(adopted_stage_names)
        ],
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
    parser.add_argument(
        "--flow_map_quality_candidate_catalog",
        default="",
        help=(
            "JSON catalog that prespecifies the exact flow-map, GIPO, and fixed "
            "solver/NFE candidates covered by quality rows."
        ),
    )
    parser.add_argument(
        "--flow_map_quality_contexts_npz",
        default="",
        help=(
            "Validation and locked-test physical contexts used to recompute quality "
            "row fingerprints."
        ),
    )
    parser.add_argument(
        "--flow_map_quality_sample_panel_npz",
        default="",
        help=(
            "Common logical seeds and initial states used by every quality candidate."
        ),
    )
    parser.add_argument(
        "--flow_map_quality_measurement_protocol",
        default="",
        help=(
            "External measurement protocol binding the evaluated artifacts, runner, "
            "reference data, candidates, metrics, contexts, sample panel, and gate settings."
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
