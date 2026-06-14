from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from genode.data.experiment_common import DATASET_PLANS, build_dataset_splits, get_otflow_paper_backbone_preset
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY, FORECAST_FAMILY, experiment_plan_by_key
from genode.data.otflow_forecast_data import build_monash_forecast_splits
from genode.data.otflow_monash_datasets import default_manifest_path, download_monash_dataset
from genode.data.otflow_paths import (
    default_cryptos_data_path,
    default_lobster_synthetic_profile_path,
    default_long_term_st_data_path,
    project_paper_dataset_root,
    project_root,
    resolve_project_path,
)
from genode.evaluation.fm_backbone_registry import (
    ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS,
    ACTIVE_FORECAST_BACKBONE_BUDGETS,
    BACKBONE_NAME_OTFLOW,
    DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE,
    build_backbone_checkpoint_id,
    expected_artifact_root,
    materialize_backbone_manifest,
    project_backbone_matrix_root,
    train_budget_label,
)
from genode.models.config import OTFlowConfig
from genode.models.otflow_train_val import capture_rng_state, evaluate_average_loss, save_json, seed_all, train_loop
from genode.runtime import resolve_torch_device

DEFAULT_DATASET = "traffic_hourly"
DEFAULT_HIDDEN_DIM = 160
CHECKPOINT_EXPORT_MODE_EXACT_BUDGET = "exact_budget"
CHECKPOINT_EXPORT_MODE_BEST_VALIDATION = "best_validation_within_budget"
CHECKPOINT_EXPORT_MODES = (CHECKPOINT_EXPORT_MODE_EXACT_BUDGET, CHECKPOINT_EXPORT_MODE_BEST_VALIDATION)
CHECKPOINT_EXPORT_PROTOCOL_EXACT_BUDGET = "exact_budget_step_state"
CHECKPOINT_EXPORT_PROTOCOL_BEST_VALIDATION = "best_validation_state_within_budget"
TEMPORAL_BACKBONE_TRAINING_STATE_VERSION = "temporal_otflow_backbone_training_state_v1"


def _project_display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return resolved.name


def _json_ready_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = _json_ready_stats(value)
        else:
            out[key] = str(value)
    return out


def _dataset_spec(dataset: str):
    plans = experiment_plan_by_key()
    if str(dataset) not in plans:
        raise KeyError(f"Unknown dataset={dataset!r}; expected one of {sorted(plans)}.")
    return plans[str(dataset)]


def _forecast_spec(dataset: str):
    spec = _dataset_spec(str(dataset))
    if str(spec.benchmark_family) != FORECAST_FAMILY:
        raise ValueError(f"{dataset!r} has benchmark_family={spec.benchmark_family!r}, not {FORECAST_FAMILY!r}.")
    return spec


def _conditional_spec(dataset: str):
    spec = _dataset_spec(str(dataset))
    if str(spec.benchmark_family) != CONDITIONAL_GENERATION_FAMILY:
        raise ValueError(
            f"{dataset!r} has benchmark_family={spec.benchmark_family!r}, not {CONDITIONAL_GENERATION_FAMILY!r}."
        )
    return spec


def build_forecast_cfg(args: argparse.Namespace) -> OTFlowConfig:
    spec = experiment_plan_by_key()[str(args.dataset)]
    cfg = OTFlowConfig()
    cfg.apply_overrides(
        device=resolve_torch_device(str(args.device)),
        levels=1,
        token_dim=1,
        history_len=int(spec.history_len),
        steps=int(args.steps),
        batch_size=int(args.batch_size) if int(args.batch_size) > 0 else 64,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        standardize=True,
        hidden_dim=int(args.hidden_dim),
        ctx_encoder=str(args.ctx_encoder),
        ctx_causal=True,
        ctx_local_kernel=int(args.ctx_local_kernel),
        ctx_pool_scales=tuple(int(x) for x in str(args.ctx_pool_scales).split(",") if x),
        use_time_features=True,
        use_time_gaps=False,
        rollout_mode="non_ar",
        future_block_len=int(spec.future_block_len),
        fu_net_type=str(args.fu_net_type),
        fu_net_layers=int(args.fu_net_layers),
        fu_net_heads=int(args.fu_net_heads),
        use_minibatch_ot=True,
        solver="euler",
        use_amp=bool(args.use_amp),
        grad_accum_steps=int(args.grad_accum_steps),
    )
    return cfg


def build_conditional_cfg(args: argparse.Namespace) -> OTFlowConfig:
    spec = _conditional_spec(str(args.dataset))
    plan = DATASET_PLANS[str(args.dataset)]
    preset = get_otflow_paper_backbone_preset(str(args.dataset))
    cfg = OTFlowConfig()
    cfg.apply_overrides(
        device=resolve_torch_device(str(args.device)),
        levels=int(preset["levels"]),
        token_dim=int(preset.get("token_dim", 4)),
        history_len=int(spec.history_len),
        steps=int(args.steps),
        batch_size=(
            int(getattr(args, "batch_size", 0))
            if int(getattr(args, "batch_size", 0) or 0) > 0
            else int(plan.batch_size)
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        standardize=True,
        use_cond_features=bool(preset.get("use_cond_features", False)),
        cond_standardize=bool(preset.get("cond_standardize", True)),
        hidden_dim=int(args.hidden_dim),
        ctx_encoder=str(preset["ctx_encoder"]),
        ctx_causal=bool(preset["ctx_causal"]),
        ctx_local_kernel=int(preset["ctx_local_kernel"]),
        ctx_pool_scales=tuple(int(x) for x in str(preset["ctx_pool_scales"]).split(",") if x),
        use_time_features=bool(preset.get("use_time_features", preset.get("use_time_gaps", False))),
        use_time_gaps=bool(preset.get("use_time_gaps", False)),
        rollout_mode="non_ar",
        future_block_len=int(spec.future_block_len),
        fu_net_type=DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE,
        fu_net_layers=int(args.fu_net_layers),
        fu_net_heads=int(args.fu_net_heads),
        use_minibatch_ot=True,
        solver="euler",
        use_amp=bool(args.use_amp),
        grad_accum_steps=int(args.grad_accum_steps),
    )
    return cfg


def ensure_forecast_dataset(dataset_root: Path, dataset: str, *, prepare: bool) -> None:
    manifest = default_manifest_path(dataset_root, dataset)
    if manifest.exists():
        return
    if not prepare:
        raise FileNotFoundError(
            f"Missing Monash manifest for {dataset}: {manifest}. Re-run with --prepare_data."
        )
    download_monash_dataset(dataset_root, dataset)


def _active_backbone_budgets(dataset: str, benchmark_family: str) -> Tuple[int, ...]:
    if str(benchmark_family) == FORECAST_FAMILY:
        return tuple(ACTIVE_FORECAST_BACKBONE_BUDGETS.get(str(dataset), ()))
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        return tuple(ACTIVE_CONDITIONAL_GENERATION_BACKBONE_BUDGETS.get(str(dataset), ()))
    return ()


def _parse_checkpoint_steps(raw: str | Sequence[int] | None, *, dataset: str, benchmark_family: str, max_steps: int) -> Tuple[int, ...]:
    if raw is None or str(raw).strip() == "":
        planned = _active_backbone_budgets(str(dataset), str(benchmark_family))
        steps = [int(value) for value in planned if int(value) <= int(max_steps)]
        if not steps:
            steps = [int(max_steps)]
        return tuple(sorted(set(steps)))
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(value) for value in raw]
    invalid = [value for value in values if value <= 0 or value > int(max_steps)]
    if invalid:
        raise ValueError(f"Checkpoint steps must be in [1, {int(max_steps)}], got {invalid}.")
    return tuple(sorted(set(values)))


def _clone_state_dict_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(dict(payload), tmp_path)
    tmp_path.replace(path)


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected training state at {path} to contain a dict payload.")
    return payload


def _normalize_checkpoint_export_mode(raw: Any) -> str:
    mode = str(raw or CHECKPOINT_EXPORT_MODE_BEST_VALIDATION).strip()
    if mode not in CHECKPOINT_EXPORT_MODES:
        raise ValueError(f"Unknown checkpoint export mode {mode!r}; expected one of {CHECKPOINT_EXPORT_MODES}.")
    return mode


def _temporal_training_signature(
    *,
    args: argparse.Namespace,
    benchmark_family: str,
    cfg: OTFlowConfig,
    checkpoint_steps: Sequence[int],
    split_stats: Mapping[str, Any],
    checkpoint_export_mode: str,
) -> Dict[str, Any]:
    return {
        "version": TEMPORAL_BACKBONE_TRAINING_STATE_VERSION,
        "dataset": str(args.dataset),
        "benchmark_family": str(benchmark_family),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "checkpoint_steps": [int(value) for value in checkpoint_steps],
        "checkpoint_export_mode": str(checkpoint_export_mode),
        "val_every": int(getattr(args, "val_every", 0) or 0),
        "val_max_batches": None
        if getattr(args, "val_max_batches", None) is None
        else int(getattr(args, "val_max_batches")),
        "cfg": cfg.to_dict(),
        "split_stats": _json_ready_stats(dict(split_stats)),
    }


def _default_training_state_path(args: argparse.Namespace, *, benchmark_family: str, signature_hash: str) -> Path:
    root = (
        project_backbone_matrix_root()
        / "_training_state"
        / BACKBONE_NAME_OTFLOW
        / str(benchmark_family)
        / str(args.dataset)
    )
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        root = root / DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
    return root / f"{signature_hash[:16]}.pt"


def _resolve_training_state_path(
    args: argparse.Namespace,
    *,
    benchmark_family: str,
    signature_hash: str,
) -> Path:
    explicit_out = str(getattr(args, "training_state_out", "") or "").strip()
    if explicit_out:
        return resolve_project_path(explicit_out)
    explicit_resume = str(getattr(args, "resume_training_state", "") or "").strip()
    if explicit_resume:
        return resolve_project_path(explicit_resume)
    return _default_training_state_path(args, benchmark_family=str(benchmark_family), signature_hash=str(signature_hash))


def _load_compatible_training_state(path: Path, *, signature_hash: str) -> Dict[str, Any]:
    payload = _torch_load(path)
    version = str(payload.get("version", ""))
    if version != TEMPORAL_BACKBONE_TRAINING_STATE_VERSION:
        raise ValueError(f"Training state {path} has version={version!r}, expected {TEMPORAL_BACKBONE_TRAINING_STATE_VERSION!r}.")
    state_hash = str(payload.get("signature_hash", ""))
    if state_hash != str(signature_hash):
        raise ValueError(f"Training state {path} does not match this run signature.")
    return payload


def _load_backbone_artifact_checkpoint(path: Path) -> Dict[str, Any]:
    payload = _torch_load(path)
    if not isinstance(payload.get("cfg"), Mapping):
        raise ValueError(f"Backbone checkpoint {path} is missing cfg metadata.")
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError(f"Backbone checkpoint {path} is missing model_state.")
    return payload


def _checkpoint_cfg_for_budget(cfg: OTFlowConfig, train_steps: int) -> OTFlowConfig:
    checkpoint_cfg = copy.deepcopy(cfg)
    checkpoint_cfg.apply_overrides(steps=int(train_steps))
    return checkpoint_cfg


def _save_backbone_artifact(
    *,
    artifact_root: Path,
    cfg: OTFlowConfig,
    state_dict: Mapping[str, torch.Tensor],
    dataset: str,
    spec: Any,
    seed: int,
    budget_steps: int,
    split_stats: Mapping[str, Any],
    selection: Mapping[str, Any],
    benchmark_family: str = FORECAST_FAMILY,
    field_network_type: Optional[str] = None,
    checkpoint_export_protocol: str = CHECKPOINT_EXPORT_PROTOCOL_BEST_VALIDATION,
) -> Dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoint_id = build_backbone_checkpoint_id(
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset),
        train_steps=int(budget_steps),
        seed=int(seed),
        field_network_type=field_network_type,
    )
    checkpoint_path = artifact_root / "model.pt"
    checkpoint_cfg = _checkpoint_cfg_for_budget(cfg, int(budget_steps))
    torch.save(
        {"cfg": checkpoint_cfg.to_dict(), "model_state": dict(state_dict)},
        checkpoint_path,
    )
    metadata = {
        "checkpoint_id": checkpoint_id,
        "dataset_key": str(dataset),
        "benchmark_family": str(benchmark_family),
        "backbone_name": BACKBONE_NAME_OTFLOW,
        "train_steps": int(budget_steps),
        "checkpoint_budget_steps": int(budget_steps),
        "effective_train_steps": int(selection.get("selected_step", budget_steps)),
        "checkpoint_export_protocol": str(checkpoint_export_protocol),
        "train_budget_label": train_budget_label(int(budget_steps)),
        "seed": int(seed),
        "history_len": int(spec.history_len),
        "future_block_len": int(spec.future_block_len),
        "rollout_mode": "non_ar",
        "cond_dim": 0,
        "checkpoint_path": _project_display_path(checkpoint_path),
        "metadata_path": _project_display_path(artifact_root / "checkpoint_metadata.json"),
        "summary_path": _project_display_path(artifact_root / "artifact_summary.json"),
        "split_stats": {**_json_ready_stats(dict(split_stats)), "cond_dim": 0},
        "cfg": checkpoint_cfg.to_dict(),
        "selection": dict(selection),
    }
    if field_network_type:
        metadata["field_network_type"] = str(field_network_type)
    save_json(metadata, str(artifact_root / "checkpoint_metadata.json"))
    save_json(metadata, str(artifact_root / "artifact_summary.json"))
    return metadata


def _data_path_for_conditional(args: argparse.Namespace, dataset: str) -> str:
    explicit = str(getattr(args, "data_path", "") or "").strip()
    if explicit:
        return explicit
    if str(dataset) == "cryptos":
        return str(getattr(args, "cryptos_path", "") or default_cryptos_data_path())
    if str(dataset) == "lobster_synthetic":
        return str(getattr(args, "lobster_synthetic_profile_path", "") or default_lobster_synthetic_profile_path())
    if str(dataset) == "long_term_st":
        return str(getattr(args, "long_term_st_path", "") or default_long_term_st_data_path())
    return ""


def _conditional_dataset_args(args: argparse.Namespace, cfg: OTFlowConfig) -> argparse.Namespace:
    plan = DATASET_PLANS[str(args.dataset)]
    return argparse.Namespace(
        dataset=str(args.dataset),
        data_path=_data_path_for_conditional(args, str(args.dataset)),
        synthetic_length=int(getattr(args, "synthetic_length", 0) or int(plan.synthetic_length)),
        seed=int(args.seed),
        device=str(args.device),
        steps=int(args.steps),
        train_frac=float(plan.train_frac),
        val_frac=float(plan.val_frac),
        test_frac=float(plan.test_frac),
        stride_train=int(getattr(args, "stride_train", plan.stride_train) or plan.stride_train),
        stride_eval=int(getattr(args, "stride_eval", plan.stride_eval) or plan.stride_eval),
        levels=int(cfg.levels),
        token_dim=int(getattr(cfg, "token_dim", 4)),
        history_len=int(cfg.history_len),
        batch_size=int(cfg.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
    )


def _train_temporal_backbone(args: argparse.Namespace, *, benchmark_family: str, spec: Any, cfg: OTFlowConfig, splits: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoint_steps = _parse_checkpoint_steps(
        getattr(args, "checkpoint_steps", None),
        dataset=str(args.dataset),
        benchmark_family=str(benchmark_family),
        max_steps=int(args.steps),
    )
    checkpoint_step_set = set(int(value) for value in checkpoint_steps)
    checkpoint_export_mode = _normalize_checkpoint_export_mode(getattr(args, "checkpoint_export_mode", ""))
    exact_budget_export = checkpoint_export_mode == CHECKPOINT_EXPORT_MODE_EXACT_BUDGET
    expected_protocol = (
        CHECKPOINT_EXPORT_PROTOCOL_EXACT_BUDGET if exact_budget_export else CHECKPOINT_EXPORT_PROTOCOL_BEST_VALIDATION
    )
    signature = _temporal_training_signature(
        args=args,
        benchmark_family=str(benchmark_family),
        cfg=cfg,
        checkpoint_steps=checkpoint_steps,
        split_stats=dict(splits.get("stats", {})),
        checkpoint_export_mode=checkpoint_export_mode,
    )
    signature_hash = _json_hash(signature)
    training_state_path = _resolve_training_state_path(
        args,
        benchmark_family=str(benchmark_family),
        signature_hash=signature_hash,
    )
    explicit_resume_path = str(getattr(args, "resume_training_state", "") or "").strip()
    resume_state_path = resolve_project_path(explicit_resume_path) if explicit_resume_path else training_state_path
    auto_resume = not bool(getattr(args, "no_auto_resume_training_state", False))
    resume_training_state = bool(explicit_resume_path) or (auto_resume and resume_state_path.exists())
    training_state_payload = (
        _load_compatible_training_state(resume_state_path, signature_hash=signature_hash)
        if resume_training_state
        else None
    )

    val_every = int(getattr(args, "val_every", 0) or 0)
    val_max_batches = getattr(args, "val_max_batches", None)
    if val_max_batches is not None:
        val_max_batches = int(val_max_batches)

    best: Dict[str, Any] = {
        "score": None,
        "state_dict": None,
        "step": 0,
        "metric_source": None,
        "validation": None,
        "train_loss": None,
        "error": None,
        "validation_available": False,
    }
    exported: List[Dict[str, Any]] = []
    exported_budget_steps: set[int] = set()

    start_step = 0
    initial_model_state = None
    optimizer_state = None
    scheduler_state = None
    scaler_state = None
    ema_state = None
    swa_model_state = None
    rng_state = None
    loader_state = None
    if training_state_payload is not None:
        start_step = int(training_state_payload.get("global_step", 0) or 0)
        initial_model_state = training_state_payload.get("model_state")
        optimizer_state = training_state_payload.get("optimizer_state")
        scheduler_state = training_state_payload.get("scheduler_state")
        scaler_state = training_state_payload.get("scaler_state")
        ema_state = training_state_payload.get("ema_state")
        swa_model_state = training_state_payload.get("swa_model_state")
        rng_state = training_state_payload.get("rng_state")
        loader_state = training_state_payload.get("loader_state")
        selector_state = dict(training_state_payload.get("selector_state", {}) or {})
        restored_best = selector_state.get("best")
        if isinstance(restored_best, dict):
            best.update(restored_best)
        restored_exported = selector_state.get("exported")
        if isinstance(restored_exported, list):
            exported.extend(dict(item) for item in restored_exported if isinstance(item, Mapping))
        exported_budget_steps.update(int(value) for value in selector_state.get("exported_budget_steps", []) or [])

    field_network_type = (
        DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
        if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY
        else None
    )

    def _artifact_root_for_budget(step: int) -> Path:
        return expected_artifact_root(
            project_backbone_matrix_root(),
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=str(benchmark_family),
            dataset_key=str(args.dataset),
            train_steps=int(step),
        )

    def _make_selection(
        *,
        step: int,
        train_loss: float,
        validation: Optional[Mapping[str, Any]],
        error: Optional[BaseException],
    ) -> Dict[str, Any]:
        if validation is not None:
            metric_source = "validation_loss"
            score = float(validation["loss"])
        else:
            metric_source = "train_loss_fallback"
            score = float(train_loss)
        return {
            "selection_metric": metric_source,
            "selection_score": float(score),
            "selected_step": int(step),
            "export_step": int(step),
            "validation": None if validation is None else dict(validation),
            "train_loss_at_selected_step": float(train_loss),
            "fallback_error": None if error is None else f"{type(error).__name__}: {error}",
        }

    def _record_candidate(
        *,
        step: int,
        model: torch.nn.Module,
        train_loss: float,
        validation: Optional[Mapping[str, Any]],
        error: Optional[BaseException],
    ) -> None:
        if validation is not None:
            score = float(validation["loss"])
            metric_source = "validation_loss"
            best["validation_available"] = True
        else:
            if bool(best.get("validation_available")):
                return
            score = float(train_loss)
            metric_source = "train_loss_fallback"
        current_metric = str(best.get("metric_source") or "")
        should_update = best["score"] is None
        if not should_update and metric_source == "validation_loss" and current_metric != "validation_loss":
            should_update = True
        elif not should_update and metric_source == current_metric and score < float(best["score"]):
            should_update = True
        if should_update:
            best.update(
                {
                    "score": float(score),
                    "state_dict": _clone_state_dict_cpu(model),
                    "step": int(step),
                    "metric_source": metric_source,
                    "validation": None if validation is None else dict(validation),
                    "train_loss": float(train_loss),
                    "error": None if error is None else f"{type(error).__name__}: {error}",
                }
            )

    def _export_budget(
        *,
        step: int,
        state_dict: Mapping[str, torch.Tensor],
        selection: Mapping[str, Any],
        checkpoint_export_protocol: str,
    ) -> None:
        artifact_root = _artifact_root_for_budget(int(step))
        metadata = _save_backbone_artifact(
            artifact_root=artifact_root,
            cfg=cfg,
            state_dict=state_dict,
            dataset=str(args.dataset),
            spec=spec,
            seed=int(args.seed),
            budget_steps=int(step),
            split_stats=dict(splits.get("stats", {})),
            selection=selection,
            benchmark_family=str(benchmark_family),
            field_network_type=field_network_type,
            checkpoint_export_protocol=str(checkpoint_export_protocol),
        )
        exported[:] = [item for item in exported if int(item.get("train_steps", -1)) != int(step)]
        exported.append(
            {
                "train_steps": int(step),
                "train_budget_label": train_budget_label(int(step)),
                "checkpoint_path": _project_display_path(artifact_root / "model.pt"),
                "metadata_path": _project_display_path(artifact_root / "checkpoint_metadata.json"),
                "checkpoint_id": str(metadata["checkpoint_id"]),
                "selected_step": int(metadata["effective_train_steps"]),
                "selection_metric": str(selection["selection_metric"]),
                "selection_score": float(selection["selection_score"]),
                "checkpoint_export_protocol": str(checkpoint_export_protocol),
            }
        )
        exported_budget_steps.add(int(step))

    def _save_restart_state(
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: torch.cuda.amp.GradScaler,
        loop_state: Mapping[str, Any],
        force: bool = False,
    ) -> None:
        save_every = int(getattr(args, "save_training_state_every", 200) or 0)
        if not force and (save_every <= 0 or int(step) % save_every != 0):
            return
        payload = {
            "version": TEMPORAL_BACKBONE_TRAINING_STATE_VERSION,
            "signature": signature,
            "signature_hash": signature_hash,
            "dataset": str(args.dataset),
            "benchmark_family": str(benchmark_family),
            "global_step": int(step),
            "checkpoint_export_mode": checkpoint_export_mode,
            "checkpoint_steps": [int(value) for value in checkpoint_steps],
            "model_state": _clone_state_dict_cpu(model),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "scheduler_state": None if scheduler is None else _cpu_tree(scheduler.state_dict()),
            "scaler_state": _cpu_tree(scaler.state_dict()),
            "ema_state": _cpu_tree(loop_state.get("ema_state")),
            "swa_model_state": _cpu_tree(loop_state.get("swa_model_state")),
            "rng_state": capture_rng_state(),
            "loader_state": dict(loop_state.get("loader_state", {}) or {}),
            "selector_state": {
                "best": _cpu_tree(best),
                "exported": list(exported),
                "exported_budget_steps": sorted(int(value) for value in exported_budget_steps),
            },
        }
        _atomic_torch_save(payload, training_state_path)

    def _on_step(step: int, model: torch.nn.Module, train_loss: float, logs: Dict[str, float]) -> None:
        del logs
        should_validate = int(step) in checkpoint_step_set or (val_every > 0 and int(step) % val_every == 0)
        validation = None
        error = None
        if should_validate:
            try:
                validation = evaluate_average_loss(
                    splits["val"],
                    model,
                    cfg,
                    model_name="otflow",
                    max_batches=val_max_batches,
                    shuffle=False,
                )
            except Exception as exc:
                error = exc
        if exact_budget_export:
            if int(step) in checkpoint_step_set:
                selection = _make_selection(
                    step=int(step),
                    train_loss=float(train_loss),
                    validation=validation,
                    error=error,
                )
                _export_budget(
                    step=int(step),
                    state_dict=_clone_state_dict_cpu(model),
                    selection=selection,
                    checkpoint_export_protocol=CHECKPOINT_EXPORT_PROTOCOL_EXACT_BUDGET,
                )
            return

        _record_candidate(
            step=int(step),
            model=model,
            train_loss=float(train_loss),
            validation=validation,
            error=error,
        )
        if int(step) in checkpoint_step_set:
            if best["state_dict"] is None:
                raise RuntimeError(f"No checkpoint candidate is available at step {int(step)}.")
            selection = {
                "selection_metric": str(best["metric_source"]),
                "selection_score": float(best["score"]),
                "selected_step": int(best["step"]),
                "export_step": int(step),
                "validation": best["validation"],
                "train_loss_at_selected_step": best["train_loss"],
                "fallback_error": best["error"],
            }
            _export_budget(
                step=int(step),
                state_dict=best["state_dict"],
                selection=selection,
                checkpoint_export_protocol=CHECKPOINT_EXPORT_PROTOCOL_BEST_VALIDATION,
            )

    def _on_training_state(
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        scaler: torch.cuda.amp.GradScaler,
        loop_state: Dict[str, Any],
    ) -> None:
        force = int(step) in checkpoint_step_set or int(step) >= int(args.steps)
        _save_restart_state(
            step=int(step),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loop_state=loop_state,
            force=force,
        )

    if training_state_payload is not None:
        print(f"Resuming temporal backbone from {_project_display_path(resume_state_path)} at step {int(start_step)}.")
    else:
        print(f"Writing temporal backbone restart state to {_project_display_path(training_state_path)}.")

    model = train_loop(
        splits["train"],
        cfg,
        model_name="otflow",
        steps=int(args.steps),
        log_every=int(args.log_every),
        on_step=_on_step,
        initial_model_state=initial_model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        ema_state=ema_state,
        swa_model_state=swa_model_state,
        rng_state=rng_state,
        loader_state=loader_state,
        start_step=int(start_step),
        on_training_state=_on_training_state,
    )
    del model

    missing_or_stale: List[int] = []
    for budget in checkpoint_steps:
        artifact_root = _artifact_root_for_budget(int(budget))
        metadata_path = artifact_root / "checkpoint_metadata.json"
        checkpoint_path = artifact_root / "model.pt"
        if not metadata_path.exists():
            missing_or_stale.append(int(budget))
            continue
        if not checkpoint_path.exists():
            missing_or_stale.append(int(budget))
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("checkpoint_export_protocol", "")) != expected_protocol:
            missing_or_stale.append(int(budget))
            continue
        if str(metadata.get("dataset_key", "")) != str(args.dataset):
            missing_or_stale.append(int(budget))
            continue
        if str(metadata.get("benchmark_family", "")) != str(benchmark_family):
            missing_or_stale.append(int(budget))
            continue
        if int(metadata.get("seed", -1)) != int(args.seed):
            missing_or_stale.append(int(budget))
            continue
        if int(metadata.get("checkpoint_budget_steps", -1)) != int(budget):
            missing_or_stale.append(int(budget))
            continue
        if exact_budget_export and int(metadata.get("effective_train_steps", -1)) != int(budget):
            missing_or_stale.append(int(budget))
            continue
        try:
            checkpoint = _load_backbone_artifact_checkpoint(checkpoint_path)
        except Exception:
            missing_or_stale.append(int(budget))
            continue
        checkpoint_cfg = dict(checkpoint["cfg"])
        checkpoint_train_cfg = dict(checkpoint_cfg.get("train", {}))
        metadata_cfg = dict(metadata.get("cfg", {}))
        metadata_train_cfg = dict(metadata_cfg.get("train", {}))
        if int(checkpoint_train_cfg.get("steps", -1)) != int(budget):
            missing_or_stale.append(int(budget))
            continue
        if int(metadata_train_cfg.get("steps", -1)) != int(budget):
            missing_or_stale.append(int(budget))
            continue
    if missing_or_stale:
        raise RuntimeError(
            "Temporal backbone training finished without valid budget artifacts for "
            f"{missing_or_stale}; expected protocol {expected_protocol!r}."
        )

    manifest = materialize_backbone_manifest(budget_steps=checkpoint_steps, seed=int(args.seed))
    final_artifact_root = _artifact_root_for_budget(int(checkpoint_steps[-1]))
    summary = {
        "status": "ready",
        "benchmark_family": str(benchmark_family),
        "checkpoint_path": _project_display_path(final_artifact_root / "model.pt"),
        "metadata_path": _project_display_path(final_artifact_root / "checkpoint_metadata.json"),
        "manifest_path": _project_display_path(project_backbone_matrix_root() / "backbone_manifest.json"),
        "manifest_ready_count": int(manifest.get("ready_count", 0)),
        "dataset": str(args.dataset),
        "train_steps": int(args.steps),
        "checkpoint_steps": [int(value) for value in checkpoint_steps],
        "exported": exported,
        "checkpoint_export_mode": checkpoint_export_mode,
        "checkpoint_export_protocol": expected_protocol,
        "training_state_path": _project_display_path(training_state_path),
        "resumed_from_training_state": training_state_payload is not None,
        "resume_start_step": int(start_step),
    }
    save_json(summary, str(final_artifact_root / "training_summary.json"))
    return summary


def train_backbone(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    dataset_root = resolve_project_path(str(args.dataset_root))
    spec = _dataset_spec(str(args.dataset))
    if str(spec.benchmark_family) == FORECAST_FAMILY:
        ensure_forecast_dataset(dataset_root, str(args.dataset), prepare=bool(args.prepare_data))
        cfg = build_forecast_cfg(args)
        splits = build_monash_forecast_splits(
            dataset_root=dataset_root,
            dataset_key=str(args.dataset),
            cfg=cfg,
            history_len=int(spec.history_len),
            horizon=int(spec.future_block_len),
            stride_train=int(args.stride_train),
            time_feature_mode="gap_elapsed",
        )
        return _train_temporal_backbone(args, benchmark_family=FORECAST_FAMILY, spec=spec, cfg=cfg, splits=splits)
    if str(spec.benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        cfg = build_conditional_cfg(args)
        splits = build_dataset_splits(_conditional_dataset_args(args, cfg), cfg)
        return _train_temporal_backbone(
            args,
            benchmark_family=CONDITIONAL_GENERATION_FAMILY,
            spec=spec,
            cfg=cfg,
            splits=splits,
        )
    raise ValueError(f"Unsupported temporal backbone family: {spec.benchmark_family!r}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train standalone genODE temporal OT flow-matching backbones.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--data_path", default="")
    parser.add_argument("--cryptos_path", default="")
    parser.add_argument("--lobster_synthetic_profile_path", default="")
    parser.add_argument("--long_term_st_path", default="")
    parser.add_argument("--synthetic_length", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size; 0 uses the canonical dataset default.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--ctx_encoder", default="hybrid")
    parser.add_argument("--ctx_local_kernel", type=int, default=7)
    parser.add_argument("--ctx_pool_scales", default="8,32")
    parser.add_argument("--fu_net_type", default="transformer", choices=("transformer", "mlp", "resmlp"))
    parser.add_argument("--fu_net_layers", type=int, default=3)
    parser.add_argument("--fu_net_heads", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride_train", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_max_batches", type=int, default=None)
    parser.add_argument(
        "--checkpoint_steps",
        default="",
        help="Comma-separated budget milestones to export. Defaults to active forecast backbone budgets <= --steps.",
    )
    parser.add_argument(
        "--checkpoint_export_mode",
        choices=CHECKPOINT_EXPORT_MODES,
        default=CHECKPOINT_EXPORT_MODE_BEST_VALIDATION,
        help=(
            "How budget checkpoints are exported. exact_budget saves the model at each requested training budget; "
            "best_validation_within_budget preserves the older cumulative validation-best behavior."
        ),
    )
    parser.add_argument(
        "--training_state_out",
        default="",
        help="Restart checkpoint path. Defaults to a signature-keyed path under the backbone matrix root.",
    )
    parser.add_argument(
        "--resume_training_state",
        default="",
        help="Explicit restart checkpoint path to resume from. Defaults to --training_state_out if it exists.",
    )
    parser.add_argument(
        "--save_training_state_every",
        type=int,
        default=200,
        help="Optimizer-step interval for restart checkpoints; budget steps and final step are always saved.",
    )
    parser.add_argument(
        "--no_auto_resume_training_state",
        action="store_true",
        help="Ignore the default restart checkpoint unless --resume_training_state is provided.",
    )
    parser.add_argument("--prepare_data", action="store_true", default=True)
    parser.add_argument("--no_prepare_data", dest="prepare_data", action="store_false")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    return parser


def main() -> None:
    summary = train_backbone(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
