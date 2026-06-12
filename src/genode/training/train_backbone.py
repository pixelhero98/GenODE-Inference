from __future__ import annotations

import argparse
import copy
import json
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
from genode.models.otflow_train_val import evaluate_average_loss, save_json, seed_all, train_loop
from genode.runtime import resolve_torch_device

DEFAULT_DATASET = "traffic_hourly"
DEFAULT_HIDDEN_DIM = 160


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
        "checkpoint_export_protocol": "best_validation_state_within_budget",
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
        if best["score"] is None or score < float(best["score"]):
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

    def _export_budget(step: int) -> None:
        if best["state_dict"] is None:
            raise RuntimeError(f"No checkpoint candidate is available at step {int(step)}.")
        field_network_type = (
            DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
            if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY
            else None
        )
        artifact_root = expected_artifact_root(
            project_backbone_matrix_root(),
            backbone_name=BACKBONE_NAME_OTFLOW,
            benchmark_family=str(benchmark_family),
            dataset_key=str(args.dataset),
            train_steps=int(step),
        )
        selection = {
            "selection_metric": str(best["metric_source"]),
            "selection_score": float(best["score"]),
            "selected_step": int(best["step"]),
            "export_step": int(step),
            "validation": best["validation"],
            "train_loss_at_selected_step": best["train_loss"],
            "fallback_error": best["error"],
        }
        metadata = _save_backbone_artifact(
            artifact_root=artifact_root,
            cfg=cfg,
            state_dict=best["state_dict"],
            dataset=str(args.dataset),
            spec=spec,
            seed=int(args.seed),
            budget_steps=int(step),
            split_stats=dict(splits.get("stats", {})),
            selection=selection,
            benchmark_family=str(benchmark_family),
            field_network_type=field_network_type,
        )
        exported.append(
            {
                "train_steps": int(step),
                "train_budget_label": train_budget_label(int(step)),
                "checkpoint_path": _project_display_path(artifact_root / "model.pt"),
                "metadata_path": _project_display_path(artifact_root / "checkpoint_metadata.json"),
                "checkpoint_id": str(metadata["checkpoint_id"]),
                "selected_step": int(best["step"]),
                "selection_metric": str(best["metric_source"]),
                "selection_score": float(best["score"]),
            }
        )

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
        _record_candidate(
            step=int(step),
            model=model,
            train_loss=float(train_loss),
            validation=validation,
            error=error,
        )
        if int(step) in checkpoint_step_set:
            _export_budget(int(step))

    model = train_loop(
        splits["train"],
        cfg,
        model_name="otflow",
        steps=int(args.steps),
        log_every=int(args.log_every),
        on_step=_on_step,
    )
    del model
    manifest = materialize_backbone_manifest(budget_steps=checkpoint_steps, seed=int(args.seed))
    final_artifact_root = expected_artifact_root(
        project_backbone_matrix_root(),
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(args.dataset),
        train_steps=int(checkpoint_steps[-1]),
    )
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
