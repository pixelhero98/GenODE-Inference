from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

from genode.data.molecule_xyz import (
    DEFAULT_MOLECULE_DATASET_KEY,
    MOLECULE_BENCHMARK_FAMILY,
    MoleculeWindowDataset,
    build_molecule_dataset_splits,
    configure_molecule_otflow,
    default_molecule_processed_dir,
    ensure_molecule_processed,
)
from genode.data.otflow_paths import project_outputs_root, project_root, resolve_project_path
from genode.evaluation.fm_backbone_registry import materialize_backbone_manifest
from genode.models.config import OTFlowConfig
from genode.models.otflow_train_val import evaluate_average_loss, save_json, seed_all, train_loop
from genode.runtime import resolve_torch_device
from genode.training.train_backbone import _clone_state_dict_cpu, _json_ready_stats


DEFAULT_HISTORY_LEN = 16
DEFAULT_STEPS = 20_000
DEFAULT_VAL_EVERY = 200
DEFAULT_BUDGET_STEPS = (4_000, 8_000, 12_000, 16_000, 20_000)


def molecule_artifact_root(
    *,
    out_dir: str | Path | None,
    variant: str,
    train_steps: int,
    dataset_key: str = DEFAULT_MOLECULE_DATASET_KEY,
    member_key: str = "",
    stratum: str = "",
) -> Path:
    root = project_outputs_root() / "molecule_3d_backbones" if out_dir is None else resolve_project_path(out_dir)
    parts = [root, Path(str(dataset_key))]
    if str(member_key):
        parts.append(Path(str(member_key)))
    if str(stratum):
        parts.append(Path(str(stratum)))
    parts.extend([Path(str(variant)), Path(f"{int(train_steps)}_steps")])
    out = parts[0]
    for part in parts[1:]:
        out = out / part
    return out


def _backbone_manifest_write_path(args: argparse.Namespace, molecule_backbone_root: Path) -> Path:
    raw = str(getattr(args, "backbone_manifest", "") or "").strip()
    if raw:
        return resolve_project_path(raw)
    resolved_root = Path(molecule_backbone_root).expanduser().resolve()
    manifest_base = resolved_root.parent if resolved_root.name == "molecule_3d_backbones" else resolved_root
    return manifest_base / "backbone_matrix" / "backbone_manifest.json"


def _project_display_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return resolved.name


def _checkpoint_cfg_for_budget(cfg: OTFlowConfig, train_steps: int) -> OTFlowConfig:
    checkpoint_cfg = copy.deepcopy(cfg)
    checkpoint_cfg.apply_overrides(steps=int(train_steps))
    return checkpoint_cfg


def build_molecule_cfg(args: argparse.Namespace, *, atom_count: int, context_feature_dim: int) -> OTFlowConfig:
    horizon = int(args.future_horizon)
    if horizon != 1:
        raise ValueError("Molecule 3D backbone training is AR-only for this phase; use --future_horizon 1.")
    cfg = configure_molecule_otflow(
        OTFlowConfig(),
        history_len=int(args.history_len),
        future_horizon=horizon,
        rollout_mode="autoregressive",
        atom_count=int(atom_count),
        context_feature_dim=int(context_feature_dim),
    )
    cfg.apply_overrides(
        device=resolve_torch_device(str(args.device)),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        standardize=True,
        hidden_dim=int(args.hidden_dim),
        ctx_encoder=str(args.ctx_encoder),
        ctx_causal=True,
        ctx_local_kernel=int(args.ctx_local_kernel),
        ctx_pool_scales=tuple(int(x) for x in str(args.ctx_pool_scales).split(",") if x),
        train_variable_context=bool(args.train_variable_context),
        train_context_min=int(args.train_context_min),
        train_context_max=int(args.train_context_max),
        fu_net_type=str(args.fu_net_type),
        fu_net_layers=int(args.fu_net_layers),
        fu_net_heads=int(args.fu_net_heads),
        use_minibatch_ot=bool(args.use_minibatch_ot),
        solver=str(args.solver),
        use_amp=bool(args.use_amp),
        grad_accum_steps=int(args.grad_accum_steps),
        ema_decay=float(args.ema_decay),
        use_swa=bool(args.use_swa),
    )
    return cfg


def _parse_budget_steps(raw: str, *, train_steps: int) -> List[int]:
    values = [int(part) for part in str(raw).split(",") if part.strip()]
    if not values:
        values = [int(train_steps)]
    values.append(int(train_steps))
    budgets = sorted({value for value in values if 0 < int(value) <= int(train_steps)})
    if not budgets:
        raise ValueError("At least one positive --budget_steps value must be <= --steps.")
    return budgets


def _evaluate_iso_balanced_loss(
    ds: MoleculeWindowDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    *,
    max_batches: Optional[int],
) -> Dict[str, Any]:
    if len(ds) <= 0:
        raise ValueError("Cannot evaluate Iso-balanced loss on an empty molecule dataset.")
    group_values = np.unique(ds.data.trajectory_ids[ds.start_indices])
    per_iso: Dict[str, Any] = {}
    losses: List[float] = []
    total_examples = 0
    weighted_loss = 0.0
    for group_id in group_values:
        starts = ds.start_indices[ds.data.trajectory_ids[ds.start_indices] == int(group_id)]
        if len(starts) == 0:
            continue
        key = str(ds.data.trajectory_keys[int(group_id)]) if int(group_id) < len(ds.data.trajectory_keys) else str(int(group_id))
        sub_ds = MoleculeWindowDataset(
            ds.data,
            split=f"{ds.split}_trajectory_{int(group_id)}",
            start_indices=starts,
            history_len=ds.H,
            future_horizon=ds.future_horizon,
            stats=ds.stats,
        )
        result = evaluate_average_loss(
            sub_ds,
            model,
            cfg,
            model_name="otflow",
            max_batches=max_batches,
            shuffle=False,
        )
        loss = float(result["loss"])
        examples = int(result["examples"])
        per_iso[key] = result
        losses.append(loss)
        total_examples += examples
        weighted_loss += loss * float(examples)
    if not losses:
        raise ValueError("Iso-balanced evaluation produced no Iso groups.")
    return {
        "loss": float(np.mean(np.asarray(losses, dtype=np.float64))),
        "window_weighted_loss": float(weighted_loss / float(max(1, total_examples))),
        "examples": int(total_examples),
        "iso_count": int(len(losses)),
        "per_iso": per_iso,
    }


def _save_molecule_artifact(
    *,
    artifact_root: Path,
    cfg: OTFlowConfig,
    state_dict: Mapping[str, torch.Tensor],
    variant: str,
    seed: int,
    budget_steps: int,
    split_stats: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_root / "model.pt"
    checkpoint_cfg = _checkpoint_cfg_for_budget(cfg, int(budget_steps))
    dataset_key = str(split_stats.get("dataset_key", DEFAULT_MOLECULE_DATASET_KEY))
    stratum = str(split_stats.get("stratum", ""))
    member_key = str(split_stats.get("member_key", ""))
    source_zip_name = str(split_stats.get("source_zip_name", ""))
    atom_count = int(split_stats.get("atom_count", 0) or 0)
    formula = str(split_stats.get("formula", ""))
    snapshot_dim = int(split_stats["snapshot_dim"])
    context_feature_dim = int(split_stats["context_feature_dim"])
    checkpoint_id = (
        f"otflow_{MOLECULE_BENCHMARK_FAMILY}_{dataset_key}_{member_key or stratum}_"
        f"{variant}_{int(budget_steps)}_steps_seed{int(seed)}"
    )
    torch.save(
        {
            "cfg": checkpoint_cfg.to_dict(),
            "model_state": dict(state_dict),
            "molecule_stats": dict(split_stats.get("stats", {})),
            "dataset_key": dataset_key,
            "stratum": stratum,
        },
        checkpoint_path,
    )
    metadata: Dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "dataset_key": dataset_key,
        "stratum": stratum,
        "member_key": member_key,
        "benchmark_family": MOLECULE_BENCHMARK_FAMILY,
        "backbone_name": "otflow_molecule_3d",
        "variant": str(variant),
        "train_steps": int(budget_steps),
        "seed": int(seed),
        "history_len": int(cfg.history_len),
        "future_block_len": int(cfg.prediction_horizon),
        "rollout_mode": str(cfg.model.rollout_mode),
        "atom_count": atom_count,
        "formula": formula,
        "source_zip_name": source_zip_name,
        "snapshot_dim": snapshot_dim,
        "context_feature_dim": context_feature_dim,
        "checkpoint_path": _project_display_path(checkpoint_path),
        "metadata_path": _project_display_path(artifact_root / "checkpoint_metadata.json"),
        "summary_path": _project_display_path(artifact_root / "artifact_summary.json"),
        "split_stats": _json_ready_stats(dict(split_stats)),
        "cfg": checkpoint_cfg.to_dict(),
        "selection": dict(selection),
    }
    save_json(metadata, str(artifact_root / "checkpoint_metadata.json"))
    save_json(metadata, str(artifact_root / "artifact_summary.json"))
    return metadata


def train_molecule_backbone(args: argparse.Namespace) -> Dict[str, Any]:
    seed_all(int(args.seed))
    dataset_key = str(getattr(args, "dataset_key", DEFAULT_MOLECULE_DATASET_KEY) or DEFAULT_MOLECULE_DATASET_KEY)
    stratum = str(getattr(args, "stratum", "") or "")
    processed_dir = (
        default_molecule_processed_dir(dataset_key, stratum)
        if args.processed_dir is None
        else resolve_project_path(str(args.processed_dir))
    )
    zip_path = None if args.zip_path is None else resolve_project_path(str(args.zip_path))
    metadata = ensure_molecule_processed(
        zip_path=zip_path,
        processed_dir=processed_dir,
        prepare=bool(args.prepare_data),
        dataset_key=dataset_key,
        stratum=stratum,
    )
    if not bool(metadata.get("trainable", True)) and not bool(getattr(args, "allow_non_trainable", False)):
        raise ValueError(f"{metadata.get('dataset_key')}/{metadata.get('stratum')} is marked preprocess-only and is not queued for training.")
    cfg = build_molecule_cfg(
        args,
        atom_count=int(metadata["atom_count"]),
        context_feature_dim=int(metadata["context_feature_dim"]),
    )
    splits = build_molecule_dataset_splits(
        processed_dir=processed_dir,
        cfg=cfg,
        history_len=int(args.history_len),
        future_horizon=int(args.future_horizon),
        stride_train=int(args.stride_train),
        stride_eval=int(args.stride_eval),
        dataset_key=dataset_key,
        stratum=stratum,
    )
    val_every = int(args.val_every)
    if val_every <= 0:
        raise ValueError("Molecule backbone training requires --val_every > 0 for validation checkpoint selection.")
    val_max_batches = None if args.val_max_batches is None else int(args.val_max_batches)
    budget_steps = _parse_budget_steps(str(args.budget_steps), train_steps=int(args.steps))

    best: Dict[str, Any] = {
        "score": None,
        "state_dict": None,
        "step": 0,
        "validation": None,
        "train_loss": None,
    }
    budget_best: Dict[int, Dict[str, Any]] = {}

    def _record_validation_candidate(
        *,
        step: int,
        model: torch.nn.Module,
        train_loss: float,
        validation: Mapping[str, Any],
    ) -> None:
        score = float(validation["loss"])
        if best["score"] is None or score < float(best["score"]):
            best.update(
                {
                    "score": score,
                    "state_dict": _clone_state_dict_cpu(model),
                    "step": int(step),
                    "validation": dict(validation),
                    "train_loss": float(train_loss),
                }
            )

    def _capture_budget_candidates(*, max_step: int) -> None:
        if best["state_dict"] is None:
            return
        for budget in budget_steps:
            if int(budget) > int(max_step) or int(budget) in budget_best:
                continue
            budget_best[int(budget)] = {
                "score": float(best["score"]),
                "state_dict": dict(best["state_dict"]),
                "step": int(best["step"]),
                "validation": dict(best["validation"]),
                "train_loss": float(best["train_loss"]),
            }

    def _on_step(step: int, model: torch.nn.Module, train_loss: float, logs: Dict[str, float]) -> None:
        del logs
        should_validate = (int(step) % val_every == 0) or int(step) == int(args.steps)
        if not should_validate:
            return
        _capture_budget_candidates(max_step=int(step) - 1)
        clean_validation = evaluate_average_loss(
            splits["val_clean"],
            model,
            cfg,
            model_name="otflow",
            max_batches=val_max_batches,
            shuffle=False,
        )
        iso_balanced_validation = _evaluate_iso_balanced_loss(
            splits["val_clean"],
            model,
            cfg,
            max_batches=val_max_batches,
        )
        validation = {
            "loss": float(clean_validation["loss"]),
            "selection_scope": "val_clean_window_balanced",
            "clean_window_loss": clean_validation,
            "clean_iso_balanced_loss": iso_balanced_validation,
        }
        _record_validation_candidate(
            step=int(step),
            model=model,
            train_loss=float(train_loss),
            validation=validation,
        )
        _capture_budget_candidates(max_step=int(step))

    model = train_loop(
        splits["train"],
        cfg,
        model_name="otflow",
        steps=int(args.steps),
        log_every=int(args.log_every),
        on_step=_on_step,
    )
    del model
    _capture_budget_candidates(max_step=int(args.steps))
    if best["state_dict"] is None:
        raise RuntimeError("No validation-selected molecule checkpoint is available.")
    missing_budgets = [budget for budget in budget_steps if budget not in budget_best]
    if missing_budgets:
        raise RuntimeError(f"No validation-selected molecule checkpoints are available for budgets {missing_budgets}.")

    variant = str(args.variant or "ar_h1")
    member_key = str(splits["stats"].get("member_key", ""))
    budget_artifacts: Dict[str, Any] = {}
    final_budget = int(max(budget_steps))
    for budget in budget_steps:
        candidate = budget_best[int(budget)]
        artifact_root = molecule_artifact_root(
            out_dir=args.out_dir,
            variant=variant,
            train_steps=int(budget),
            dataset_key=dataset_key,
            member_key=member_key,
            stratum=stratum,
        )
        selection = {
            "selection_metric": "clean_validation_loss",
            "selection_score": float(candidate["score"]),
            "selected_step": int(candidate["step"]),
            "export_step": int(budget),
            "validation": candidate["validation"],
            "train_loss_at_selected_step": candidate["train_loss"],
        }
        metadata = _save_molecule_artifact(
            artifact_root=artifact_root,
            cfg=cfg,
            state_dict=candidate["state_dict"],
            variant=variant,
            seed=int(args.seed),
            budget_steps=int(budget),
            split_stats=dict(splits.get("stats", {})),
            selection=selection,
        )
        budget_summary = {
            "status": "ready",
            "dataset": dataset_key,
            "member_key": member_key,
            "stratum": stratum,
            "variant": variant,
            "train_steps": int(args.steps),
            "export_step": int(budget),
            "checkpoint_path": _project_display_path(artifact_root / "model.pt"),
            "metadata_path": _project_display_path(artifact_root / "checkpoint_metadata.json"),
            "selected_step": int(candidate["step"]),
            "selection_metric": "clean_validation_loss",
            "selection_score": float(candidate["score"]),
            "validation": candidate["validation"],
            "split_examples": {
                "train": int(len(splits["train"])),
                "val": int(len(splits["val"])),
                "val_clean": int(len(splits["val_clean"])),
                "test": int(len(splits["test"])),
                "test_clean": int(len(splits["test_clean"])),
            },
            "metadata": metadata,
        }
        save_json(budget_summary, str(artifact_root / "training_summary.json"))
        budget_artifacts[str(int(budget))] = {
            "checkpoint_path": _project_display_path(artifact_root / "model.pt"),
            "metadata_path": _project_display_path(artifact_root / "checkpoint_metadata.json"),
            "summary_path": _project_display_path(artifact_root / "training_summary.json"),
            "selected_step": int(candidate["step"]),
            "selection_score": float(candidate["score"]),
        }

    final_candidate = budget_best[final_budget]
    final_artifact_root = molecule_artifact_root(
        out_dir=args.out_dir,
        variant=variant,
        train_steps=final_budget,
        dataset_key=dataset_key,
        member_key=member_key,
        stratum=stratum,
    )
    molecule_backbone_root = project_outputs_root() / "molecule_3d_backbones" if args.out_dir is None else resolve_project_path(str(args.out_dir))
    manifest_path = _backbone_manifest_write_path(args, molecule_backbone_root)
    manifest = materialize_backbone_manifest(
        matrix_root=manifest_path.parent,
        budget_steps=budget_steps,
        seed=int(args.seed),
        molecule_backbone_root=molecule_backbone_root,
        molecule_group_root=None if getattr(args, "molecule_group_root", None) is None else resolve_project_path(str(args.molecule_group_root)),
        write_path=manifest_path,
    )
    summary = {
        "status": "ready",
        "dataset": dataset_key,
        "member_key": member_key,
        "stratum": stratum,
        "variant": variant,
        "train_steps": int(args.steps),
        "checkpoint_path": _project_display_path(final_artifact_root / "model.pt"),
        "metadata_path": _project_display_path(final_artifact_root / "checkpoint_metadata.json"),
        "selected_step": int(final_candidate["step"]),
        "selection_metric": "clean_validation_loss",
        "selection_score": float(final_candidate["score"]),
        "validation": final_candidate["validation"],
        "budget_artifacts": budget_artifacts,
        "manifest_path": _project_display_path(manifest_path),
        "manifest_ready_count": int(manifest.get("ready_count", 0)),
        "split_examples": {
            "train": int(len(splits["train"])),
            "val": int(len(splits["val"])),
            "val_clean": int(len(splits["val_clean"])),
            "test": int(len(splits["test"])),
            "test_clean": int(len(splits["test_clean"])),
        },
    }
    save_json(summary, str(final_artifact_root / "training_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train molecule 3D coordinate OTFlow backbones.")
    parser.add_argument("--dataset_key", default=DEFAULT_MOLECULE_DATASET_KEY)
    parser.add_argument("--stratum", default="")
    parser.add_argument("--zip_path", default=None)
    parser.add_argument("--processed_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--backbone_manifest", default="")
    parser.add_argument("--molecule_group_root", default=None)
    parser.add_argument("--variant", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--ctx_encoder", default="hybrid")
    parser.add_argument("--ctx_local_kernel", type=int, default=7)
    parser.add_argument("--ctx_pool_scales", default="8")
    parser.add_argument("--fu_net_type", default="transformer", choices=("transformer", "mlp", "resmlp"))
    parser.add_argument("--fu_net_layers", type=int, default=3)
    parser.add_argument("--fu_net_heads", type=int, default=4)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN)
    parser.add_argument("--future_horizon", type=int, default=1)
    parser.add_argument("--train_variable_context", action="store_true", default=True)
    parser.add_argument("--fixed_context", dest="train_variable_context", action="store_false")
    parser.add_argument("--train_context_min", type=int, default=8)
    parser.add_argument("--train_context_max", type=int, default=16)
    parser.add_argument("--stride_train", type=int, default=1)
    parser.add_argument("--stride_eval", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--val_every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--val_max_batches", type=int, default=None)
    parser.add_argument("--budget_steps", default=",".join(str(value) for value in DEFAULT_BUDGET_STEPS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--solver", default="euler")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_swa", action="store_true", default=False)
    parser.add_argument("--use_minibatch_ot", action="store_true", default=True)
    parser.add_argument("--no_minibatch_ot", dest="use_minibatch_ot", action="store_false")
    parser.add_argument("--prepare_data", action="store_true", default=True)
    parser.add_argument("--no_prepare_data", dest="prepare_data", action="store_false")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--allow_non_trainable", action="store_true", default=False)
    return parser


def main() -> None:
    summary = train_molecule_backbone(build_argparser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
