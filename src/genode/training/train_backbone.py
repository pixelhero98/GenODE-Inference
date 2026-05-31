from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from genode.data.otflow_experiment_plan import FORECAST_FAMILY, experiment_plan_by_key
from genode.data.otflow_forecast_data import build_monash_forecast_splits
from genode.data.otflow_monash_datasets import default_manifest_path, download_monash_dataset
from genode.data.otflow_paths import project_paper_dataset_root, resolve_project_path
from genode.evaluation.fm_backbone_registry import (
    BACKBONE_NAME_OTFLOW,
    build_backbone_checkpoint_id,
    expected_artifact_root,
    materialize_backbone_manifest,
    project_backbone_matrix_root,
    train_budget_label,
)
from genode.models.config import OTFlowConfig
from genode.models.otflow_train_val import save_json, seed_all, train_loop
from genode.runtime import resolve_torch_device

DEFAULT_DATASET = "san_francisco_traffic"
DEFAULT_HIDDEN_DIM = 160


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


def build_sf_traffic_cfg(args: argparse.Namespace) -> OTFlowConfig:
    spec = experiment_plan_by_key()[str(args.dataset)]
    cfg = OTFlowConfig()
    cfg.apply_overrides(
        device=resolve_torch_device(str(args.device)),
        levels=1,
        token_dim=1,
        history_len=int(spec.history_len),
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


def ensure_forecast_dataset(dataset_root: Path, dataset: str, *, prepare: bool) -> None:
    manifest = default_manifest_path(dataset_root, dataset)
    if manifest.exists():
        return
    if not prepare:
        raise FileNotFoundError(
            f"Missing Monash manifest for {dataset}: {manifest}. Re-run with --prepare_data."
        )
    download_monash_dataset(dataset_root, dataset)


def train_backbone(args: argparse.Namespace) -> Dict[str, Any]:
    if str(args.dataset) != DEFAULT_DATASET:
        raise ValueError("genODE v1 backbone training is scoped to san_francisco_traffic.")
    seed_all(int(args.seed))
    dataset_root = resolve_project_path(str(args.dataset_root))
    ensure_forecast_dataset(dataset_root, str(args.dataset), prepare=bool(args.prepare_data))
    cfg = build_sf_traffic_cfg(args)
    spec = experiment_plan_by_key()[str(args.dataset)]
    splits = build_monash_forecast_splits(
        dataset_root=dataset_root,
        dataset_key=str(args.dataset),
        cfg=cfg,
        history_len=int(spec.history_len),
        horizon=int(spec.future_block_len),
        stride_train=int(args.stride_train),
        time_feature_mode="gap_elapsed",
    )
    model = train_loop(
        splits["train"],
        cfg,
        model_name="otflow",
        steps=int(args.steps),
        log_every=int(args.log_every),
    )
    artifact_root = expected_artifact_root(
        project_backbone_matrix_root(),
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=FORECAST_FAMILY,
        dataset_key=str(args.dataset),
        train_steps=int(args.steps),
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    checkpoint_id = build_backbone_checkpoint_id(
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=FORECAST_FAMILY,
        dataset_key=str(args.dataset),
        train_steps=int(args.steps),
        seed=int(args.seed),
    )
    checkpoint_path = artifact_root / "model.pt"
    torch.save({"cfg": cfg.to_dict(), "model_state": model.state_dict()}, checkpoint_path)
    metadata = {
        "checkpoint_id": checkpoint_id,
        "dataset_key": str(args.dataset),
        "benchmark_family": FORECAST_FAMILY,
        "backbone_name": BACKBONE_NAME_OTFLOW,
        "train_steps": int(args.steps),
        "train_budget_label": train_budget_label(int(args.steps)),
        "seed": int(args.seed),
        "history_len": int(spec.history_len),
        "future_block_len": int(spec.future_block_len),
        "cond_dim": 0,
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(artifact_root / "checkpoint_metadata.json"),
        "summary_path": str(artifact_root / "artifact_summary.json"),
        "split_stats": {**_json_ready_stats(dict(splits.get("stats", {}))), "cond_dim": 0},
        "cfg": cfg.to_dict(),
    }
    save_json(metadata, str(artifact_root / "checkpoint_metadata.json"))
    save_json(metadata, str(artifact_root / "artifact_summary.json"))
    manifest = materialize_backbone_manifest(budget_steps=(int(args.steps),), seed=int(args.seed))
    summary = {
        "status": "ready",
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(artifact_root / "checkpoint_metadata.json"),
        "manifest_path": str(project_backbone_matrix_root() / "backbone_manifest.json"),
        "manifest_ready_count": int(manifest.get("ready_count", 0)),
        "dataset": str(args.dataset),
        "train_steps": int(args.steps),
    }
    save_json(summary, str(artifact_root / "training_summary.json"))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the standalone genODE SF traffic OT flow-matching backbone.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_root", default=str(project_paper_dataset_root()))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=64)
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
