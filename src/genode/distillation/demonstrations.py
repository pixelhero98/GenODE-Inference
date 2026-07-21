from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterable, Sequence

import numpy as np
import torch

from genode.distillation.artifacts import (
    DEMONSTRATION_MANIFEST_NAME,
    DEMONSTRATION_TRAINING_SPLITS,
    load_demonstration_manifest,
    write_demonstration_manifest,
    write_npz_shard,
)
from genode.distillation.gipo_policy import GIPOSchedulePolicy, load_gipo_schedule_policy
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.gipo.density_representation import DENSITY_BIN_COUNT
from genode.models.conditioning import ConditioningCache
from genode.models.otflow_model import OTFlow
from genode.provenance import file_sha256
from genode.runtime import resolve_torch_device
from genode.solver_protocol import FlowTrajectory, SUPPORTED_SOLVER_KEYS, normalize_solver_nfe_fields


DEFAULT_DISTILLATION_NFES = (4, 6, 8, 10, 12, 14, 16, 20)
ALLOWED_DEMONSTRATION_SPLITS = DEMONSTRATION_TRAINING_SPLITS


@dataclass(frozen=True, order=True)
class DistillationSetting:
    solver_key: str
    target_nfe: int

    def __post_init__(self) -> None:
        normalized = normalize_solver_nfe_fields(
            self.solver_key,
            self.target_nfe,
            source="distillation setting",
        )
        object.__setattr__(self, "solver_key", normalized.solver_key)
        object.__setattr__(self, "target_nfe", normalized.target_nfe)


@dataclass(frozen=True)
class DistillationContexts:
    context_ids: tuple[str, ...]
    histories: torch.Tensor
    conditions: torch.Tensor | None = None

    def validate(self) -> "DistillationContexts":
        if not self.context_ids:
            raise ValueError("At least one context is required for demonstration collection.")
        if len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("context_ids must be unique.")
        if any(not str(value).strip() for value in self.context_ids):
            raise ValueError("context_ids may not be empty.")
        if self.histories.ndim != 3:
            raise ValueError(
                "histories must have shape [contexts, history_steps, features], "
                f"got {tuple(self.histories.shape)}."
            )
        if int(self.histories.shape[0]) != len(self.context_ids):
            raise ValueError("context_ids and histories must have the same first dimension.")
        if not self.histories.is_floating_point() or not torch.isfinite(self.histories).all():
            raise ValueError("histories must contain finite floating-point values.")
        if self.conditions is not None:
            if self.conditions.ndim != 2 or int(self.conditions.shape[0]) != len(self.context_ids):
                raise ValueError("conditions must have shape [contexts, condition_features].")
            if not self.conditions.is_floating_point() or not torch.isfinite(self.conditions).all():
                raise ValueError("conditions must contain finite floating-point values.")
        return self


def default_distillation_settings() -> tuple[DistillationSetting, ...]:
    return tuple(
        DistillationSetting(solver_key, target_nfe)
        for solver_key in SUPPORTED_SOLVER_KEYS
        for target_nfe in DEFAULT_DISTILLATION_NFES
    )


def parse_distillation_settings(value: str) -> tuple[DistillationSetting, ...]:
    text = str(value).strip()
    if not text:
        return default_distillation_settings()
    settings: list[DistillationSetting] = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid setting {item!r}; expected comma-separated solver_key:target_nfe values."
            )
        settings.append(DistillationSetting(parts[0].strip(), int(parts[1])))
    if len(set(settings)) != len(settings):
        raise ValueError("Distillation settings may not contain duplicates.")
    return tuple(settings)


def _cache_to_cpu(cache: ConditioningCache) -> dict[str, torch.Tensor | None]:
    return {
        "ctx_tokens": cache.ctx_tokens.detach().cpu(),
        "ctx_summary": cache.ctx_summary.detach().cpu(),
        "summary": cache.summary.detach().cpu(),
        "cond_emb": None if cache.cond_emb is None else cache.cond_emb.detach().cpu(),
    }


def _concatenate_caches(caches: Sequence[dict[str, torch.Tensor | None]]) -> dict[str, torch.Tensor | None]:
    if not caches:
        raise ValueError("Cannot concatenate an empty context cache list.")
    output: dict[str, torch.Tensor | None] = {}
    for key in ("ctx_tokens", "ctx_summary", "summary", "cond_emb"):
        values = [cache[key] for cache in caches]
        if all(value is None for value in values):
            output[key] = None
            continue
        if any(value is None for value in values):
            raise ValueError(f"Context cache field {key!r} is inconsistently present.")
        tensors = [value for value in values if value is not None]
        output[key] = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    return output


def _slice_cache(
    cache: dict[str, torch.Tensor | None],
    index: int,
    *,
    repeats: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ConditioningCache:
    def repeated(name: str) -> torch.Tensor | None:
        value = cache[name]
        if value is None:
            return None
        selected = value[index : index + 1].to(device=device, dtype=dtype)
        return selected.expand(repeats, *selected.shape[1:]).contiguous()

    ctx_tokens = repeated("ctx_tokens")
    ctx_summary = repeated("ctx_summary")
    summary = repeated("summary")
    if ctx_tokens is None or ctx_summary is None or summary is None:
        raise ValueError("Context cache is missing required backbone fields.")
    return ConditioningCache(
        ctx_tokens=ctx_tokens,
        ctx_summary=ctx_summary,
        summary=summary,
        cond_emb=repeated("cond_emb"),
    )


def _noise_seed(
    base_seed: int,
    *,
    context_id: str,
    rollout_index: int,
) -> int:
    # Common random numbers make teacher endpoints directly comparable across
    # solver/NFE settings for the same context and rollout.
    encoded = f"{int(base_seed)}\0{context_id}\0{int(rollout_index)}".encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _initial_states(
    *,
    state_dim: int,
    seeds: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows = []
    for seed in seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        rows.append(torch.randn((state_dim,), generator=generator, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device=device, dtype=dtype)


def _validate_output_root(root: Path, *, overwrite: bool) -> None:
    manifest_path = root / DEMONSTRATION_MANIFEST_NAME
    shards_path = root / "shards"
    if root.exists() and not root.is_dir():
        raise ValueError("Demonstration output must be a directory path.")
    if not manifest_path.exists() and not shards_path.exists():
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing to write demonstrations into a non-empty unmanaged directory.")
        return
    if not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite an existing demonstration artifact in {root.name!r}."
        )
    if not manifest_path.is_file():
        raise ValueError(
            "Safe overwrite requires an intact demonstration manifest; refusing partial output."
        )
    manifest = load_demonstration_manifest(manifest_path)
    managed_files = {
        manifest_path.resolve(),
        *(
            Path(record["resolved_path"]).resolve()
            for record in [*manifest["context_shards"], *manifest["trajectory_shards"]]
        ),
    }
    discovered_files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Refusing to overwrite a demonstration directory containing links.")
        if path.is_file():
            discovered_files.add(path.resolve())
    if discovered_files != managed_files:
        raise ValueError(
            "Refusing to overwrite a demonstration directory containing unmanaged files."
        )


def _promote_staged_artifact(staging: Path, target: Path) -> None:
    if not target.exists():
        os.replace(staging, target)
        return
    if not any(target.iterdir()):
        target.rmdir()
        os.replace(staging, target)
        return
    backup = target.with_name(f".{target.name}.backup-{time.time_ns()}")
    os.replace(target, backup)
    try:
        os.replace(staging, target)
    except BaseException:
        os.replace(backup, target)
        raise
    shutil.rmtree(backup)


@torch.no_grad()
def _collect_flow_map_demonstrations_into(
    backbone_model: OTFlow,
    gipo_policy: GIPOSchedulePolicy,
    contexts: DistillationContexts,
    *,
    settings: Sequence[DistillationSetting],
    output_dir: str | Path,
    split_phase: str,
    scenario_key: str,
    benchmark_family: str,
    backbone_checkpoint_sha256: str,
    gipo_checkpoint_sha256: str,
    rollouts_per_context: int = 4,
    context_batch_size: int = 8,
    shard_contexts: int = 8,
    seed: int = 0,
) -> Path:
    """Collect frozen GIPO-guided teacher trajectories as portable NPZ shards."""

    contexts.validate()
    requested_settings = tuple(settings)
    if not requested_settings:
        raise ValueError("At least one solver/NFE setting is required.")
    if len(set(requested_settings)) != len(requested_settings):
        raise ValueError("Distillation settings may not contain duplicates.")
    phase = str(split_phase).strip()
    if phase not in ALLOWED_DEMONSTRATION_SPLITS:
        raise ValueError(
            f"split_phase must be a non-test training or validation split, got {split_phase!r}."
        )
    rollout_count = int(rollouts_per_context)
    if rollout_count <= 0:
        raise ValueError("rollouts_per_context must be positive.")
    cache_batch = int(context_batch_size)
    shard_size = int(shard_contexts)
    if cache_batch <= 0 or shard_size <= 0:
        raise ValueError("context_batch_size and shard_contexts must be positive.")
    if gipo_policy.density_dim != DENSITY_BIN_COUNT:
        raise ValueError(
            f"Flow-map distillation requires the {DENSITY_BIN_COUNT}-bin GIPO density representation."
        )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    device = next(backbone_model.parameters()).device
    dtype = next(backbone_model.parameters()).dtype
    backbone_model.eval()
    gipo_policy.student.eval()
    context_shards: list[dict[str, object]] = []
    trajectory_shards: list[dict[str, object]] = []
    state_dim = int(backbone_model.cfg.sample_state_dim)
    for shard_index, shard_start in enumerate(range(0, len(contexts.context_ids), shard_size)):
        shard_stop = min(len(contexts.context_ids), shard_start + shard_size)
        cache_parts: list[dict[str, torch.Tensor | None]] = []
        for batch_start in range(shard_start, shard_stop, cache_batch):
            batch_stop = min(shard_stop, batch_start + cache_batch)
            histories = contexts.histories[batch_start:batch_stop].to(device=device, dtype=dtype)
            conditions = None
            if contexts.conditions is not None:
                conditions = contexts.conditions[batch_start:batch_stop].to(
                    device=device,
                    dtype=dtype,
                )
            cache_parts.append(
                _cache_to_cpu(backbone_model.backbone.precompute(histories, cond=conditions))
            )
        shard_cache = _concatenate_caches(cache_parts)
        ctx_summary = shard_cache["ctx_summary"]
        ctx_tokens = shard_cache["ctx_tokens"]
        summary = shard_cache["summary"]
        if ctx_summary is None or ctx_tokens is None or summary is None:
            raise ValueError("Backbone context cache is missing required fields.")

        arrays: dict[str, object] = {
            "context_index": np.arange(shard_start, shard_stop, dtype=np.int64),
            "context_id": np.asarray(contexts.context_ids[shard_start:shard_stop], dtype=np.str_),
            "ctx_tokens": ctx_tokens.numpy(),
            "ctx_summary": ctx_summary.numpy(),
            "summary": summary.numpy(),
        }
        if shard_cache["cond_emb"] is not None:
            arrays["cond_emb"] = shard_cache["cond_emb"].numpy()  # type: ignore[union-attr]
        record = write_npz_shard(
            root,
            f"shards/contexts_{shard_index:05d}.npz",
            arrays,
        )
        context_shards.append(record)

        for setting in requested_settings:
            context_index_rows: list[np.ndarray] = []
            seed_rows: list[np.ndarray] = []
            initial_rows: list[np.ndarray] = []
            grid_rows: list[np.ndarray] = []
            state_rows: list[np.ndarray] = []
            density_rows: list[np.ndarray] = []
            for local_index, context_index in enumerate(range(shard_start, shard_stop)):
                context_summary = ctx_summary[local_index : local_index + 1].to(
                    device=device,
                    dtype=dtype,
                )
                schedule = gipo_policy.predict(
                    context_summary,
                    solver_key=setting.solver_key,
                    target_nfe=setting.target_nfe,
                )
                seeds = [
                    _noise_seed(
                        seed,
                        context_id=contexts.context_ids[context_index],
                        rollout_index=rollout_index,
                    )
                    for rollout_index in range(rollout_count)
                ]
                initial = _initial_states(
                    state_dim=state_dim,
                    seeds=seeds,
                    device=device,
                    dtype=dtype,
                )
                conditioning_cache = _slice_cache(
                    shard_cache,
                    local_index,
                    repeats=rollout_count,
                    device=device,
                    dtype=dtype,
                )
                trajectory = backbone_model.solve(
                    initial,
                    conditioning_cache=conditioning_cache,
                    solver_key=setting.solver_key,
                    target_nfe=setting.target_nfe,
                    time_grid=schedule.time_grid[0],
                    return_trajectory=True,
                )
                if not isinstance(trajectory, FlowTrajectory):
                    raise RuntimeError("OTFlow.solve did not return a trajectory.")
                context_index_rows.append(np.full((rollout_count,), context_index, dtype=np.int64))
                seed_rows.append(np.asarray(seeds, dtype=np.int64))
                initial_rows.append(initial.detach().cpu().numpy())
                grid_rows.append(
                    schedule.time_grid.expand(rollout_count, -1).detach().cpu().numpy()
                )
                state_rows.append(trajectory.states.detach().cpu().numpy())
                density_rows.append(
                    schedule.density_mass.expand(rollout_count, -1).detach().cpu().numpy()
                )
            arrays = {
                "context_index": np.concatenate(context_index_rows, axis=0),
                "noise_seed": np.concatenate(seed_rows, axis=0),
                "initial_state": np.concatenate(initial_rows, axis=0),
                "time_grid": np.concatenate(grid_rows, axis=0),
                "states": np.concatenate(state_rows, axis=0),
                "density_mass": np.concatenate(density_rows, axis=0),
            }
            record = write_npz_shard(
                root,
                (
                    f"shards/trajectories_{setting.solver_key}_nfe{setting.target_nfe}_"
                    f"{shard_index:05d}.npz"
                ),
                arrays,
            )
            record.update(
                {
                    "solver_key": setting.solver_key,
                    "target_nfe": int(setting.target_nfe),
                    "context_start": int(shard_start),
                    "context_stop": int(shard_stop),
                }
            )
            trajectory_shards.append(record)

    metadata = {
        "artifact_version": 1,
        "split_phase": phase,
        "scenario_key": str(scenario_key),
        "benchmark_family": str(benchmark_family),
        "context_count": int(len(contexts.context_ids)),
        "rollouts_per_context": rollout_count,
        "sample_state_dim": state_dim,
        "density_bin_count": int(gipo_policy.density_dim),
        "density_reference_time_grid": list(gipo_policy.reference_time_grid),
        "setting_encoder_config": gipo_policy.setting_encoder_config.to_payload(),
        "settings": [
            {"solver_key": item.solver_key, "target_nfe": int(item.target_nfe)}
            for item in requested_settings
        ],
        "backbone_checkpoint_sha256": str(backbone_checkpoint_sha256),
        "gipo_checkpoint_sha256": str(gipo_checkpoint_sha256),
        "classifier_free_guidance_scale": float(backbone_model.cfg.sample.cfg_scale),
        "locked_test_used": False,
    }
    return write_demonstration_manifest(
        root,
        context_shards=context_shards,
        trajectory_shards=trajectory_shards,
        metadata=metadata,
    )


def collect_flow_map_demonstrations(
    backbone_model: OTFlow,
    gipo_policy: GIPOSchedulePolicy,
    contexts: DistillationContexts,
    *,
    settings: Sequence[DistillationSetting],
    output_dir: str | Path,
    split_phase: str,
    scenario_key: str,
    benchmark_family: str,
    backbone_checkpoint_sha256: str,
    gipo_checkpoint_sha256: str,
    rollouts_per_context: int = 4,
    context_batch_size: int = 8,
    shard_contexts: int = 8,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Collect demonstrations in staging, then atomically replace a verified artifact."""

    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    _validate_output_root(target, overwrite=bool(overwrite))
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=target.parent,
        )
    ).resolve()
    try:
        _collect_flow_map_demonstrations_into(
            backbone_model,
            gipo_policy,
            contexts,
            settings=settings,
            output_dir=staging,
            split_phase=split_phase,
            scenario_key=scenario_key,
            benchmark_family=benchmark_family,
            backbone_checkpoint_sha256=backbone_checkpoint_sha256,
            gipo_checkpoint_sha256=gipo_checkpoint_sha256,
            rollouts_per_context=rollouts_per_context,
            context_batch_size=context_batch_size,
            shard_contexts=shard_contexts,
            seed=seed,
        )
        _promote_staged_artifact(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target / DEMONSTRATION_MANIFEST_NAME


def load_distillation_contexts(path: str | Path) -> DistillationContexts:
    input_path = Path(path).expanduser().resolve()
    try:
        with np.load(input_path, allow_pickle=False) as payload:
            required = {"context_ids", "histories"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise ValueError(f"Context NPZ is missing arrays: {missing}.")
            ids_array = np.asarray(payload["context_ids"])
            if ids_array.ndim != 1 or ids_array.dtype.kind not in {"U", "S"}:
                raise ValueError("context_ids must be a one-dimensional Unicode or byte-string array.")
            context_ids = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in ids_array.tolist()
            )
            histories = torch.from_numpy(np.asarray(payload["histories"])).float()
            conditions = (
                torch.from_numpy(np.asarray(payload["conditions"])).float()
                if "conditions" in payload.files
                else None
            )
    except OSError as exc:
        raise ValueError(f"Could not read context NPZ {input_path.name!r}: {exc}") from exc
    return DistillationContexts(context_ids, histories, conditions).validate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect frozen GIPO-guided OTFlow trajectories for endpoint-map distillation."
    )
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--gipo-checkpoint", required=True)
    parser.add_argument("--contexts-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-phase", required=True)
    parser.add_argument("--scenario-key", required=True)
    parser.add_argument("--benchmark-family", required=True)
    parser.add_argument(
        "--settings",
        default="",
        help="Comma-separated solver_key:target_nfe pairs; empty uses the supported 4x8 matrix.",
    )
    parser.add_argument("--rollouts-per-context", type=int, default=4)
    parser.add_argument("--context-batch-size", type=int, default=8)
    parser.add_argument("--shard-contexts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    device = resolve_torch_device(str(args.device))
    backbone_path = Path(args.backbone_checkpoint).expanduser().resolve()
    gipo_path = Path(args.gipo_checkpoint).expanduser().resolve()
    contexts_path = Path(args.contexts_npz).expanduser().resolve()
    output_manifest = (
        Path(args.output_dir).expanduser().resolve() / DEMONSTRATION_MANIFEST_NAME
    )
    if output_manifest in {backbone_path, gipo_path, contexts_path}:
        raise ValueError("Demonstration output must differ from every input artifact path.")
    backbone_model, _ = load_checkpoint_model(backbone_path, device)
    gipo_policy = load_gipo_schedule_policy(gipo_path, device=device)
    manifest = collect_flow_map_demonstrations(
        backbone_model,
        gipo_policy,
        load_distillation_contexts(contexts_path),
        settings=parse_distillation_settings(args.settings),
        output_dir=args.output_dir,
        split_phase=args.split_phase,
        scenario_key=args.scenario_key,
        benchmark_family=args.benchmark_family,
        backbone_checkpoint_sha256=file_sha256(backbone_path),
        gipo_checkpoint_sha256=file_sha256(gipo_path),
        rollouts_per_context=args.rollouts_per_context,
        context_batch_size=args.context_batch_size,
        shard_contexts=args.shard_contexts,
        seed=args.seed,
        overwrite=bool(args.overwrite),
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALLOWED_DEMONSTRATION_SPLITS",
    "DEFAULT_DISTILLATION_NFES",
    "DistillationContexts",
    "DistillationSetting",
    "collect_flow_map_demonstrations",
    "default_distillation_settings",
    "load_distillation_contexts",
    "main",
    "parse_distillation_settings",
]
