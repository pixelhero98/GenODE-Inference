from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import torch

from genode.distillation.artifacts import load_demonstration_manifest, write_json
from genode.distillation.checkpoint import save_flow_map_checkpoint
from genode.distillation.model import EndpointFlowMap, endpoint_consistency_loss
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.gipo.models import (
    setting_encoder_config_from_payload,
    setting_feature_dim,
    setting_features,
)
from genode.models.conditioning import ConditioningCache, ConditioningState
from genode.models.otflow_model import OTFlow
from genode.provenance import file_sha256
from genode.runtime import resolve_torch_device
from genode.solver_protocol import normalize_solver_nfe_fields


SplitName = Literal["train", "validation"]


@dataclass(frozen=True)
class FlowMapBatch:
    state: torch.Tensor
    source_time: torch.Tensor
    teacher_endpoint: torch.Tensor
    density_mass: torch.Tensor
    setting: torch.Tensor
    conditioning_cache: ConditioningCache


def _context_split(
    context_ids: Mapping[int, str],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[int, SplitName]:
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    if len(context_ids) < 2:
        raise ValueError("Context-disjoint validation requires at least two contexts.")
    ranked = sorted(
        context_ids,
        key=lambda index: hashlib.sha256(
            f"{int(seed)}\0{context_ids[index]}".encode("utf-8")
        ).digest(),
    )
    validation_count = min(
        len(ranked) - 1,
        max(1, int(round(len(ranked) * fraction))),
    )
    validation_indices = set(ranked[:validation_count])
    return {
        index: ("validation" if index in validation_indices else "train")
        for index in context_ids
    }


class DemonstrationStore:
    """Validated, shard-aware sampler for endpoint-map demonstrations."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        validation_fraction: float = 0.1,
        split_seed: int = 0,
        shard_cache_size: int = 2,
    ):
        self.manifest = load_demonstration_manifest(manifest_path)
        self.metadata = dict(self.manifest["metadata"])
        if bool(self.metadata.get("locked_test_used", False)):
            raise ValueError("Locked-test demonstrations may not be used for flow-map training.")
        self.setting_encoder_config = setting_encoder_config_from_payload(
            self.metadata.get("setting_encoder_config")
        )
        self.density_dim = int(self.metadata.get("density_bin_count", 0))
        self.state_dim = int(self.metadata.get("sample_state_dim", 0))
        if self.density_dim <= 1 or self.state_dim <= 0:
            raise ValueError("Demonstration metadata has invalid state or density dimensions.")
        self.rollouts_per_context = int(self.metadata.get("rollouts_per_context", 0))
        self.expected_context_count = int(self.metadata.get("context_count", 0))
        self.expected_settings = {
            (
                normalized.solver_key,
                normalized.target_nfe,
            )
            for normalized in (
                normalize_solver_nfe_fields(
                    str(item.get("solver_key", "")),
                    int(item.get("target_nfe", 0)),
                    source="demonstration metadata setting",
                )
                for item in self.metadata.get("settings", [])
            )
        }
        self._cache_limit = max(1, int(shard_cache_size))
        self._trajectory_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._context_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self.context_ids: dict[int, str] = {}
        self.context_records = [dict(record) for record in self.manifest["context_shards"]]
        self._context_locations: dict[int, tuple[int, int]] = {}
        context_schema: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        cond_emb_presence: set[bool] = set()
        seen_context_ids: set[str] = set()
        for record_index, record in enumerate(self.context_records):
            with np.load(record["resolved_path"], allow_pickle=False) as payload:
                required = {"context_index", "context_id", "ctx_tokens", "ctx_summary", "summary"}
                missing = sorted(required - set(payload.files))
                if missing:
                    raise ValueError(f"Context shard is missing arrays: {missing}.")
                raw_indices = np.asarray(payload["context_index"])
                if raw_indices.dtype.kind not in {"i", "u"}:
                    raise ValueError("Context shard context_index must use an integer dtype.")
                indices = raw_indices.astype(np.int64, copy=False)
                ids = np.asarray(payload["context_id"])
                if indices.ndim != 1 or ids.ndim != 1 or indices.shape != ids.shape:
                    raise ValueError("Context shard identifiers must be aligned one-dimensional arrays.")
                if ids.dtype.kind not in {"U", "S"}:
                    raise ValueError("Context shard context_id must use a string dtype.")
                for row_index, (index, context_id) in enumerate(zip(indices.tolist(), ids.tolist())):
                    integer_index = int(index)
                    if integer_index in self.context_ids:
                        raise ValueError(f"Duplicate context_index {integer_index} in demonstration artifact.")
                    text_id = context_id.decode("utf-8") if isinstance(context_id, bytes) else str(context_id)
                    if text_id in seen_context_ids:
                        raise ValueError(f"Duplicate context_id {text_id!r} in demonstration artifact.")
                    self.context_ids[integer_index] = text_id
                    self._context_locations[integer_index] = (record_index, row_index)
                    seen_context_ids.add(text_id)
                for name in ("ctx_tokens", "ctx_summary", "summary"):
                    value = np.asarray(payload[name])
                    if int(value.shape[0]) != int(indices.size):
                        raise ValueError(f"Context shard array {name!r} is not aligned with context_index.")
                    schema = (tuple(int(size) for size in value.shape[1:]), value.dtype)
                    if name in context_schema and context_schema[name] != schema:
                        raise ValueError(f"Context shard array {name!r} has an inconsistent schema.")
                    context_schema[name] = schema
                present = "cond_emb" in payload.files
                cond_emb_presence.add(present)
                if present:
                    value = np.asarray(payload["cond_emb"])
                    if int(value.shape[0]) != int(indices.size):
                        raise ValueError("Context shard cond_emb is not aligned with context_index.")
                    schema = (tuple(int(size) for size in value.shape[1:]), value.dtype)
                    if "cond_emb" in context_schema and context_schema["cond_emb"] != schema:
                        raise ValueError("Context shard cond_emb has an inconsistent schema.")
                    context_schema["cond_emb"] = schema
        if len(cond_emb_presence) > 1:
            raise ValueError("cond_emb must be present in every context shard or in none of them.")
        expected_indices = set(range(len(self.context_ids)))
        if set(self.context_ids) != expected_indices:
            raise ValueError("context_index values must be contiguous and start at zero.")
        if len(self.context_ids) != self.expected_context_count:
            raise ValueError("Context shards do not match metadata context_count.")
        self._has_cond_emb = True in cond_emb_presence
        self._context_schema = context_schema
        self.split_by_context = _context_split(
            self.context_ids,
            validation_fraction=validation_fraction,
            seed=split_seed,
        )
        self.trajectory_records = [dict(record) for record in self.manifest["trajectory_shards"]]
        self._eligible_rows: dict[tuple[int, SplitName], np.ndarray] = {}
        self._validate_trajectory_shards()

    def _read_context(self, record_index: int) -> dict[str, np.ndarray]:
        path = Path(self.context_records[record_index]["resolved_path"])
        cached = self._context_cache.pop(path, None)
        if cached is not None:
            self._context_cache[path] = cached
            return cached
        with np.load(path, allow_pickle=False) as payload:
            arrays = {str(name): np.asarray(payload[name]) for name in payload.files}
        self._context_cache[path] = arrays
        while len(self._context_cache) > self._cache_limit:
            self._context_cache.popitem(last=False)
        return arrays

    def _context_batch(
        self,
        context_indices: np.ndarray,
        *,
        device: torch.device,
    ) -> ConditioningCache:
        flat_indices = np.asarray(context_indices, dtype=np.int64)
        if flat_indices.ndim != 1 or flat_indices.size == 0:
            raise ValueError("A context batch must contain at least one context index.")
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, context_index in enumerate(flat_indices.tolist()):
            location = self._context_locations.get(int(context_index))
            if location is None:
                raise ValueError(f"Unknown context_index {int(context_index)}.")
            record_index, row_index = location
            grouped.setdefault(record_index, []).append((output_index, row_index))

        names = ["ctx_tokens", "ctx_summary", "summary"]
        if self._has_cond_emb:
            names.append("cond_emb")
        output: dict[str, np.ndarray] = {
            name: np.empty(
                (int(flat_indices.size), *self._context_schema[name][0]),
                dtype=self._context_schema[name][1],
            )
            for name in names
        }
        for record_index, locations in grouped.items():
            arrays = self._read_context(record_index)
            output_rows = np.asarray([item[0] for item in locations], dtype=np.int64)
            shard_rows = np.asarray([item[1] for item in locations], dtype=np.int64)
            for name in names:
                output[name][output_rows] = arrays[name][shard_rows]

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(output[name], dtype=torch.float32, device=device)

        return ConditioningCache(
            ctx_tokens=tensor("ctx_tokens"),
            ctx_summary=tensor("ctx_summary"),
            summary=tensor("summary"),
            cond_emb=tensor("cond_emb") if self._has_cond_emb else None,
        )

    def _read_trajectory(self, record_index: int) -> dict[str, np.ndarray]:
        path = Path(self.trajectory_records[record_index]["resolved_path"])
        cached = self._trajectory_cache.pop(path, None)
        if cached is not None:
            self._trajectory_cache[path] = cached
            return cached
        with np.load(path, allow_pickle=False) as payload:
            arrays = {str(name): np.asarray(payload[name]) for name in payload.files}
        self._trajectory_cache[path] = arrays
        while len(self._trajectory_cache) > self._cache_limit:
            self._trajectory_cache.popitem(last=False)
        return arrays

    def _validate_trajectory_shards(self) -> None:
        required = {
            "context_index",
            "noise_seed",
            "initial_state",
            "time_grid",
            "states",
            "density_mass",
        }
        split_counts: dict[SplitName, int] = {"train": 0, "validation": 0}
        self._records_by_context_range: dict[
            tuple[int, int], dict[tuple[str, int], int]
        ] = {}
        coverage: dict[tuple[str, int], list[tuple[int, int]]] = {
            setting: [] for setting in self.expected_settings
        }
        seed_signatures: dict[int, tuple[int, ...]] = {}
        for record_index, record in enumerate(self.trajectory_records):
            arrays = self._read_trajectory(record_index)
            missing = sorted(required - set(arrays))
            if missing:
                raise ValueError(f"Trajectory shard is missing arrays: {missing}.")
            context_index = arrays["context_index"]
            row_count = int(context_index.size)
            if context_index.ndim != 1 or row_count <= 0:
                raise ValueError("Trajectory context_index must be a non-empty vector.")
            if any(int(index) not in self.context_ids for index in context_index.tolist()):
                raise ValueError("Trajectory shard references an unknown context_index.")
            if arrays["noise_seed"].shape != (row_count,):
                raise ValueError("Trajectory noise_seed is not aligned with context_index.")
            if arrays["noise_seed"].dtype.kind not in {"i", "u"}:
                raise ValueError("Trajectory noise_seed must use an integer dtype.")
            initial = arrays["initial_state"]
            grids = arrays["time_grid"]
            states = arrays["states"]
            density = arrays["density_mass"]
            if initial.shape != (row_count, self.state_dim):
                raise ValueError("Trajectory initial_state has an incompatible shape.")
            if states.ndim != 3 or states.shape[0] != row_count or states.shape[2] != self.state_dim:
                raise ValueError("Trajectory states must have shape [rows, macro_steps + 1, state_dim].")
            if grids.shape != states.shape[:2]:
                raise ValueError("Trajectory time_grid must align with states.")
            if density.shape != (row_count, self.density_dim):
                raise ValueError("Trajectory density_mass has an incompatible shape.")
            if not np.allclose(initial, states[:, 0], atol=1e-6, rtol=1e-6):
                raise ValueError("Trajectory initial_state does not match states[:, 0].")
            if not np.allclose(grids[:, 0], 0.0) or not np.allclose(grids[:, -1], 1.0):
                raise ValueError("Every trajectory time grid must start at zero and end at one.")
            if not np.all(np.diff(grids, axis=1) > 0.0):
                raise ValueError("Every trajectory time grid must be strictly increasing.")
            if np.any(density < 0.0) or not np.allclose(density.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError("Every density_mass row must be nonnegative and sum to one.")
            nfe = normalize_solver_nfe_fields(
                str(record.get("solver_key", "")),
                int(record.get("target_nfe", 0)),
                source="demonstration trajectory shard",
            )
            if int(states.shape[1]) != nfe.macro_steps + 1:
                raise ValueError("Trajectory length does not match its solver/NFE metadata.")
            setting = (nfe.solver_key, nfe.target_nfe)
            if setting not in self.expected_settings:
                raise ValueError("Trajectory shard uses a setting not declared in metadata.")
            try:
                context_start = int(record["context_start"])
                context_stop = int(record["context_stop"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Trajectory shard requires integer context_start/context_stop metadata.") from exc
            if not 0 <= context_start < context_stop <= self.expected_context_count:
                raise ValueError("Trajectory shard context range is invalid.")
            expected_contexts = np.arange(context_start, context_stop, dtype=np.int64)
            actual_contexts, actual_counts = np.unique(context_index, return_counts=True)
            if not np.array_equal(actual_contexts, expected_contexts):
                raise ValueError("Trajectory shard does not exactly cover its declared context range.")
            if not np.all(actual_counts == self.rollouts_per_context):
                raise ValueError("Trajectory shard does not contain the declared rollouts per context.")
            for context_value in actual_contexts.tolist():
                selected_seeds = tuple(
                    sorted(
                        int(value)
                        for value in arrays["noise_seed"][context_index == context_value].tolist()
                    )
                )
                if len(set(selected_seeds)) != self.rollouts_per_context:
                    raise ValueError("Trajectory noise seeds must be unique within each context.")
                previous = seed_signatures.setdefault(int(context_value), selected_seeds)
                if previous != selected_seeds:
                    raise ValueError(
                        "Every solver/NFE setting must use the same rollout seeds per context."
                    )
            coverage[setting].append((context_start, context_stop))
            range_records = self._records_by_context_range.setdefault(
                (context_start, context_stop),
                {},
            )
            if setting in range_records:
                raise ValueError("A setting has duplicate trajectory shards for one context range.")
            range_records[setting] = record_index
            for split in ("train", "validation"):
                eligible = np.asarray(
                    [
                        row_index
                        for row_index, index in enumerate(context_index.tolist())
                        if self.split_by_context[int(index)] == split
                    ],
                    dtype=np.int64,
                )
                self._eligible_rows[(record_index, split)] = eligible
                split_counts[split] += int(eligible.size)
        for setting, ranges in coverage.items():
            cursor = 0
            for start, stop in sorted(ranges):
                if start != cursor:
                    raise ValueError(
                        f"Trajectory shards for setting {setting!r} have overlapping or missing contexts."
                    )
                cursor = stop
            if cursor != self.expected_context_count:
                raise ValueError(f"Trajectory shards for setting {setting!r} are incomplete.")
        if not all(split_counts.values()):
            raise ValueError("Demonstrations must contain rows for both context-disjoint splits.")

    def validation_record_panel(
        self,
        *,
        seed: int,
        shard_count: int,
    ) -> tuple[tuple[int, ...], ...]:
        count = int(shard_count)
        if count <= 0:
            raise ValueError("validation_shards must be positive.")
        complete_ranges = []
        for context_range, records_by_setting in self._records_by_context_range.items():
            if set(records_by_setting) != self.expected_settings:
                raise ValueError("A trajectory context range is incomplete across settings.")
            records = tuple(records_by_setting[setting] for setting in sorted(self.expected_settings))
            if any(self._eligible_rows[(record, "validation")].size > 0 for record in records):
                complete_ranges.append((context_range, records))
        ranked = sorted(
            complete_ranges,
            key=lambda item: hashlib.sha256(
                f"{int(seed)}\0{item[0][0]}\0{item[0][1]}".encode("utf-8")
            ).digest(),
        )
        if not ranked:
            raise ValueError("No validation context shards are available.")
        return tuple(records for _, records in ranked[:count])

    def sample_batch(
        self,
        batch_size: int,
        *,
        split: SplitName,
        generator: np.random.Generator,
        device: torch.device,
        setting: tuple[str, int] | None = None,
        record_index: int | None = None,
    ) -> FlowMapBatch:
        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive.")
        requested_setting = None
        if setting is not None:
            normalized = normalize_solver_nfe_fields(
                setting[0],
                setting[1],
                source="demonstration batch setting",
            )
            requested_setting = (normalized.solver_key, normalized.target_nfe)
        candidates = [
            record_index
            for record_index in range(len(self.trajectory_records))
            if self._eligible_rows[(record_index, split)].size > 0
            and (
                requested_setting is None
                or (
                    str(self.trajectory_records[record_index]["solver_key"]),
                    int(self.trajectory_records[record_index]["target_nfe"]),
                )
                == requested_setting
            )
        ]
        if not candidates:
            raise ValueError(f"No {split} rows are available.")
        if record_index is None:
            candidate_counts = np.asarray(
                [self._eligible_rows[(candidate, split)].size for candidate in candidates],
                dtype=np.float64,
            )
            selected_record = int(
                generator.choice(candidates, p=candidate_counts / candidate_counts.sum())
            )
        else:
            selected_record = int(record_index)
            if selected_record not in candidates:
                raise ValueError(
                    f"Trajectory record {selected_record} is not eligible for the requested batch."
                )
        record_index = selected_record
        eligible = self._eligible_rows[(record_index, split)]
        selected = generator.choice(eligible, size=size, replace=int(eligible.size) < size).astype(np.int64)
        arrays = self._read_trajectory(record_index)
        states = arrays["states"][selected]
        grids = arrays["time_grid"][selected]
        terminal_index = int(states.shape[1] - 1)
        source_indices = np.zeros((size,), dtype=np.int64)
        if terminal_index > 1:
            use_interior = generator.random(size) >= 0.5
            source_indices[use_interior] = generator.integers(
                1,
                terminal_index,
                size=int(use_interior.sum()),
            )
        rows = np.arange(size, dtype=np.int64)
        context_indices = arrays["context_index"][selected].astype(np.int64)
        record = self.trajectory_records[record_index]
        features = setting_features(
            str(record["solver_key"]),
            int(record["target_nfe"]),
            mode=self.setting_encoder_config.mode,
            config=self.setting_encoder_config,
        )

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        cache = self._context_batch(context_indices, device=device)
        return FlowMapBatch(
            state=tensor(states[rows, source_indices]),
            source_time=tensor(grids[rows, source_indices, None]),
            teacher_endpoint=tensor(states[:, -1]),
            density_mass=tensor(arrays["density_mass"][selected]),
            setting=features.to(device=device).unsqueeze(0).expand(size, -1),
            conditioning_cache=cache,
        )

    def split_context_count(self, split: SplitName) -> int:
        return sum(value == split for value in self.split_by_context.values())


def _conditioning_for_batch(
    backbone_model: OTFlow,
    batch: FlowMapBatch,
) -> ConditioningState:
    return backbone_model.backbone.build_conditioning(
        hist=None,  # The validated cache is the source of context features.
        x_ref=batch.state,
        t=batch.source_time,
        cache=batch.conditioning_cache,
    )


@torch.no_grad()
def _validation_loss(
    flow_map: EndpointFlowMap,
    backbone_model: OTFlow,
    store: DemonstrationStore,
    *,
    batch_size: int,
    validation_shards: int,
    seed: int,
    device: torch.device,
) -> float:
    flow_map.eval()
    generator = np.random.default_rng(int(seed))
    values = []
    for records in store.validation_record_panel(seed=seed, shard_count=validation_shards):
        for record_index in records:
            eligible_count = int(store._eligible_rows[(record_index, "validation")].size)
            if eligible_count <= 0:
                continue
            batch = store.sample_batch(
                min(int(batch_size), eligible_count),
                split="validation",
                generator=generator,
                device=device,
                record_index=record_index,
            )
            prediction = flow_map(
                batch.state,
                batch.source_time,
                _conditioning_for_batch(backbone_model, batch),
                batch.setting,
                batch.density_mass,
            )
            values.append(
                float(
                    endpoint_consistency_loss(
                        prediction,
                        batch.teacher_endpoint,
                        delta=flow_map.loss_delta,
                    ).cpu()
                )
            )
    return float(np.mean(values))


def train_endpoint_flow_map(
    backbone_model: OTFlow,
    store: DemonstrationStore,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    validation_interval: int,
    validation_shards: int,
    batches_per_shard: int,
    seed: int,
    initialize_from_backbone: bool = True,
) -> tuple[EndpointFlowMap, dict[str, Any]]:
    total_steps = int(steps)
    if total_steps <= 0:
        raise ValueError("steps must be positive.")
    if int(batch_size) <= 0 or float(learning_rate) <= 0.0:
        raise ValueError("batch_size and learning_rate must be positive.")
    if int(validation_interval) <= 0:
        raise ValueError("validation_interval must be positive.")
    if int(validation_shards) <= 0:
        raise ValueError("validation_shards must be positive.")
    if int(batches_per_shard) <= 0:
        raise ValueError("batches_per_shard must be positive.")
    device = next(backbone_model.parameters()).device
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    for parameter in backbone_model.parameters():
        parameter.requires_grad_(False)
    backbone_model.eval()
    setting_dim = setting_feature_dim(
        store.setting_encoder_config.mode,
        config=store.setting_encoder_config,
    )
    flow_map = EndpointFlowMap(
        backbone_model.cfg,
        setting_dim=setting_dim,
        density_dim=store.density_dim,
    ).to(device)
    if initialize_from_backbone:
        flow_map.initialize_from_teacher(backbone_model)
    optimizer = torch.optim.AdamW(
        flow_map.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    training_generator = np.random.default_rng(int(seed))
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    best_validation_loss = math.inf
    last_training_loss = math.inf
    validations: list[dict[str, float | int]] = []
    active_record: int | None = None
    for step in range(1, total_steps + 1):
        flow_map.train()
        if active_record is None or (step - 1) % int(batches_per_shard) == 0:
            eligible_records = [
                index
                for index in range(len(store.trajectory_records))
                if store._eligible_rows[(index, "train")].size > 0
            ]
            record_counts = np.asarray(
                [store._eligible_rows[(index, "train")].size for index in eligible_records],
                dtype=np.float64,
            )
            active_record = int(
                training_generator.choice(
                    eligible_records,
                    p=record_counts / record_counts.sum(),
                )
            )
        batch = store.sample_batch(
            batch_size,
            split="train",
            generator=training_generator,
            device=device,
            record_index=active_record,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = flow_map(
            batch.state,
            batch.source_time,
            _conditioning_for_batch(backbone_model, batch),
            batch.setting,
            batch.density_mass,
        )
        loss = endpoint_consistency_loss(
            prediction,
            batch.teacher_endpoint,
            delta=flow_map.loss_delta,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite flow-map loss at step {step}.")
        loss.backward()
        if float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(flow_map.parameters(), float(grad_clip))
        optimizer.step()
        last_training_loss = float(loss.detach().cpu())
        if step % int(validation_interval) == 0 or step == total_steps:
            value = _validation_loss(
                flow_map,
                backbone_model,
                store,
                batch_size=batch_size,
                validation_shards=validation_shards,
                seed=int(seed) + 1_000_003,
                device=device,
            )
            validations.append({"step": int(step), "loss": float(value)})
            if value < best_validation_loss:
                best_validation_loss = value
                best_step = step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in flow_map.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("Flow-map training completed without a validation checkpoint.")
    flow_map.load_state_dict(best_state, strict=True)
    flow_map.eval()
    summary = {
        "protocol": "context_disjoint_endpoint_consistency",
        "steps": total_steps,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
        "source_state_sampling": "half_initial_half_uniform_interior",
        "endpoint_target": "frozen_teacher_final_state",
        "loss": "pseudo_huber",
        "loss_delta": float(flow_map.loss_delta),
        "initialized_from_backbone_field": bool(initialize_from_backbone),
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "last_training_loss": float(last_training_loss),
        "validation_history": validations,
        "validation_shards": int(validation_shards),
        "validation_setting_count": len(store.expected_settings),
        "batches_per_shard": int(batches_per_shard),
        "train_context_count": store.split_context_count("train"),
        "validation_context_count": store.split_context_count("validation"),
        "locked_test_used_for_selection": False,
        "checkpoint_export": "final_selected_only",
    }
    return flow_map, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a one-evaluation endpoint flow map.")
    parser.add_argument("--demonstration-manifest", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--gipo-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--validation-shards", type=int, default=16)
    parser.add_argument("--batches-per-shard", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-backbone-initialization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    device = resolve_torch_device(args.device)
    backbone_path = Path(args.backbone_checkpoint).expanduser().resolve()
    gipo_path = Path(args.gipo_checkpoint).expanduser().resolve()
    manifest_path = Path(args.demonstration_manifest).expanduser().resolve()
    output_path = Path(args.output_checkpoint).expanduser().resolve()
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if str(args.summary_json).strip()
        else None
    )
    named_paths = {
        "demonstration manifest": manifest_path,
        "backbone checkpoint": backbone_path,
        "GIPO checkpoint": gipo_path,
        "flow-map checkpoint": output_path,
    }
    if summary_path is not None:
        named_paths["training summary"] = summary_path
    reverse: dict[Path, list[str]] = {}
    for label, path in named_paths.items():
        reverse.setdefault(path, []).append(label)
    collisions = [labels for labels in reverse.values() if len(labels) > 1]
    if collisions:
        raise ValueError(
            "Distillation input and output paths must be pairwise distinct; collisions: "
            + ", ".join("/".join(labels) for labels in collisions)
        )
    store = DemonstrationStore(
        manifest_path,
        validation_fraction=args.validation_fraction,
        split_seed=args.seed,
    )
    demonstration_root = manifest_path.parent
    for label, path in (
        ("flow-map checkpoint", output_path),
        ("training summary", summary_path),
    ):
        if path is not None and (
            path == demonstration_root or demonstration_root in path.parents
        ):
            raise ValueError(
                f"{label} must be outside the managed demonstration artifact directory."
            )
    expected_backbone_hash = str(store.metadata.get("backbone_checkpoint_sha256", ""))
    expected_gipo_hash = str(store.metadata.get("gipo_checkpoint_sha256", ""))
    if file_sha256(backbone_path) != expected_backbone_hash:
        raise ValueError("Demonstrations were generated by a different backbone checkpoint.")
    if file_sha256(gipo_path) != expected_gipo_hash:
        raise ValueError("Demonstrations were generated by a different GIPO checkpoint.")
    backbone_model, _ = load_checkpoint_model(backbone_path, device)
    if int(backbone_model.cfg.sample_state_dim) != store.state_dim:
        raise ValueError("Demonstration state dimension is incompatible with the backbone checkpoint.")
    flow_map, training_summary = train_endpoint_flow_map(
        backbone_model,
        store,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        validation_interval=args.validation_interval,
        validation_shards=args.validation_shards,
        batches_per_shard=args.batches_per_shard,
        seed=args.seed,
        initialize_from_backbone=not bool(args.no_backbone_initialization),
    )
    checkpoint = save_flow_map_checkpoint(
        output_path,
        flow_map,
        backbone_checkpoint=backbone_path,
        gipo_checkpoint=gipo_path,
        setting_encoder_config=store.setting_encoder_config,
        training_summary=training_summary,
        demonstration_manifest_sha256=file_sha256(manifest_path),
        overwrite=bool(args.overwrite),
    )
    if summary_path is not None:
        write_json(
            summary_path,
            {
                **training_summary,
                "status": "completed",
                "checkpoint_name": checkpoint.name,
                "checkpoint_sha256": file_sha256(checkpoint),
            },
        )
    print(checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DemonstrationStore",
    "FlowMapBatch",
    "main",
    "train_endpoint_flow_map",
]
