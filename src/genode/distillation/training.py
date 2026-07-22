from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
import torch

from genode.artifact_bundle import (
    bundle_journal_path,
    discard_temporary_bundle_path,
    preflight_artifact_bundle,
    promote_artifact_bundle,
    recover_artifact_bundle,
    temporary_bundle_path,
    validate_artifact_bundle,
    validate_artifact_bundle_layout,
)
from genode.checkpoint_validation import validate_strict_integer
from genode.data.otflow_paths import resolve_project_path
from genode.distillation.artifacts import (
    DEMONSTRATION_TRAINING_SPLITS,
    load_demonstration_manifest,
    validate_context_binding,
)
from genode.distillation.checkpoint import save_flow_map_checkpoint
from genode.distillation.gipo_policy import load_gipo_schedule_policy
from genode.distillation.model import EndpointFlowMap, endpoint_consistency_loss
from genode.evaluation.otflow_evaluation_support import load_checkpoint_model
from genode.gipo.models import (
    setting_encoder_config_from_payload,
    setting_feature_dim,
    setting_features,
)
from genode.gipo.policy import validate_context_embedding_kind
from genode.gipo.density_representation import (
    density_mass_to_time_grid,
    validate_reference_grid,
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


def _validate_continuous_array(
    value: np.ndarray,
    *,
    name: str,
    rank: int,
) -> None:
    if value.dtype.kind != "f":
        raise ValueError(
            f"{name} must use a real floating-point dtype, got {value.dtype}."
        )
    if value.ndim != int(rank):
        raise ValueError(f"{name} must have rank {int(rank)}, got shape {value.shape}.")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} must have non-empty dimensions, got shape {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")


def _context_split(
    context_fingerprints: Mapping[int, str],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[int, SplitName]:
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    split_seed = validate_strict_integer(
        seed,
        label="context split seed",
        minimum=0,
    )
    if len(context_fingerprints) < 2:
        raise ValueError("Context-disjoint validation requires at least two contexts.")
    ranked = sorted(
        context_fingerprints,
        key=lambda index: hashlib.sha256(
            f"{split_seed}\0{context_fingerprints[index]}".encode("utf-8")
        ).digest(),
    )
    validation_count = min(
        len(ranked) - 1,
        max(1, int(round(len(ranked) * fraction))),
    )
    validation_indices = set(ranked[:validation_count])
    return {
        index: ("validation" if index in validation_indices else "train")
        for index in context_fingerprints
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
        split_phase = str(self.metadata.get("split_phase", "")).strip()
        if split_phase not in DEMONSTRATION_TRAINING_SPLITS:
            raise ValueError(
                "Flow-map training requires demonstrations from a training split; "
                f"got {split_phase!r}."
            )
        self.context_binding = validate_context_binding(
            self.metadata.get("context_binding", {})
        )
        self.setting_encoder_config = setting_encoder_config_from_payload(
            self.metadata.get("setting_encoder_config"), require_complete=True
        )
        self.density_dim = validate_strict_integer(
            self.metadata.get("density_bin_count"),
            label="Demonstration density_bin_count",
            minimum=2,
        )
        self.state_dim = validate_strict_integer(
            self.metadata.get("sample_state_dim"),
            label="Demonstration sample_state_dim",
            minimum=1,
        )
        self.density_reference_time_grid = validate_reference_grid(
            self.metadata.get("density_reference_time_grid", ())
        )
        if len(self.density_reference_time_grid) != self.density_dim + 1:
            raise ValueError(
                "Demonstration density reference grid does not match density_bin_count."
            )
        self.rollouts_per_context = validate_strict_integer(
            self.metadata.get("rollouts_per_context"),
            label="Demonstration rollouts_per_context",
            minimum=1,
        )
        self.expected_context_count = validate_strict_integer(
            self.metadata.get("context_count"),
            label="Demonstration context_count",
            minimum=1,
        )
        self.expected_settings = {
            (
                normalized.solver_key,
                normalized.target_nfe,
            )
            for normalized in (
                normalize_solver_nfe_fields(
                    str(item.get("solver_key", "")),
                    validate_strict_integer(
                        item.get("target_nfe"),
                        label="Demonstration setting target_nfe",
                        minimum=1,
                    ),
                    source="demonstration metadata setting",
                )
                for item in self.metadata.get("settings", [])
            )
        }
        self._cache_limit = max(1, int(shard_cache_size))
        self._trajectory_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._context_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self.context_ids: dict[int, str] = {}
        self.context_fingerprints: dict[int, str] = {}
        self.context_records = [dict(record) for record in self.manifest["context_shards"]]
        self._context_locations: dict[int, tuple[int, int]] = {}
        context_schema: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
        cond_emb_presence: set[bool] = set()
        seen_context_ids: set[str] = set()
        for record_index, record in enumerate(self.context_records):
            with np.load(record["resolved_path"], allow_pickle=False) as payload:
                required = {
                    "context_index",
                    "context_id",
                    "context_fingerprint",
                    "ctx_tokens",
                    "ctx_summary",
                    "summary",
                }
                missing = sorted(required - set(payload.files))
                if missing:
                    raise ValueError(f"Context shard is missing arrays: {missing}.")
                raw_indices = np.asarray(payload["context_index"])
                if raw_indices.dtype.kind not in {"i", "u"}:
                    raise ValueError("Context shard context_index must use an integer dtype.")
                indices = raw_indices.astype(np.int64, copy=False)
                ids = np.asarray(payload["context_id"])
                fingerprints = np.asarray(payload["context_fingerprint"])
                if (
                    indices.ndim != 1
                    or ids.ndim != 1
                    or fingerprints.ndim != 1
                    or indices.shape != ids.shape
                    or indices.shape != fingerprints.shape
                ):
                    raise ValueError("Context shard identifiers must be aligned one-dimensional arrays.")
                if ids.dtype.kind not in {"U", "S"} or fingerprints.dtype.kind not in {"U", "S"}:
                    raise ValueError(
                        "Context shard context_id and context_fingerprint must use string dtypes."
                    )
                for row_index, (index, context_id, fingerprint) in enumerate(
                    zip(indices.tolist(), ids.tolist(), fingerprints.tolist(), strict=True)
                ):
                    integer_index = int(index)
                    if integer_index in self.context_ids:
                        raise ValueError(f"Duplicate context_index {integer_index} in demonstration artifact.")
                    text_id = context_id.decode("utf-8") if isinstance(context_id, bytes) else str(context_id)
                    text_fingerprint = (
                        fingerprint.decode("utf-8")
                        if isinstance(fingerprint, bytes)
                        else str(fingerprint)
                    )
                    if text_id in seen_context_ids:
                        raise ValueError(f"Duplicate context_id {text_id!r} in demonstration artifact.")
                    self.context_ids[integer_index] = text_id
                    self.context_fingerprints[integer_index] = text_fingerprint
                    self._context_locations[integer_index] = (record_index, row_index)
                    seen_context_ids.add(text_id)
                for name in ("ctx_tokens", "ctx_summary", "summary"):
                    value = np.asarray(payload[name])
                    _validate_continuous_array(
                        value,
                        name=f"Context shard {name}",
                        rank=3 if name == "ctx_tokens" else 2,
                    )
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
                    _validate_continuous_array(
                        value,
                        name="Context shard cond_emb",
                        rank=2,
                    )
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
        self.validation_fraction = float(validation_fraction)
        self.split_seed = validate_strict_integer(
            split_seed,
            label="demonstration split seed",
            minimum=0,
        )
        self.split_by_context = _context_split(
            self.context_fingerprints,
            validation_fraction=self.validation_fraction,
            seed=self.split_seed,
        )
        self.trajectory_records = [dict(record) for record in self.manifest["trajectory_shards"]]
        self._eligible_rows: dict[tuple[int, SplitName], np.ndarray] = {}
        self._setting_by_record: dict[int, tuple[str, int]] = {}
        self._validate_trajectory_shards()
        self._setting_features = {
            setting: setting_features(
                setting[0],
                setting[1],
                mode=self.setting_encoder_config.mode,
                config=self.setting_encoder_config,
            )
            for setting in self.expected_settings
        }
        self._candidate_records: dict[
            tuple[SplitName, tuple[str, int] | None],
            tuple[int, ...],
        ] = {}
        self._candidate_probabilities: dict[
            tuple[SplitName, tuple[str, int] | None],
            np.ndarray,
        ] = {}
        for split in ("train", "validation"):
            for setting in (None, *sorted(self.expected_settings)):
                records = tuple(
                    index
                    for index in range(len(self.trajectory_records))
                    if self._eligible_rows[(index, split)].size > 0
                    and (setting is None or self._setting_by_record[index] == setting)
                )
                self._candidate_records[(split, setting)] = records
                if records:
                    counts = np.asarray(
                        [self._eligible_rows[(index, split)].size for index in records],
                        dtype=np.float64,
                    )
                    self._candidate_probabilities[(split, setting)] = counts / counts.sum()

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
        initial_state_by_context_seed: dict[tuple[int, int], np.ndarray] = {}
        schedule_by_context_setting: dict[
            tuple[int, tuple[str, int]], tuple[np.ndarray, np.ndarray]
        ] = {}
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
            for name, value, rank in (
                ("initial_state", initial, 2),
                ("time_grid", grids, 2),
                ("states", states, 3),
                ("density_mass", density, 2),
            ):
                _validate_continuous_array(
                    value,
                    name=f"Trajectory {name}",
                    rank=rank,
                )
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
                validate_strict_integer(
                    record.get("target_nfe"),
                    label="Demonstration trajectory target_nfe",
                    minimum=1,
                ),
                source="demonstration trajectory shard",
            )
            if int(states.shape[1]) != nfe.macro_steps + 1:
                raise ValueError("Trajectory length does not match its solver/NFE metadata.")
            reconstructed_grids = np.asarray(
                [
                    density_mass_to_time_grid(
                        row.tolist(),
                        reference_time_grid=self.density_reference_time_grid,
                        macro_steps=nfe.macro_steps,
                    )
                    for row in density
                ],
                dtype=np.float64,
            )
            if not np.allclose(grids, reconstructed_grids, atol=2e-6, rtol=2e-5):
                raise ValueError(
                    "Trajectory time_grid does not match its paired density_mass."
                )
            setting = (nfe.solver_key, nfe.target_nfe)
            if setting not in self.expected_settings:
                raise ValueError("Trajectory shard uses a setting not declared in metadata.")
            self._setting_by_record[record_index] = setting
            context_start = validate_strict_integer(
                record.get("context_start"),
                label="Trajectory shard context_start",
                minimum=0,
            )
            context_stop = validate_strict_integer(
                record.get("context_stop"),
                label="Trajectory shard context_stop",
                minimum=1,
            )
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
            for row_index, (context_value, noise_seed) in enumerate(
                zip(context_index.tolist(), arrays["noise_seed"].tolist(), strict=True)
            ):
                schedule_key = (int(context_value), setting)
                previous_schedule = schedule_by_context_setting.get(schedule_key)
                if previous_schedule is None:
                    schedule_by_context_setting[schedule_key] = (
                        density[row_index].copy(),
                        grids[row_index].copy(),
                    )
                elif not (
                    np.allclose(
                        previous_schedule[0],
                        density[row_index],
                        atol=1e-7,
                        rtol=1e-6,
                    )
                    and np.allclose(
                        previous_schedule[1],
                        grids[row_index],
                        atol=1e-7,
                        rtol=1e-6,
                    )
                ):
                    raise ValueError(
                        "A frozen GIPO policy must produce one deterministic schedule for "
                        "each context and solver/NFE setting across rollout seeds."
                    )
                state_key = (int(context_value), int(noise_seed))
                previous_state = initial_state_by_context_seed.get(state_key)
                if previous_state is None:
                    initial_state_by_context_seed[state_key] = initial[row_index].copy()
                    continue
                if not np.allclose(
                    previous_state,
                    initial[row_index],
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError(
                        "Every solver/NFE setting must use the same initial state for each "
                        "context and rollout seed."
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
        candidates = self._candidate_records.get((split, requested_setting), ())
        if not candidates:
            raise ValueError(f"No {split} rows are available.")
        if record_index is None:
            selected_record = int(
                generator.choice(
                    candidates,
                    p=self._candidate_probabilities[(split, requested_setting)],
                )
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
        features = self._setting_features[self._setting_by_record[record_index]]

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
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    if not math.isfinite(float(grad_clip)) or float(grad_clip) < 0.0:
        raise ValueError("grad_clip must be finite and nonnegative.")
    if int(validation_interval) <= 0:
        raise ValueError("validation_interval must be positive.")
    if int(validation_shards) <= 0:
        raise ValueError("validation_shards must be positive.")
    if int(batches_per_shard) <= 0:
        raise ValueError("batches_per_shard must be positive.")
    training_seed = validate_strict_integer(
        seed,
        label="flow-map training seed",
        minimum=0,
    )
    device = next(backbone_model.parameters()).device
    torch.manual_seed(training_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_seed)
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
    training_generator = np.random.default_rng(training_seed)
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
                seed=training_seed + 1_000_003,
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
        "seed": training_seed,
        "split_seed": int(store.split_seed),
        "source_state_sampling": "half_initial_half_uniform_interior",
        "endpoint_target": "frozen_teacher_final_state",
        "loss": "pseudo_huber",
        "loss_delta": float(flow_map.loss_delta),
        "initialized_from_backbone_field": bool(initialize_from_backbone),
        "best_step": int(best_step),
        "best_validation_loss": float(best_validation_loss),
        "last_training_loss": float(last_training_loss),
        "validation_history": validations,
        "validation_fraction": float(store.validation_fraction),
        "validation_interval": int(validation_interval),
        "validation_shards": int(validation_shards),
        "validation_seed": training_seed + 1_000_003,
        "validation_setting_count": len(store.expected_settings),
        "batches_per_shard": int(batches_per_shard),
        "train_context_count": store.split_context_count("train"),
        "validation_context_count": store.split_context_count("validation"),
        "locked_test_used_for_selection": False,
        "checkpoint_export": "final_selected_only",
    }
    return flow_map, summary


def _flow_map_bundle_targets(
    checkpoint_path: Path,
    summary_path: Path | None,
) -> dict[str, Path]:
    targets = {"checkpoint": checkpoint_path.expanduser().resolve()}
    if summary_path is not None:
        targets["summary"] = summary_path.expanduser().resolve()
    validate_artifact_bundle_layout(targets["checkpoint"], targets)
    return targets


def _validate_flow_map_bundle_identity(
    paths: Mapping[str, Path],
    targets: Mapping[str, Path],
) -> None:
    checkpoint_path = paths["checkpoint"]
    if "summary" not in targets:
        return
    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read flow-map training summary: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise ValueError("Flow-map training summary must contain a JSON object.")
    if (
        summary.get("status") != "completed"
        or summary.get("checkpoint_name") != targets["checkpoint"].name
        or summary.get("checkpoint_sha256") != file_sha256(checkpoint_path)
    ):
        raise ValueError(
            "Flow-map checkpoint and training summary do not form a complete bound bundle."
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError(f"Could not read flow-map checkpoint: {exc}") from exc
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("training_summary"), Mapping
    ):
        raise ValueError("Flow-map checkpoint is missing its embedded training summary.")
    report_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"status", "checkpoint_name", "checkpoint_sha256"}
    }
    if dict(checkpoint["training_summary"]) != report_summary:
        raise ValueError(
            "Flow-map checkpoint and JSON training summaries do not match."
        )


def recover_flow_map_bundle(
    checkpoint_path: str | Path,
    summary_path: str | Path | None,
) -> None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    summary = (
        Path(summary_path).expanduser().resolve() if summary_path is not None else None
    )
    targets = _flow_map_bundle_targets(checkpoint, summary)
    recover_artifact_bundle(
        checkpoint,
        targets,
        validator=_validate_flow_map_bundle_identity,
    )


def validate_flow_map_bundle(
    checkpoint_path: str | Path,
    summary_path: str | Path | None,
) -> None:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    summary = (
        Path(summary_path).expanduser().resolve() if summary_path is not None else None
    )
    targets = _flow_map_bundle_targets(checkpoint, summary)
    validate_artifact_bundle(
        checkpoint,
        targets,
        validator=_validate_flow_map_bundle_identity,
    )


def _preflight_flow_map_bundle(
    checkpoint_path: Path,
    summary_path: Path | None,
    *,
    overwrite: bool,
) -> None:
    targets = _flow_map_bundle_targets(checkpoint_path, summary_path)
    preflight_artifact_bundle(
        checkpoint_path,
        targets,
        overwrite=overwrite,
        validator=_validate_flow_map_bundle_identity,
    )


def _write_flow_map_bundle(
    *,
    checkpoint_path: Path,
    summary_path: Path | None,
    training_summary: Mapping[str, Any],
    checkpoint_writer: Callable[[Path], Path],
    overwrite: bool = False,
    precommit_validator: Callable[[], None] | None = None,
) -> Path:
    if summary_path is not None and checkpoint_path == summary_path:
        raise ValueError("Flow-map checkpoint and training summary paths must differ.")
    targets = _flow_map_bundle_targets(checkpoint_path, summary_path)
    staged: dict[str, Path] = {}
    staged_hashes: dict[str, str] = {}
    try:
        checkpoint_temporary = temporary_bundle_path(checkpoint_path)
        staged["checkpoint"] = checkpoint_temporary
        written_path = Path(checkpoint_writer(checkpoint_temporary)).resolve()
        if written_path != checkpoint_temporary.resolve():
            raise ValueError("Flow-map checkpoint writer returned an unexpected path.")
        if summary_path is not None:
            summary_temporary = temporary_bundle_path(summary_path)
            staged["summary"] = summary_temporary
            summary_temporary.write_text(
                json.dumps(
                    {
                        **dict(training_summary),
                        "status": "completed",
                        "checkpoint_name": checkpoint_path.name,
                        "checkpoint_sha256": file_sha256(checkpoint_temporary),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        _validate_flow_map_bundle_identity(staged, targets)
        staged_hashes = {role: file_sha256(path) for role, path in staged.items()}
        promote_artifact_bundle(
            checkpoint_path,
            targets,
            staged,
            overwrite=overwrite,
            validator=_validate_flow_map_bundle_identity,
            precommit_validator=precommit_validator,
        )
        return checkpoint_path
    finally:
        cleanup_allowed = not bundle_journal_path(checkpoint_path).exists()
        for role, temporary in staged.items():
            if not cleanup_allowed:
                continue
            discard_temporary_bundle_path(
                temporary,
                targets[role],
                expected_sha256=staged_hashes.get(role),
            )


def build_argparser() -> argparse.ArgumentParser:
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
    args = build_argparser().parse_args(list(argv) if argv is not None else None)
    device = resolve_torch_device(args.device)
    backbone_path = resolve_project_path(args.backbone_checkpoint)
    gipo_path = resolve_project_path(args.gipo_checkpoint)
    manifest_path = resolve_project_path(args.demonstration_manifest)
    output_path = resolve_project_path(args.output_checkpoint)
    summary_path = (
        resolve_project_path(args.summary_json)
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
    _preflight_flow_map_bundle(
        output_path,
        summary_path,
        overwrite=bool(args.overwrite),
    )
    expected_demonstration_manifest_hash = file_sha256(manifest_path)
    store = DemonstrationStore(
        manifest_path,
        validation_fraction=args.validation_fraction,
        split_seed=args.seed,
    )
    demonstration_shard_identities = tuple(
        (
            Path(record["resolved_path"]).resolve(),
            str(record["sha256"]),
        )
        for record in [
            *store.manifest["context_shards"],
            *store.manifest["trajectory_shards"],
        ]
    )
    expected_backbone_hash = str(store.metadata.get("backbone_checkpoint_sha256", ""))
    expected_gipo_hash = str(store.metadata.get("gipo_checkpoint_sha256", ""))

    def validate_training_source_identities() -> None:
        if file_sha256(manifest_path) != expected_demonstration_manifest_hash:
            raise ValueError(
                "Demonstration manifest changed after its training identity was captured."
            )
        for shard_path, expected_hash in demonstration_shard_identities:
            if file_sha256(shard_path) != expected_hash:
                raise ValueError(
                    f"Demonstration shard {shard_path.name!r} changed during flow-map training."
                )
        if file_sha256(backbone_path) != expected_backbone_hash:
            raise ValueError(
                "Backbone source checkpoint changed after demonstration validation."
            )
        if file_sha256(gipo_path) != expected_gipo_hash:
            raise ValueError(
                "GIPO source checkpoint changed after demonstration validation."
            )

    validate_training_source_identities()
    gipo_policy = load_gipo_schedule_policy(gipo_path, device=device)
    validate_training_source_identities()
    if (
        gipo_policy.setting_encoder_config.to_payload()
        != store.setting_encoder_config.to_payload()
    ):
        raise ValueError(
            "Demonstration setting encoder is incompatible with the GIPO checkpoint."
        )
    if gipo_policy.density_dim != store.density_dim or not np.allclose(
        gipo_policy.reference_time_grid,
        store.density_reference_time_grid,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError(
            "Demonstration density representation is incompatible with the GIPO checkpoint."
        )
    demonstration_embedding_kind = validate_context_embedding_kind(
        store.metadata.get("context_embedding_kind")
    )
    if demonstration_embedding_kind != gipo_policy.context_embedding_kind:
        raise ValueError(
            "Demonstration context embedding kind is incompatible with the GIPO checkpoint."
        )
    backbone_model, _ = load_checkpoint_model(backbone_path, device)
    validate_training_source_identities()
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
    def write_checkpoint(staged_path: Path) -> Path:
        validate_training_source_identities()
        return save_flow_map_checkpoint(
            staged_path,
            flow_map,
            backbone_checkpoint=backbone_path,
            gipo_checkpoint=gipo_path,
            setting_encoder_config=store.setting_encoder_config,
            training_summary=training_summary,
            demonstration_manifest_sha256=expected_demonstration_manifest_hash,
            demonstration_metadata=store.metadata,
            expected_backbone_checkpoint_sha256=expected_backbone_hash,
            expected_gipo_checkpoint_sha256=expected_gipo_hash,
            overwrite=False,
        )

    checkpoint = _write_flow_map_bundle(
        checkpoint_path=output_path,
        summary_path=summary_path,
        training_summary=training_summary,
        checkpoint_writer=write_checkpoint,
        overwrite=bool(args.overwrite),
        precommit_validator=validate_training_source_identities,
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DemonstrationStore",
    "FlowMapBatch",
    "build_argparser",
    "main",
    "recover_flow_map_bundle",
    "train_endpoint_flow_map",
    "validate_flow_map_bundle",
]
