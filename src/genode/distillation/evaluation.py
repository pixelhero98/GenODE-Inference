from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np

from genode.checkpoint_validation import validate_strict_integer
from genode.data.otflow_paths import resolve_project_path
from genode.distillation.artifacts import (
    context_fingerprint,
    load_demonstration_manifest,
    validate_context_binding,
    write_json,
)
from genode.distillation.checkpoint import load_flow_map_checkpoint
from genode.distillation.measurement_protocol import (
    measurement_protocol_sha256 as compute_measurement_protocol_sha256,
    read_quality_measurement_protocol,
    validate_quality_measurement_protocol,
)
from genode.experiment_layout import (
    density_source_key_for_schedule,
    scenario_family_for_key,
)
from genode.gipo.models import validate_time_grid
from genode.gipo.objectives import (
    METRIC_DIRECTION_HIGHER,
    METRIC_DIRECTION_LOWER,
    teacher_objective_specs_for_scenario,
)
from genode.provenance import file_sha256
from genode.schedule_transfer.diffusion_flow_schedules import (
    EXPERIMENTAL_FIXED_SCHEDULE_KEYS,
    build_schedule_grid,
)
from genode.solver_protocol import normalize_solver_nfe_fields


VALIDATION_PHASE = "validation_tuning"
LOCKED_TEST_PHASE = "locked_test"
FLOW_MAP_METHOD = "flow_map"
GIPO_METHOD = "gipo"
FIXED_METHOD = "fixed"
SUPPORTED_METHODS = (FLOW_MAP_METHOD, GIPO_METHOD, FIXED_METHOD)
MINIMUM_PAIRED_CONTEXTS = 20
ARTIFACT_BINDING_FIELDS = (
    "scenario_key",
    "flow_map_checkpoint_sha256",
    "backbone_checkpoint_sha256",
    "gipo_checkpoint_sha256",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
QUALITY_CONTEXT_PROTOCOL = "flow_map_quality_contexts"
QUALITY_SAMPLE_PANEL_PROTOCOL = "flow_map_quality_sample_panel"
FLOW_MAP_EXECUTION_KIND = "endpoint_flow_map"
GIPO_EXECUTION_KIND = "gipo_ode_rollout"
FIXED_EXECUTION_KIND = "fixed_time_grid"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    weight: float = 1.0
    applicable_key: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        direction = str(self.direction).strip().lower()
        if not name:
            raise ValueError("Metric names may not be empty.")
        if direction not in {METRIC_DIRECTION_LOWER, METRIC_DIRECTION_HIGHER}:
            raise ValueError(
                f"Metric direction must be {METRIC_DIRECTION_LOWER!r} or "
                f"{METRIC_DIRECTION_HIGHER!r}."
            )
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Primary metric weights must be finite and positive.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "applicable_key", str(self.applicable_key).strip())

    @property
    def sign(self) -> float:
        return -1.0 if self.direction == METRIC_DIRECTION_LOWER else 1.0


@dataclass(frozen=True)
class QualityGateConfig:
    bootstrap_samples: int = 10_000
    familywise_alpha: float = 0.05
    margin: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        bootstrap_samples = validate_strict_integer(
            self.bootstrap_samples,
            label="bootstrap_samples",
            minimum=1,
        )
        bootstrap_seed = validate_strict_integer(
            self.seed,
            label="bootstrap seed",
            minimum=0,
        )
        if bootstrap_samples < 1_000:
            raise ValueError("bootstrap_samples must be at least 1,000.")
        if not 0.0 < float(self.familywise_alpha) < 1.0:
            raise ValueError("familywise_alpha must lie strictly between zero and one.")
        if not math.isfinite(float(self.margin)) or float(self.margin) != 0.0:
            raise ValueError("The release gate uses a fixed zero superiority margin.")
        object.__setattr__(self, "bootstrap_samples", bootstrap_samples)
        object.__setattr__(self, "seed", bootstrap_seed)


@dataclass(frozen=True, order=True)
class CandidateSetting:
    method: str
    candidate_key: str
    solver_key: str
    target_nfe: int
    execution_json: str

    @property
    def execution(self) -> dict[str, Any]:
        payload = json.loads(self.execution_json)
        if not isinstance(payload, dict):  # Defensive: construction is internal.
            raise ValueError("Candidate execution provenance must be an object.")
        return payload

    @property
    def execution_sha256(self) -> str:
        return hashlib.sha256(self.execution_json.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "candidate_key": self.candidate_key,
            "solver_key": self.solver_key,
            "target_nfe": int(self.target_nfe),
            "execution": self.execution,
        }


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_unique_json(text: str, *, label: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {exc}") from exc


def _reject_duplicate_npz_members(path: Path, *, label: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = [record.filename for record in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{label} is not a valid NPZ archive: {exc}") from exc
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate archive members: {sorted(duplicates)}."
        )


def _semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _time_grid_sha256(time_grid: Sequence[float]) -> str:
    return _semantic_sha256(
        {"time_grid": [round(float(value), 12) for value in time_grid]}
    )


def _normalize_candidate_execution(
    *,
    method: str,
    execution: Any,
    solver_key: str,
    target_nfe: int,
    macro_steps: int,
    index: int,
) -> dict[str, Any]:
    if not isinstance(execution, Mapping):
        raise ValueError(f"Candidate catalog entry {index} execution must be an object.")
    raw = dict(execution)
    if method == FLOW_MAP_METHOD:
        required = {"kind", "density_source"}
        if set(raw) != required:
            missing = sorted(required - set(raw))
            extra = sorted(set(raw) - required)
            raise ValueError(
                f"Flow-map candidate execution fields are invalid; missing={missing}, extra={extra}."
            )
        normalized = {
            "kind": str(raw["kind"]).strip(),
            "density_source": str(raw["density_source"]).strip(),
        }
        if normalized != {
            "kind": FLOW_MAP_EXECUTION_KIND,
            "density_source": "bound_gipo_checkpoint",
        }:
            raise ValueError(
                "Flow-map claim candidates must use the endpoint map with the bound "
                "GIPO checkpoint as their density source."
            )
        return normalized
    if method == GIPO_METHOD:
        required = {"kind", "policy_sha256"}
        if set(raw) != required:
            missing = sorted(required - set(raw))
            extra = sorted(set(raw) - required)
            raise ValueError(
                f"GIPO candidate execution fields are invalid; missing={missing}, extra={extra}."
            )
        normalized = {
            "kind": str(raw["kind"]).strip(),
            "policy_sha256": str(raw["policy_sha256"]).strip(),
        }
        if normalized["kind"] != GIPO_EXECUTION_KIND or _SHA256_PATTERN.fullmatch(
            normalized["policy_sha256"]
        ) is None:
            raise ValueError(
                "GIPO claim candidates require kind='gipo_ode_rollout' and a "
                "lowercase policy SHA-256."
            )
        return normalized

    required = {
        "kind",
        "scheduler_key",
        "density_source_key",
        "time_grid",
        "time_grid_sha256",
    }
    if set(raw) != required:
        missing = sorted(required - set(raw))
        extra = sorted(set(raw) - required)
        raise ValueError(
            f"Fixed candidate execution fields are invalid; missing={missing}, extra={extra}."
        )
    kind = str(raw["kind"]).strip()
    scheduler_key = str(raw["scheduler_key"]).strip()
    density_source_key = str(raw["density_source_key"]).strip()
    if kind != FIXED_EXECUTION_KIND or scheduler_key not in set(
        EXPERIMENTAL_FIXED_SCHEDULE_KEYS
    ):
        raise ValueError(
            "Fixed claim candidates require kind='fixed_time_grid' and a registered "
            "fixed scheduler_key."
        )
    if density_source_key != density_source_key_for_schedule(scheduler_key):
        raise ValueError(
            "Fixed candidate density_source_key does not match its registered scheduler."
        )
    if not isinstance(raw["time_grid"], Sequence) or isinstance(
        raw["time_grid"], (str, bytes)
    ):
        raise ValueError("Fixed candidate time_grid must be a numeric list.")
    time_grid = validate_time_grid(raw["time_grid"], macro_steps=int(macro_steps))
    declared_hash = str(raw["time_grid_sha256"]).strip()
    actual_hash = _time_grid_sha256(time_grid)
    if declared_hash != actual_hash:
        raise ValueError("Fixed candidate time_grid_sha256 does not match time_grid.")
    registered_grid = build_schedule_grid(scheduler_key, int(macro_steps))
    if registered_grid is None or _time_grid_sha256(registered_grid) != actual_hash:
        raise ValueError(
            "Fixed candidate time_grid does not match its registered scheduler implementation."
        )
    return {
        "kind": kind,
        "scheduler_key": scheduler_key,
        "density_source_key": density_source_key,
        "time_grid": [float(value) for value in time_grid],
        "time_grid_sha256": actual_hash,
    }


def _normalize_candidate_catalog(
    candidates: Sequence[CandidateSetting | Mapping[str, Any]],
) -> tuple[CandidateSetting, ...]:
    if not candidates:
        raise ValueError("A performance claim requires a non-empty candidate catalog.")
    normalized: list[CandidateSetting] = []
    for index, raw in enumerate(candidates):
        if isinstance(raw, CandidateSetting):
            method = raw.method
            candidate_key = raw.candidate_key
            solver_key = raw.solver_key
            target_nfe = raw.target_nfe
            execution = raw.execution
        elif isinstance(raw, Mapping):
            required = {
                "method",
                "candidate_key",
                "solver_key",
                "target_nfe",
                "execution",
            }
            if set(raw) != required:
                missing = sorted(required - set(raw))
                extra = sorted(set(raw) - required)
                raise ValueError(
                    f"Candidate catalog entry {index} has invalid fields; "
                    f"missing={missing}, extra={extra}."
                )
            method = str(raw["method"]).strip()
            candidate_key = str(raw["candidate_key"]).strip()
            solver_key = str(raw["solver_key"]).strip()
            execution = raw["execution"]
            target_nfe = validate_strict_integer(
                raw["target_nfe"],
                label=f"candidate catalog entry {index} target_nfe",
                minimum=1,
            )
        else:
            raise ValueError(f"Candidate catalog entry {index} must be an object.")
        method = str(method).strip()
        candidate_key = str(candidate_key).strip()
        solver_key = str(solver_key).strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Candidate catalog entry {index} method must be one of "
                f"{SUPPORTED_METHODS}, got {method!r}."
            )
        if not candidate_key:
            raise ValueError(f"Candidate catalog entry {index} requires candidate_key.")
        nfe = normalize_solver_nfe_fields(
            solver_key,
            validate_strict_integer(
                target_nfe,
                label=f"candidate catalog entry {index} target_nfe",
                minimum=1,
            ),
            source=f"candidate catalog entry {index}",
        )
        normalized_execution = _normalize_candidate_execution(
            method=method,
            execution=execution,
            solver_key=nfe.solver_key,
            target_nfe=nfe.target_nfe,
            macro_steps=nfe.macro_steps,
            index=index,
        )
        normalized.append(
            CandidateSetting(
                method,
                candidate_key,
                nfe.solver_key,
                nfe.target_nfe,
                _stable_json(normalized_execution),
            )
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Candidate catalog entries must be unique.")
    candidate_labels = [
        (candidate.method, candidate.candidate_key) for candidate in normalized
    ]
    if len(set(candidate_labels)) != len(candidate_labels):
        raise ValueError(
            "candidate_key values must be unique within each candidate method."
        )
    methods = {candidate.method for candidate in normalized}
    if methods != set(SUPPORTED_METHODS):
        raise ValueError(
            "Candidate catalog must contain flow_map, gipo, and fixed candidates."
        )
    candidates_by_method = {
        method: [candidate for candidate in normalized if candidate.method == method]
        for method in SUPPORTED_METHODS
    }
    for method, method_candidates in candidates_by_method.items():
        if len(method_candidates) < 2:
            raise ValueError(
                f"Candidate catalog method {method!r} requires at least two candidates."
            )
        solver_nfe_settings = {
            (candidate.solver_key, int(candidate.target_nfe))
            for candidate in method_candidates
        }
        if len(solver_nfe_settings) < 2:
            raise ValueError(
                f"Candidate catalog method {method!r} must span at least two "
                "solver/NFE settings."
            )
    fixed_scheduler_keys = {
        str(candidate.execution["scheduler_key"])
        for candidate in candidates_by_method[FIXED_METHOD]
    }
    fixed_density_sources = {
        str(candidate.execution["density_source_key"])
        for candidate in candidates_by_method[FIXED_METHOD]
    }
    if len(fixed_scheduler_keys) < 2 or len(fixed_density_sources) < 2:
        raise ValueError(
            "Fixed candidates must span at least two registered schedule and "
            "density-source families."
        )
    return tuple(sorted(normalized))


def candidate_catalog_sha256(
    candidates: Sequence[CandidateSetting | Mapping[str, Any]],
) -> str:
    payload = [
        candidate.to_payload()
        for candidate in _normalize_candidate_catalog(candidates)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_input_file_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def _require_input_files_unchanged(
    paths: Mapping[str, Path],
    expected_hashes: Mapping[str, str],
) -> None:
    if set(paths) != set(expected_hashes):
        raise ValueError("Quality-evaluation input identity fields are inconsistent.")
    for name, path in sorted(paths.items()):
        if file_sha256(path) != str(expected_hashes[name]):
            raise ValueError(
                f"Quality-evaluation input {name!r} changed while it was being read."
            )


def _quality_protocol_binding(payload: Mapping[str, Any]) -> dict[str, str]:
    required = {
        "protocol_hash",
        "scenario_key",
        "candidate_catalog_sha256",
        "quality_rows_sha256",
        "quality_contexts_sha256",
        "quality_sample_panel_sha256",
        "measurement_protocol_sha256",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            "Quality protocol binding fields are invalid; "
            f"missing={missing}, extra={extra}."
        )
    binding = {name: str(payload[name]).strip() for name in required}
    if not binding["scenario_key"]:
        raise ValueError("Quality protocol binding requires scenario_key.")
    for name in (
        "protocol_hash",
        "candidate_catalog_sha256",
        "quality_rows_sha256",
        "quality_contexts_sha256",
        "quality_sample_panel_sha256",
        "measurement_protocol_sha256",
    ):
        if _SHA256_PATTERN.fullmatch(binding[name]) is None:
            raise ValueError(
                f"Quality protocol binding {name!r} must be a lowercase SHA-256 digest."
            )
    return binding


def validate_quality_protocol(
    payload: Mapping[str, Any],
    *,
    scenario_key: str,
) -> tuple[dict[str, Any], str]:
    """Validate a complete, self-hashed full-pipeline protocol document."""

    if not isinstance(payload, Mapping):
        raise ValueError("Quality protocol JSON must contain an object.")
    document = dict(payload)
    declared_hash = str(document.pop("protocol_hash", "")).strip()
    computed_hash = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared_hash != computed_hash:
        raise ValueError("Quality protocol hash does not match its payload.")
    if str(document.get("scenario_key", "")).strip() != str(scenario_key).strip():
        raise ValueError("Quality protocol scenario does not match --scenario-key.")
    flow_map = document.get("flow_map")
    if not isinstance(flow_map, Mapping):
        raise ValueError("Quality protocol is missing flow_map settings.")
    return {**document, "protocol_hash": declared_hash}, declared_hash


def read_quality_protocol(
    path: str | Path,
    *,
    scenario_key: str,
) -> tuple[dict[str, Any], str]:
    input_path = resolve_project_path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Could not read quality protocol {input_path.name!r}: {exc}"
        ) from exc
    payload = _parse_unique_json(
        text,
        label=f"Quality protocol {input_path.name!r}",
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Quality protocol JSON must contain an object.")
    validated, digest = validate_quality_protocol(
        payload,
        scenario_key=scenario_key,
    )
    return validated, digest


def quality_protocol_binding(
    protocol: Mapping[str, Any],
    *,
    candidate_catalog: Sequence[CandidateSetting | Mapping[str, Any]],
    rows_sha256: str,
    quality_contexts_sha256: str,
    quality_sample_panel_sha256: str,
    measurement_protocol_sha256: str,
) -> dict[str, str]:
    scenario_key = str(protocol.get("scenario_key", "")).strip()
    validated_protocol, _ = validate_quality_protocol(
        protocol,
        scenario_key=scenario_key,
    )
    flow_map = validated_protocol.get("flow_map")
    if not isinstance(flow_map, Mapping):
        raise ValueError("Quality protocol is missing flow_map settings.")
    binding = _quality_protocol_binding(
        {
            "protocol_hash": validated_protocol.get("protocol_hash", ""),
            "scenario_key": validated_protocol.get("scenario_key", ""),
            "candidate_catalog_sha256": flow_map.get(
                "quality_candidate_catalog_sha256", ""
            ),
            "quality_rows_sha256": flow_map.get("quality_rows_sha256", ""),
            "quality_contexts_sha256": flow_map.get(
                "quality_contexts_sha256", ""
            ),
            "quality_sample_panel_sha256": flow_map.get(
                "quality_sample_panel_sha256", ""
            ),
            "measurement_protocol_sha256": flow_map.get(
                "quality_measurement_protocol_sha256", ""
            ),
        }
    )
    actual_catalog_hash = candidate_catalog_sha256(candidate_catalog)
    if binding["candidate_catalog_sha256"] != actual_catalog_hash:
        raise ValueError(
            "Candidate catalog does not match the bound pipeline protocol."
        )
    if binding["quality_rows_sha256"] != str(rows_sha256):
        raise ValueError("Quality rows do not match the bound pipeline protocol.")
    if binding["quality_contexts_sha256"] != str(quality_contexts_sha256):
        raise ValueError(
            "Quality contexts do not match the bound pipeline protocol."
        )
    if binding["quality_sample_panel_sha256"] != str(
        quality_sample_panel_sha256
    ):
        raise ValueError(
            "Quality sample panel does not match the bound pipeline protocol."
        )
    if binding["measurement_protocol_sha256"] != str(
        measurement_protocol_sha256
    ):
        raise ValueError(
            "Measurement protocol does not match the bound pipeline protocol."
        )
    return binding


def metric_specs_for_scenario(scenario_key: str) -> tuple[MetricSpec, ...]:
    return tuple(
        MetricSpec(spec.metric_key, spec.direction, spec.weight, spec.applicable_key)
        for spec in teacher_objective_specs_for_scenario(str(scenario_key))
    )


def _metric_spec_payloads(
    metric_specs: Sequence[MetricSpec],
) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "direction": spec.direction,
            "weight": float(spec.weight),
            "applicable_key": spec.applicable_key,
        }
        for spec in metric_specs
    ]


def _finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return number


def _finite_mean(values: Sequence[float], *, label: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} requires a non-empty finite value vector.")
    scale = float(np.max(np.abs(array)))
    if not math.isfinite(scale):
        raise ValueError(f"{label} overflowed while determining its scale.")
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(array / scale) * scale)
    if not math.isfinite(result):
        raise ValueError(f"{label} overflowed while aggregating finite values.")
    return result


def _decode_string_array(value: Any, *, label: str) -> list[str]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{label} must be a one-dimensional string array.")
    decoded = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    ]
    if any(not item or item != item.strip() for item in decoded):
        raise ValueError(f"{label} values must be non-empty trimmed strings.")
    return decoded


def validate_quality_context_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(payload)
    required = {
        "protocol",
        "artifact_sha256",
        "context_count",
        "contexts",
        "set_sha256",
    }
    if set(binding) != required:
        missing = sorted(required - set(binding))
        extra = sorted(set(binding) - required)
        raise ValueError(
            f"Quality context binding fields are invalid; missing={missing}, extra={extra}."
        )
    if binding["protocol"] != QUALITY_CONTEXT_PROTOCOL:
        raise ValueError("Unsupported quality context binding protocol.")
    if _SHA256_PATTERN.fullmatch(str(binding["artifact_sha256"])) is None:
        raise ValueError("Quality context artifact_sha256 must be a lowercase SHA-256.")
    contexts = binding["contexts"]
    if not isinstance(contexts, Sequence) or isinstance(contexts, (str, bytes)):
        raise ValueError("Quality context binding contexts must be a list.")
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(contexts):
        if not isinstance(raw, Mapping) or set(raw) != {
            "split_phase",
            "context_id",
            "context_fingerprint",
        }:
            raise ValueError(f"Quality context binding row {index} has invalid fields.")
        phase = str(raw["split_phase"]).strip()
        context_id = str(raw["context_id"]).strip()
        fingerprint = str(raw["context_fingerprint"]).strip()
        if phase not in {VALIDATION_PHASE, LOCKED_TEST_PHASE}:
            raise ValueError(f"Quality context binding row {index} has invalid split_phase.")
        if not context_id or _SHA256_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError(f"Quality context binding row {index} has invalid identity.")
        normalized.append(
            {
                "split_phase": phase,
                "context_id": context_id,
                "context_fingerprint": fingerprint,
            }
        )
    normalized = sorted(
        normalized, key=lambda row: (row["split_phase"], row["context_id"])
    )
    count = validate_strict_integer(
        binding["context_count"], label="quality context_count", minimum=1
    )
    if count != len(normalized):
        raise ValueError("Quality context binding count does not match its rows.")
    identities = [row["context_id"] for row in normalized]
    fingerprints = [row["context_fingerprint"] for row in normalized]
    if len(set(identities)) != len(identities):
        raise ValueError("Quality context ids must be unique across the evaluation panels.")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Quality physical contexts must be unique across evaluation panels.")
    if {row["split_phase"] for row in normalized} != {
        VALIDATION_PHASE,
        LOCKED_TEST_PHASE,
    }:
        raise ValueError("Quality contexts must contain validation and locked-test panels.")
    expected_set_hash = _semantic_sha256(normalized)
    if str(binding["set_sha256"]) != expected_set_hash:
        raise ValueError("Quality context binding set_sha256 does not match its rows.")
    return {
        "protocol": QUALITY_CONTEXT_PROTOCOL,
        "artifact_sha256": str(binding["artifact_sha256"]),
        "context_count": count,
        "contexts": normalized,
        "set_sha256": expected_set_hash,
    }


def read_quality_contexts(path: str | Path) -> dict[str, Any]:
    input_path = resolve_project_path(path)
    _reject_duplicate_npz_members(input_path, label="Quality context NPZ")
    try:
        with np.load(input_path, allow_pickle=False) as payload:
            files = set(payload.files)
            required = {"context_ids", "split_phases", "histories"}
            if files != required and files != required | {"conditions"}:
                raise ValueError(
                    "Quality context NPZ requires exactly context_ids, split_phases, "
                    "histories, and optional conditions."
                )
            context_ids = _decode_string_array(
                payload["context_ids"], label="quality context_ids"
            )
            split_phases = _decode_string_array(
                payload["split_phases"], label="quality split_phases"
            )
            histories = np.asarray(payload["histories"])
            conditions = (
                np.asarray(payload["conditions"])
                if "conditions" in files
                else None
            )
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"Could not read quality context NPZ {input_path.name!r}: {exc}"
        ) from exc
    count = len(context_ids)
    if len(split_phases) != count or histories.ndim < 2 or histories.shape[0] != count:
        raise ValueError("Quality context arrays have inconsistent leading dimensions.")
    if histories.dtype.kind != "f" or not bool(np.isfinite(histories).all()):
        raise ValueError("Quality histories must contain finite floating-point values.")
    if conditions is not None and (
        conditions.ndim < 2
        or conditions.shape[0] != count
        or conditions.dtype.kind != "f"
        or not bool(np.isfinite(conditions).all())
    ):
        raise ValueError("Quality conditions must be a finite floating-point row matrix.")
    contexts = [
        {
            "split_phase": split_phases[index],
            "context_id": context_ids[index],
            "context_fingerprint": context_fingerprint(
                histories[index],
                None if conditions is None else conditions[index],
            ),
        }
        for index in range(count)
    ]
    contexts = sorted(
        contexts, key=lambda row: (row["split_phase"], row["context_id"])
    )
    return validate_quality_context_binding(
        {
            "protocol": QUALITY_CONTEXT_PROTOCOL,
            "artifact_sha256": file_sha256(input_path),
            "context_count": count,
            "contexts": contexts,
            "set_sha256": _semantic_sha256(contexts),
        }
    )


def validate_quality_sample_panel_binding(
    payload: Mapping[str, Any],
    *,
    quality_context_binding: Mapping[str, Any],
) -> dict[str, Any]:
    contexts = validate_quality_context_binding(quality_context_binding)
    binding = dict(payload)
    required = {
        "protocol",
        "artifact_sha256",
        "context_count",
        "sample_count",
        "replicate_count",
        "panels",
        "set_sha256",
    }
    if set(binding) != required:
        missing = sorted(required - set(binding))
        extra = sorted(set(binding) - required)
        raise ValueError(
            f"Quality sample-panel fields are invalid; missing={missing}, extra={extra}."
        )
    if binding["protocol"] != QUALITY_SAMPLE_PANEL_PROTOCOL:
        raise ValueError("Unsupported quality sample-panel protocol.")
    if _SHA256_PATTERN.fullmatch(str(binding["artifact_sha256"])) is None:
        raise ValueError("Quality sample-panel artifact_sha256 must be lowercase SHA-256.")
    panels = binding["panels"]
    if not isinstance(panels, Sequence) or isinstance(panels, (str, bytes)):
        raise ValueError("Quality sample-panel panels must be a list.")
    normalized: list[dict[str, Any]] = []
    expected_contexts = {
        (row["split_phase"], row["context_id"]): row["context_fingerprint"]
        for row in contexts["contexts"]
    }
    for index, raw in enumerate(panels):
        if not isinstance(raw, Mapping) or set(raw) != {
            "split_phase",
            "context_id",
            "context_fingerprint",
            "sample_panel_sha256",
            "replicate_count",
        }:
            raise ValueError(f"Quality sample-panel row {index} has invalid fields.")
        phase = str(raw["split_phase"]).strip()
        context_id = str(raw["context_id"]).strip()
        fingerprint = str(raw["context_fingerprint"]).strip()
        panel_hash = str(raw["sample_panel_sha256"]).strip()
        replicate_count = validate_strict_integer(
            raw["replicate_count"],
            label=f"quality sample-panel row {index} replicate_count",
            minimum=1,
        )
        if expected_contexts.get((phase, context_id)) != fingerprint:
            raise ValueError("Quality sample-panel row does not match its physical context.")
        if _SHA256_PATTERN.fullmatch(panel_hash) is None:
            raise ValueError("Quality sample-panel row hash must be lowercase SHA-256.")
        normalized.append(
            {
                "split_phase": phase,
                "context_id": context_id,
                "context_fingerprint": fingerprint,
                "sample_panel_sha256": panel_hash,
                "replicate_count": replicate_count,
            }
        )
    normalized = sorted(
        normalized, key=lambda row: (row["split_phase"], row["context_id"])
    )
    if {
        (row["split_phase"], row["context_id"]) for row in normalized
    } != set(expected_contexts):
        raise ValueError("Quality sample panels must exactly cover the quality contexts.")
    replicate_count = validate_strict_integer(
        binding["replicate_count"], label="quality replicate_count", minimum=1
    )
    if any(row["replicate_count"] != replicate_count for row in normalized):
        raise ValueError("Every quality context must use the same replicate count.")
    context_count = validate_strict_integer(
        binding["context_count"], label="quality sample context_count", minimum=1
    )
    sample_count = validate_strict_integer(
        binding["sample_count"], label="quality sample_count", minimum=1
    )
    if context_count != len(normalized) or sample_count != context_count * replicate_count:
        raise ValueError("Quality sample-panel counts are inconsistent.")
    expected_set_hash = _semantic_sha256(normalized)
    if str(binding["set_sha256"]) != expected_set_hash:
        raise ValueError("Quality sample-panel set_sha256 does not match its rows.")
    return {
        "protocol": QUALITY_SAMPLE_PANEL_PROTOCOL,
        "artifact_sha256": str(binding["artifact_sha256"]),
        "context_count": context_count,
        "sample_count": sample_count,
        "replicate_count": replicate_count,
        "panels": normalized,
        "set_sha256": expected_set_hash,
    }


def read_quality_sample_panel(
    path: str | Path,
    *,
    quality_context_binding: Mapping[str, Any],
) -> dict[str, Any]:
    input_path = resolve_project_path(path)
    _reject_duplicate_npz_members(input_path, label="Quality sample-panel NPZ")
    contexts = validate_quality_context_binding(quality_context_binding)
    context_by_id = {
        row["context_id"]: row for row in contexts["contexts"]
    }
    try:
        with np.load(input_path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "context_ids",
                "logical_seeds",
                "initial_states",
            }:
                raise ValueError(
                    "Quality sample-panel NPZ requires exactly context_ids, "
                    "logical_seeds, and initial_states."
                )
            context_ids = _decode_string_array(
                payload["context_ids"], label="quality sample context_ids"
            )
            logical_seeds = np.asarray(payload["logical_seeds"])
            initial_states = np.asarray(payload["initial_states"])
    except (OSError, EOFError) as exc:
        raise ValueError(
            f"Could not read quality sample-panel NPZ {input_path.name!r}: {exc}"
        ) from exc
    sample_count = len(context_ids)
    if (
        logical_seeds.ndim != 1
        or logical_seeds.shape[0] != sample_count
        or logical_seeds.dtype.kind not in {"i", "u"}
        or initial_states.ndim < 2
        or initial_states.shape[0] != sample_count
    ):
        raise ValueError("Quality sample-panel arrays have inconsistent shapes or dtypes.")
    if initial_states.dtype.kind != "f" or not bool(
        np.isfinite(initial_states).all()
    ):
        raise ValueError("Quality initial_states must be finite floating-point values.")
    samples_by_context: dict[str, list[dict[str, Any]]] = {
        context_id: [] for context_id in context_by_id
    }
    seen_seeds: set[tuple[str, int]] = set()
    for index, (context_id, raw_seed) in enumerate(zip(context_ids, logical_seeds.tolist())):
        if context_id not in context_by_id:
            raise ValueError("Quality sample panel references an unknown context_id.")
        seed = validate_strict_integer(
            raw_seed, label=f"quality logical seed {index}", minimum=0
        )
        if (context_id, seed) in seen_seeds:
            raise ValueError("Quality logical seeds must be unique within each context.")
        seen_seeds.add((context_id, seed))
        state_hash = context_fingerprint(initial_states[index], None)
        samples_by_context[context_id].append(
            {
                "logical_seed": seed,
                "initial_state_sha256": state_hash,
            }
        )
    panels: list[dict[str, Any]] = []
    replicate_counts: set[int] = set()
    for context_id, context in context_by_id.items():
        samples = sorted(
            samples_by_context[context_id],
            key=lambda sample: (sample["logical_seed"], sample["initial_state_sha256"]),
        )
        if not samples:
            raise ValueError("Every quality context requires at least one sample.")
        replicate_counts.add(len(samples))
        panels.append(
            {
                **context,
                "sample_panel_sha256": _semantic_sha256(
                    {
                        "context_fingerprint": context["context_fingerprint"],
                        "samples": samples,
                    }
                ),
                "replicate_count": len(samples),
            }
        )
    if len(replicate_counts) != 1:
        raise ValueError("Every quality context must use the same replicate count.")
    panels = sorted(panels, key=lambda row: (row["split_phase"], row["context_id"]))
    replicate_count = next(iter(replicate_counts))
    return validate_quality_sample_panel_binding(
        {
            "protocol": QUALITY_SAMPLE_PANEL_PROTOCOL,
            "artifact_sha256": file_sha256(input_path),
            "context_count": len(panels),
            "sample_count": sample_count,
            "replicate_count": replicate_count,
            "panels": panels,
            "set_sha256": _semantic_sha256(panels),
        },
        quality_context_binding=contexts,
    )


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    artifact_binding: Mapping[str, str],
    candidate_catalog: Sequence[CandidateSetting],
    quality_context_binding: Mapping[str, Any],
    quality_sample_panel_binding: Mapping[str, Any],
    measurement_protocol_sha256: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Quality-gate evaluation requires non-empty metric rows.")
    names = [spec.name for spec in metric_specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("Primary metric specifications must be non-empty and unique.")
    context_binding = validate_quality_context_binding(quality_context_binding)
    sample_binding = validate_quality_sample_panel_binding(
        quality_sample_panel_binding,
        quality_context_binding=context_binding,
    )
    expected_contexts = {
        (row["split_phase"], row["context_id"]): row["context_fingerprint"]
        for row in context_binding["contexts"]
    }
    expected_panels = {
        (row["split_phase"], row["context_id"]): row
        for row in sample_binding["panels"]
    }
    candidates_by_setting = {
        (
            candidate.method,
            candidate.candidate_key,
            candidate.solver_key,
            int(candidate.target_nfe),
        ): candidate
        for candidate in candidate_catalog
    }
    if len(candidates_by_setting) != len(candidate_catalog):
        raise ValueError("Candidate catalog settings must be unique.")
    normalized: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    context_fingerprint_by_key: dict[tuple[str, str], str] = {}
    context_key_by_fingerprint: dict[tuple[str, str], str] = {}
    for row_index, raw in enumerate(rows):
        row = dict(raw)
        phase = str(row.get("split_phase", "")).strip()
        if phase not in {VALIDATION_PHASE, LOCKED_TEST_PHASE}:
            raise ValueError(
                f"Quality row {row_index} split_phase must be validation_tuning or locked_test."
            )
        for field, expected in artifact_binding.items():
            if str(row.get(field, "")).strip() != str(expected):
                raise ValueError(
                    f"Quality row {row_index} {field!r} does not match the evaluated artifact."
                )
        method = str(row.get("method", "")).strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Quality row {row_index} method must be one of {SUPPORTED_METHODS}, got {method!r}."
            )
        candidate_key = str(row.get("candidate_key", "")).strip()
        context_id = str(row.get("context_id", "")).strip()
        if not candidate_key or not context_id:
            raise ValueError(f"Quality row {row_index} requires candidate_key and context_id.")
        context_fingerprint = str(row.get("context_fingerprint", "")).strip()
        if _SHA256_PATTERN.fullmatch(context_fingerprint) is None:
            raise ValueError(
                f"Quality row {row_index} context_fingerprint must be a lowercase "
                "SHA-256 digest of the physical context values."
            )
        context_key = (phase, context_id)
        expected_fingerprint = expected_contexts.get(context_key)
        if expected_fingerprint is None:
            raise ValueError(
                f"Quality row {row_index} references a context absent from the bound context artifact."
            )
        if context_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Quality row {row_index} context_fingerprint does not match the bound physical values."
            )
        previous_fingerprint = context_fingerprint_by_key.setdefault(
            context_key, context_fingerprint
        )
        if previous_fingerprint != context_fingerprint:
            raise ValueError(
                "A quality context_id may not identify multiple physical contexts."
            )
        fingerprint_key = (phase, context_fingerprint)
        previous_context_id = context_key_by_fingerprint.setdefault(
            fingerprint_key, context_id
        )
        if previous_context_id != context_id:
            raise ValueError(
                "A physical quality context may not be duplicated under multiple context_ids."
            )
        nfe = normalize_solver_nfe_fields(
            str(row.get("solver_key", "")),
            validate_strict_integer(
                row.get("target_nfe"),
                label=f"quality row {row_index} target_nfe",
                minimum=1,
            ),
            source=f"quality row {row_index}",
        )
        candidate = candidates_by_setting.get(
            (method, candidate_key, nfe.solver_key, nfe.target_nfe)
        )
        if candidate is None:
            raise ValueError(
                f"Quality row {row_index} does not identify a prespecified candidate."
            )
        execution_hash = str(row.get("candidate_execution_sha256", "")).strip()
        if execution_hash != candidate.execution_sha256:
            raise ValueError(
                f"Quality row {row_index} candidate execution does not match the catalog."
            )
        row_measurement_protocol_sha256 = str(
            row.get("measurement_protocol_sha256", "")
        ).strip()
        if row_measurement_protocol_sha256 != measurement_protocol_sha256:
            raise ValueError(
                f"Quality row {row_index} measurement_protocol_sha256 does not "
                "match the bound external measurement protocol."
            )
        expected_panel = expected_panels[context_key]
        sample_panel_hash = str(row.get("sample_panel_sha256", "")).strip()
        if sample_panel_hash != expected_panel["sample_panel_sha256"]:
            raise ValueError(
                f"Quality row {row_index} does not use the bound common sample panel."
            )
        replicate_count = validate_strict_integer(
            row.get("replicate_count"),
            label=f"quality row {row_index} replicate_count",
            minimum=1,
        )
        if replicate_count != int(expected_panel["replicate_count"]):
            raise ValueError(
                f"Quality row {row_index} replicate_count does not match the sample panel."
            )
        model_evaluations = validate_strict_integer(
            row.get("model_evaluations"),
            label=f"quality row {row_index} model_evaluations",
            minimum=1,
        )
        expected_evaluations = 1 if method == FLOW_MAP_METHOD else nfe.target_nfe
        if model_evaluations != expected_evaluations:
            raise ValueError(
                f"Quality row {row_index} model_evaluations must equal "
                f"{expected_evaluations} for method {method!r}."
            )
        row_key = (phase, method, candidate_key, context_id)
        if row_key in seen_rows:
            raise ValueError(
                "Quality rows must be unique by split, method, candidate, and context."
            )
        seen_rows.add(row_key)
        applicability: dict[str, bool] = {}
        for spec in metric_specs:
            if not spec.applicable_key:
                applicability[spec.name] = True
                continue
            value = str(row.get(spec.applicable_key, "")).strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(
                    f"Quality row {row_index} applicability field {spec.applicable_key!r} "
                    "must be true or false."
                )
            applicability[spec.name] = value in {"true", "1"}
        metric_values = {
            spec.name: (
                _finite_float(
                    row.get(spec.name),
                    label=f"quality row {row_index} metric {spec.name!r}",
                )
                if applicability[spec.name]
                else None
            )
            for spec in metric_specs
        }
        normalized.append(
            {
                "split_phase": phase,
                "method": method,
                "candidate_key": candidate_key,
                "solver_key": nfe.solver_key,
                "target_nfe": nfe.target_nfe,
                "candidate": candidate,
                "candidate_execution_sha256": execution_hash,
                "context_id": context_id,
                "context_fingerprint": context_fingerprint,
                "measurement_protocol_sha256": row_measurement_protocol_sha256,
                "sample_panel_sha256": sample_panel_hash,
                "replicate_count": replicate_count,
                "model_evaluations": model_evaluations,
                "metric_applicability": applicability,
                **metric_values,
            }
        )
    phases = {row["split_phase"] for row in normalized}
    if phases != {VALIDATION_PHASE, LOCKED_TEST_PHASE}:
        raise ValueError("Quality-gate rows must include validation_tuning and locked_test phases.")
    observed_contexts = {
        (str(row["split_phase"]), str(row["context_id"])) for row in normalized
    }
    if observed_contexts != set(expected_contexts):
        raise ValueError(
            "Quality rows must exactly cover the bound validation and locked-test contexts."
        )
    validation_contexts = {
        str(row["context_id"])
        for row in normalized
        if row["split_phase"] == VALIDATION_PHASE
    }
    locked_contexts = {
        str(row["context_id"])
        for row in normalized
        if row["split_phase"] == LOCKED_TEST_PHASE
    }
    if validation_contexts & locked_contexts:
        raise ValueError("Validation and locked-test context panels must be disjoint.")
    validation_fingerprints = {
        str(row["context_fingerprint"])
        for row in normalized
        if row["split_phase"] == VALIDATION_PHASE
    }
    locked_fingerprints = {
        str(row["context_fingerprint"])
        for row in normalized
        if row["split_phase"] == LOCKED_TEST_PHASE
    }
    if validation_fingerprints & locked_fingerprints:
        raise ValueError(
            "Validation and locked-test physical context panels must be disjoint."
        )
    return normalized


def _candidate(row: Mapping[str, Any]) -> CandidateSetting:
    candidate = row.get("candidate")
    if not isinstance(candidate, CandidateSetting):
        raise ValueError("Normalized quality rows are missing candidate provenance.")
    return candidate


def _select_on_validation(
    rows: Sequence[Mapping[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    method: str,
) -> tuple[CandidateSetting, dict[str, Any]]:
    validation = [
        row
        for row in rows
        if row["split_phase"] == VALIDATION_PHASE and row["method"] == method
    ]
    candidates = sorted({_candidate(row) for row in validation})
    if not candidates:
        raise ValueError(f"Validation rows do not contain a {method!r} candidate.")
    context_sets = {
        candidate: {
            str(row["context_fingerprint"])
            for row in validation
            if _candidate(row) == candidate
        }
        for candidate in candidates
    }
    reference_contexts = context_sets[candidates[0]]
    incomplete = [
        candidate
        for candidate in candidates[1:]
        if context_sets[candidate] != reference_contexts
    ]
    if incomplete:
        raise ValueError(
            f"Validation selection for method {method!r} requires identical context coverage "
            "for every candidate setting."
        )
    means: dict[CandidateSetting, dict[str, float]] = {}
    for candidate in candidates:
        selected_rows = [row for row in validation if _candidate(row) == candidate]
        means[candidate] = {}
        for spec in metric_specs:
            applicable_rows = [
                row
                for row in selected_rows
                if bool(row["metric_applicability"][spec.name])
            ]
            if not applicable_rows:
                raise ValueError(
                    f"Validation candidate {candidate.candidate_key!r} has no applicable "
                    f"rows for primary metric {spec.name!r}."
                )
            means[candidate][spec.name] = _finite_mean(
                [float(row[spec.name]) for row in applicable_rows],
                label=(
                    f"Validation mean for {candidate.candidate_key!r} metric "
                    f"{spec.name!r}"
                ),
            )
    for spec in metric_specs:
        metric_context_sets = {
            candidate: {
                str(row["context_fingerprint"])
                for row in validation
                if _candidate(row) == candidate
                and bool(row["metric_applicability"][spec.name])
            }
            for candidate in candidates
        }
        if len({frozenset(panel) for panel in metric_context_sets.values()}) != 1:
            raise ValueError(
                f"Validation applicability coverage for metric {spec.name!r} differs across candidates."
            )
    total_weight = float(sum(spec.weight for spec in metric_specs))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Primary metric weights produced a non-finite total.")
    component_weights = {
        spec.name: float(spec.weight) / total_weight for spec in metric_specs
    }
    oriented_values = {
        spec.name: {
            candidate: float(spec.sign * means[candidate][spec.name])
            for candidate in candidates
        }
        for spec in metric_specs
    }
    component_ranges: dict[str, dict[str, float]] = {}
    component_values: dict[CandidateSetting, dict[str, float]] = {
        candidate: {} for candidate in candidates
    }
    for spec in metric_specs:
        values = oriented_values[spec.name]
        lower = min(values.values())
        upper = max(values.values())
        with np.errstate(over="ignore", invalid="ignore"):
            span = float(np.subtract(upper, lower))
        if not all(math.isfinite(value) for value in (lower, upper, span)):
            raise ValueError(
                f"Validation range for metric {spec.name!r} overflowed."
            )
        component_ranges[spec.name] = {
            "oriented_min": float(lower),
            "oriented_max": float(upper),
            "span": float(span),
        }
        for candidate, value in values.items():
            component = (
                0.0 if span <= 1e-12 else float((value - lower) / span)
            )
            if not math.isfinite(component):
                raise ValueError(
                    f"Validation component for metric {spec.name!r} is non-finite."
                )
            component_values[candidate][spec.name] = component
    utility_means = {
        candidate: float(
            sum(
                component_weights[spec.name]
                * component_values[candidate][spec.name]
                for spec in metric_specs
            )
        )
        for candidate in candidates
    }
    if not all(math.isfinite(value) for value in utility_means.values()):
        raise ValueError("Validation selection produced a non-finite utility.")
    best_utility = max(utility_means.values())
    winners = [
        candidate
        for candidate, value in utility_means.items()
        if math.isclose(value, best_utility, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if len(winners) != 1:
        raise ValueError(
            f"Validation selection for method {method!r} is tied; provide a prespecified "
            "weighted primary-metric utility that uniquely freezes one candidate."
        )
    selected = winners[0]
    return selected, {
        "selection_split": VALIDATION_PHASE,
        "selection_utility": utility_means[selected],
        "selection_protocol": "weighted_normalized_primary_metric_utility",
        "selection_component_values": component_values[selected],
        "selection_component_weights": component_weights,
        "selection_component_ranges": component_ranges,
        "primary_metric_means": means[selected],
        "candidate_count": len(candidates),
        "locked_test_used_for_selection": False,
    }


def _paired_differences(
    rows: Sequence[Mapping[str, Any]],
    *,
    flow_map: CandidateSetting,
    comparator: CandidateSetting,
    metric: MetricSpec,
    margin: float,
) -> tuple[np.ndarray, list[str]]:
    def context_means(candidate: CandidateSetting) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            if row["split_phase"] != LOCKED_TEST_PHASE or _candidate(row) != candidate:
                continue
            if not bool(row["metric_applicability"][metric.name]):
                continue
            grouped.setdefault(str(row["context_fingerprint"]), []).append(
                float(row[metric.name])
            )
        return {
            fingerprint: _finite_mean(
                values,
                label=(
                    f"Locked-test mean for {candidate.candidate_key!r} metric "
                    f"{metric.name!r}"
                ),
            )
            for fingerprint, values in grouped.items()
        }

    flow_values = context_means(flow_map)
    comparator_values = context_means(comparator)
    if set(flow_values) != set(comparator_values):
        raise ValueError(
            f"Metric {metric.name!r} requires identical locked-test context coverage for "
            f"{flow_map.candidate_key!r} and {comparator.candidate_key!r}."
        )
    common = sorted(flow_values)
    if len(common) < MINIMUM_PAIRED_CONTEXTS:
        raise ValueError(
            f"Metric {metric.name!r} has fewer than {MINIMUM_PAIRED_CONTEXTS} paired "
            "locked-test contexts for "
            f"{flow_map.candidate_key!r} versus {comparator.candidate_key!r}."
        )
    flow_array = np.asarray([flow_values[context_id] for context_id in common], dtype=np.float64)
    comparator_array = np.asarray(
        [comparator_values[context_id] for context_id in common], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        differences = metric.sign * np.subtract(flow_array, comparator_array) - float(margin)
    if not bool(np.isfinite(differences).all()):
        raise ValueError(
            f"Metric {metric.name!r} produced non-finite paired differences; "
            "the supplied finite values overflow during comparison."
        )
    return differences, common


def _bootstrap_test(
    differences: np.ndarray,
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    if differences.ndim != 1 or differences.size == 0 or not bool(
        np.isfinite(differences).all()
    ):
        raise ValueError("Bootstrap differences must be a non-empty finite vector.")
    observed = float(differences.mean())
    centered = differences - observed
    if not math.isfinite(observed) or not bool(np.isfinite(centered).all()):
        raise ValueError("Bootstrap centering produced non-finite values.")
    generator = np.random.default_rng(int(seed))
    remaining = int(samples)
    null_exceedances = 0
    boot_means: list[np.ndarray] = []
    chunk_size = 1_000
    while remaining > 0:
        count = min(chunk_size, remaining)
        indices = generator.integers(0, differences.size, size=(count, differences.size))
        null_means = centered[indices].mean(axis=1)
        raw_means = differences[indices].mean(axis=1)
        if not bool(np.isfinite(null_means).all()) or not bool(
            np.isfinite(raw_means).all()
        ):
            raise ValueError("Bootstrap resampling produced non-finite means.")
        null_exceedances += int(np.count_nonzero(null_means >= observed))
        boot_means.append(raw_means)
        remaining -= count
    all_means = np.concatenate(boot_means)
    p_value = float((null_exceedances + 1) / (int(samples) + 1))
    return {
        "mean_difference": observed,
        "one_sided_p_value": p_value,
        "one_sided_lower_bound": float(np.quantile(all_means, float(alpha))),
    }


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def evaluate_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_specs: Sequence[MetricSpec],
    candidate_catalog: Sequence[CandidateSetting | Mapping[str, Any]],
    artifact_binding: Mapping[str, str],
    demonstration_context_binding: Mapping[str, Any],
    quality_context_binding: Mapping[str, Any],
    quality_sample_panel_binding: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
    quality_rows_sha256: str,
    measurement_protocol: Mapping[str, Any],
    measurement_protocol_sha256: str,
    config: QualityGateConfig | None = None,
) -> dict[str, Any]:
    """Select settings on validation, then gate every locked-test primary metric.

    The zero-margin test is intentionally conservative: a result must provide
    evidence that the flow map is better than both independently selected
    comparators. Merely failing to detect a difference does not pass.
    """

    gate_config = config or QualityGateConfig()
    binding = {field: str(artifact_binding.get(field, "")).strip() for field in ARTIFACT_BINDING_FIELDS}
    if not binding["scenario_key"]:
        raise ValueError("artifact_binding requires a scenario_key.")
    registered_specs = metric_specs_for_scenario(binding["scenario_key"])
    specs = tuple(metric_specs)
    if specs != registered_specs:
        raise ValueError(
            "metric_specs must exactly match the registered primary metric "
            f"specification for scenario {binding['scenario_key']!r}; custom metric "
            "sets cannot support a performance claim."
        )
    for field in ARTIFACT_BINDING_FIELDS[1:]:
        value = binding[field]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"artifact_binding {field!r} must be a lowercase SHA-256 digest.")
    catalog = _normalize_candidate_catalog(candidate_catalog)
    for candidate in catalog:
        if (
            candidate.method == GIPO_METHOD
            and candidate.execution["policy_sha256"]
            != binding["gipo_checkpoint_sha256"]
        ):
            raise ValueError(
                "GIPO candidate policy provenance does not match the bound checkpoint."
            )
    context_binding_payload = validate_quality_context_binding(
        quality_context_binding
    )
    sample_binding_payload = validate_quality_sample_panel_binding(
        quality_sample_panel_binding,
        quality_context_binding=context_binding_payload,
    )
    catalog_hash = candidate_catalog_sha256(catalog)
    measurement_digest = str(measurement_protocol_sha256).strip()
    if _SHA256_PATTERN.fullmatch(measurement_digest) is None:
        raise ValueError(
            "measurement_protocol_sha256 must be a lowercase SHA-256 digest."
        )
    protocol_binding = quality_protocol_binding(
        quality_protocol,
        candidate_catalog=catalog,
        rows_sha256=quality_rows_sha256,
        quality_contexts_sha256=context_binding_payload["artifact_sha256"],
        quality_sample_panel_sha256=sample_binding_payload["artifact_sha256"],
        measurement_protocol_sha256=measurement_digest,
    )
    if protocol_binding["scenario_key"] != binding["scenario_key"]:
        raise ValueError(
            "Quality protocol scenario does not match the evaluated artifact."
        )
    validated_measurement_protocol = validate_quality_measurement_protocol(
        measurement_protocol,
        scenario_key=binding["scenario_key"],
        candidate_catalog_sha256=catalog_hash,
        quality_contexts_sha256=context_binding_payload["artifact_sha256"],
        quality_sample_panel_sha256=sample_binding_payload["artifact_sha256"],
        artifact_binding={
            field: binding[field] for field in ARTIFACT_BINDING_FIELDS[1:]
        },
        primary_metrics=_metric_spec_payloads(specs),
        bootstrap_samples=gate_config.bootstrap_samples,
        bootstrap_seed=gate_config.seed,
        familywise_alpha=gate_config.familywise_alpha,
    )
    if (
        compute_measurement_protocol_sha256(validated_measurement_protocol)
        != measurement_digest
    ):
        raise ValueError(
            "measurement_protocol_sha256 does not match the validated protocol payload."
        )
    normalized = _normalize_rows(
        rows,
        specs,
        artifact_binding=binding,
        candidate_catalog=catalog,
        quality_context_binding=context_binding_payload,
        quality_sample_panel_binding=sample_binding_payload,
        measurement_protocol_sha256=measurement_digest,
    )
    observed_catalog = tuple(
        sorted({_candidate(row) for row in normalized if row["split_phase"] == VALIDATION_PHASE})
    )
    if observed_catalog != catalog:
        expected = set(catalog)
        observed = set(observed_catalog)
        missing = [item.to_payload() for item in sorted(expected - observed)]
        extra = [item.to_payload() for item in sorted(observed - expected)]
        raise ValueError(
            "Validation rows must exactly cover the prespecified candidate catalog; "
            f"missing={missing}, extra={extra}."
        )
    demonstration_binding = validate_context_binding(demonstration_context_binding)
    evaluation_context_fingerprints = {
        str(row["context_fingerprint"]) for row in normalized
    }
    overlap = evaluation_context_fingerprints & set(
        demonstration_binding["context_fingerprints"]
    )
    if overlap:
        raise ValueError(
            "Quality validation and locked-test contexts must be disjoint from every "
            "demonstration context."
        )
    validation_panels = {
        method: {
            str(row["context_fingerprint"])
            for row in normalized
            if row["split_phase"] == VALIDATION_PHASE and row["method"] == method
        }
        for method in SUPPORTED_METHODS
    }
    if len({frozenset(panel) for panel in validation_panels.values()}) != 1:
        raise ValueError("Every method must use the same validation context panel.")
    for phase in (VALIDATION_PHASE, LOCKED_TEST_PHASE):
        for spec in specs:
            flags_by_context: dict[str, set[bool]] = {}
            for row in normalized:
                if row["split_phase"] != phase:
                    continue
                flags_by_context.setdefault(
                    str(row["context_fingerprint"]), set()
                ).add(
                    bool(row["metric_applicability"][spec.name])
                )
            if any(len(flags) != 1 for flags in flags_by_context.values()):
                raise ValueError(
                    f"Metric {spec.name!r} applicability must be a shared context "
                    f"property across all methods and candidates in {phase!r}."
                )
    selections: dict[str, CandidateSetting] = {}
    selection_metadata: dict[str, Any] = {}
    for method in SUPPORTED_METHODS:
        selection, metadata = _select_on_validation(normalized, specs, method=method)
        selections[method] = selection
        selection_metadata[method] = {**selection.to_payload(), **metadata}

    locked_panel = {
        str(row["context_fingerprint"])
        for row in context_binding_payload["contexts"]
        if row["split_phase"] == LOCKED_TEST_PHASE
    }
    for method, selection in selections.items():
        candidate_panel = {
            str(row["context_fingerprint"])
            for row in normalized
            if row["split_phase"] == LOCKED_TEST_PHASE
            and _candidate(row) == selection
        }
        if candidate_panel != locked_panel:
            raise ValueError(
                "Every validation-selected candidate must exactly cover the entire "
                f"bound locked-test physical context panel; method={method!r}, "
                f"missing={len(locked_panel - candidate_panel)}, "
                f"extra={len(candidate_panel - locked_panel)}."
            )
    for spec in specs:
        selected_applicable_panels = {
            method: {
                str(row["context_fingerprint"])
                for row in normalized
                if row["split_phase"] == LOCKED_TEST_PHASE
                and _candidate(row) == selection
                and bool(row["metric_applicability"][spec.name])
            }
            for method, selection in selections.items()
        }
        if len(
            {frozenset(panel) for panel in selected_applicable_panels.values()}
        ) != 1:
            raise ValueError(
                f"Validation-selected candidates must share locked-test applicability "
                f"for metric {spec.name!r}."
            )

    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for comparator_method in (GIPO_METHOD, FIXED_METHOD):
        for metric_index, spec in enumerate(specs):
            differences, context_ids = _paired_differences(
                normalized,
                flow_map=selections[FLOW_MAP_METHOD],
                comparator=selections[comparator_method],
                metric=spec,
                margin=gate_config.margin,
            )
            result = _bootstrap_test(
                differences,
                samples=gate_config.bootstrap_samples,
                alpha=gate_config.familywise_alpha,
                seed=int(gate_config.seed) + len(comparisons) * 104_729 + metric_index,
            )
            raw_p_values.append(result["one_sided_p_value"])
            comparisons.append(
                {
                    "comparator_method": comparator_method,
                    "metric": spec.name,
                    "direction": spec.direction,
                    "paired_context_count": len(context_ids),
                    "margin": float(gate_config.margin),
                    **result,
                }
            )
    adjusted = _holm_adjust(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted):
        comparison["holm_adjusted_p_value"] = float(adjusted_p)
        comparison["passed"] = bool(
            float(comparison["mean_difference"]) >= 0.0
            and float(adjusted_p) <= float(gate_config.familywise_alpha)
        )
    passed = all(bool(comparison["passed"]) for comparison in comparisons)
    return {
        "status": "passed" if passed else "failed",
        "performance_claim": bool(passed),
        "protocol": "validation_frozen_paired_bootstrap",
        "artifact_binding": {
            **binding,
            "demonstration_context_set_sha256": demonstration_binding["set_sha256"],
            "quality_context_set_sha256": context_binding_payload["set_sha256"],
            "quality_sample_panel_set_sha256": sample_binding_payload["set_sha256"],
        },
        "selection": selection_metadata,
        "candidate_catalog": [candidate.to_payload() for candidate in catalog],
        "candidate_catalog_sha256": catalog_hash,
        "quality_protocol_hash": protocol_binding["protocol_hash"],
        "quality_rows_sha256": protocol_binding["quality_rows_sha256"],
        "quality_contexts_sha256": protocol_binding["quality_contexts_sha256"],
        "quality_sample_panel_sha256": protocol_binding[
            "quality_sample_panel_sha256"
        ],
        "measurement_protocol_sha256": measurement_digest,
        "measurement_protocol": validated_measurement_protocol,
        "replicate_count": int(sample_binding_payload["replicate_count"]),
        "primary_metrics": _metric_spec_payloads(specs),
        "comparisons": comparisons,
        "bootstrap_samples": int(gate_config.bootstrap_samples),
        "bootstrap_seed": int(gate_config.seed),
        "familywise_alpha": float(gate_config.familywise_alpha),
        "multiple_testing_correction": "holm",
        "margin": float(gate_config.margin),
        "all_primary_metrics_required": True,
        "minimum_paired_contexts": MINIMUM_PAIRED_CONTEXTS,
        "locked_test_used_for_selection": False,
    }


def not_evaluated_report(
    *,
    reason: str,
    artifact_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    text = str(reason).strip()
    if not text:
        raise ValueError("A not-evaluated report requires a reason.")
    report = {
        "status": "not_evaluated",
        "protocol": "validation_frozen_paired_bootstrap",
        "reason": text,
        "performance_claim": False,
        "locked_test_used_for_selection": False,
    }
    if artifact_binding is not None:
        report["artifact_binding"] = dict(artifact_binding)
    return report


def read_quality_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = resolve_project_path(path)
    try:
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("Quality rows CSV requires a header row.")
            if any(not name or name != name.strip() for name in fieldnames):
                raise ValueError(
                    "Quality rows CSV headers must be non-empty and have no surrounding whitespace."
                )
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError("Quality rows CSV headers must be unique.")
            rows = []
            for index, raw in enumerate(reader):
                if None in raw:
                    raise ValueError(
                        f"Quality row {index} contains values beyond the declared headers."
                    )
                if any(raw[name] is None for name in fieldnames):
                    raise ValueError(
                        f"Quality row {index} has fewer values than the declared headers."
                    )
                rows.append(dict(raw))
    except OSError as exc:
        raise ValueError(f"Could not read quality rows {input_path.name!r}: {exc}") from exc
    for index, row in enumerate(rows):
        for field in ("target_nfe", "replicate_count", "model_evaluations"):
            raw_value = row.get(field)
            if not isinstance(raw_value, str) or re.fullmatch(
                r"[1-9][0-9]*", raw_value
            ) is None:
                raise ValueError(
                    f"Quality row {index} {field} must be a plain positive decimal integer."
                )
            row[field] = int(raw_value)
    return rows


def read_candidate_catalog(path: str | Path) -> tuple[CandidateSetting, ...]:
    input_path = resolve_project_path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Could not read candidate catalog {input_path.name!r}: {exc}"
        ) from exc
    payload = _parse_unique_json(
        text,
        label=f"Candidate catalog {input_path.name!r}",
    )
    if not isinstance(payload, list):
        raise ValueError("Candidate catalog JSON must contain a list of candidate objects.")
    return _normalize_candidate_catalog(payload)


def _metric_specs_from_json(text: str) -> tuple[MetricSpec, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--metrics-json is invalid JSON: {exc}") from exc
    if isinstance(payload, Mapping):
        return tuple(MetricSpec(str(name), str(direction)) for name, direction in payload.items())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if not all(isinstance(item, Mapping) for item in payload):
            raise ValueError("--metrics-json metric lists may contain only metric objects.")
        return tuple(
            MetricSpec(
                str(item["name"]),
                str(item["direction"]),
                float(item.get("weight", 1.0)),
                str(item.get("applicable_key", "")),
            )
            for item in payload
        )
    raise ValueError("--metrics-json must be an object or a list of metric objects.")


def _resolve_metric_specs(
    scenario_key: str,
    metrics_json: str,
) -> tuple[MetricSpec, ...]:
    """Return the registered claim metrics, rejecting cherry-picked overrides."""
    registered = metric_specs_for_scenario(scenario_key)
    if not str(metrics_json).strip():
        return registered
    requested = _metric_specs_from_json(metrics_json)
    if requested != registered:
        raise ValueError(
            "--metrics-json must exactly match the registered primary metric "
            f"specification for scenario {scenario_key!r}; custom metric sets "
            "cannot support a performance claim."
        )
    return registered


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the validation-frozen, familywise flow-map quality gate."
    )
    parser.add_argument("--rows-csv", default="")
    parser.add_argument("--candidate-catalog", default="")
    parser.add_argument("--quality-contexts-npz", default="")
    parser.add_argument("--quality-sample-panel-npz", default="")
    parser.add_argument("--measurement-protocol-json", default="")
    parser.add_argument("--quality-protocol-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scenario-key", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--flow-map-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--gipo-checkpoint", required=True)
    parser.add_argument("--demonstration-manifest", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--not-evaluated-reason",
        default="",
        help="Write an explicit code-only status without reading benchmark rows.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argparser().parse_args(list(argv) if argv is not None else None)
    scenario_key = str(args.scenario_key).strip()
    if not scenario_key:
        raise ValueError("--scenario-key is required to bind the quality report.")
    flow_map_path = resolve_project_path(args.flow_map_checkpoint)
    backbone_path = resolve_project_path(args.backbone_checkpoint)
    gipo_path = resolve_project_path(args.gipo_checkpoint)
    output_path = resolve_project_path(args.output_json)
    if output_path.exists() and not output_path.is_file():
        raise ValueError("Quality report output path must be a file.")
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Refusing to overwrite existing quality report {output_path.name!r}."
        )
    rows_path = (
        resolve_project_path(args.rows_csv)
        if str(args.rows_csv).strip()
        else None
    )
    candidate_catalog_path = (
        resolve_project_path(args.candidate_catalog)
        if str(args.candidate_catalog).strip()
        else None
    )
    quality_contexts_path = (
        resolve_project_path(args.quality_contexts_npz)
        if str(args.quality_contexts_npz).strip()
        else None
    )
    quality_sample_panel_path = (
        resolve_project_path(args.quality_sample_panel_npz)
        if str(args.quality_sample_panel_npz).strip()
        else None
    )
    measurement_protocol_path = (
        resolve_project_path(args.measurement_protocol_json)
        if str(args.measurement_protocol_json).strip()
        else None
    )
    demonstration_manifest_path = (
        resolve_project_path(args.demonstration_manifest)
        if str(args.demonstration_manifest).strip()
        else None
    )
    quality_protocol_path = (
        resolve_project_path(args.quality_protocol_json)
        if str(args.quality_protocol_json).strip()
        else None
    )
    protected_inputs = {flow_map_path, backbone_path, gipo_path}
    if rows_path is not None:
        protected_inputs.add(rows_path)
    if candidate_catalog_path is not None:
        protected_inputs.add(candidate_catalog_path)
    if quality_contexts_path is not None:
        protected_inputs.add(quality_contexts_path)
    if quality_sample_panel_path is not None:
        protected_inputs.add(quality_sample_panel_path)
    if measurement_protocol_path is not None:
        protected_inputs.add(measurement_protocol_path)
    if demonstration_manifest_path is not None:
        protected_inputs.add(demonstration_manifest_path)
    if quality_protocol_path is not None:
        protected_inputs.add(quality_protocol_path)
    if output_path in protected_inputs:
        raise ValueError("Quality report output must differ from every input artifact path.")
    input_paths = {
        "backbone_checkpoint": backbone_path,
        "flow_map_checkpoint": flow_map_path,
        "gipo_checkpoint": gipo_path,
    }
    for name, path in (
        ("candidate_catalog", candidate_catalog_path),
        ("demonstration_manifest", demonstration_manifest_path),
        ("measurement_protocol", measurement_protocol_path),
        ("quality_contexts", quality_contexts_path),
        ("quality_protocol", quality_protocol_path),
        ("quality_rows", rows_path),
        ("quality_sample_panel", quality_sample_panel_path),
    ):
        if path is not None:
            input_paths[name] = path
    input_hashes = _capture_input_file_hashes(input_paths)
    _, checkpoint_payload = load_flow_map_checkpoint(
        flow_map_path,
        backbone_checkpoint=backbone_path,
        gipo_checkpoint=gipo_path,
    )
    artifact_binding = {
        "scenario_key": scenario_key,
        "flow_map_checkpoint_sha256": input_hashes["flow_map_checkpoint"],
        "backbone_checkpoint_sha256": str(checkpoint_payload["backbone_checkpoint_sha256"]),
        "gipo_checkpoint_sha256": str(checkpoint_payload["gipo_checkpoint_sha256"]),
    }
    checkpoint_scenario = str(checkpoint_payload.get("scenario_key", "")).strip()
    if checkpoint_scenario and checkpoint_scenario != scenario_key:
        raise ValueError(
            f"Flow-map checkpoint belongs to scenario {checkpoint_scenario!r}, not {scenario_key!r}."
        )
    checkpoint_family = str(checkpoint_payload.get("benchmark_family", "")).strip()
    try:
        expected_family = scenario_family_for_key(scenario_key)
    except KeyError:
        expected_family = ""
    if expected_family and checkpoint_family and checkpoint_family != expected_family:
        raise ValueError("Flow-map checkpoint benchmark family does not match --scenario-key.")
    if str(args.not_evaluated_reason).strip():
        report = not_evaluated_report(
            reason=args.not_evaluated_reason,
            artifact_binding=artifact_binding,
        )
        if quality_protocol_path is not None:
            _, report["quality_protocol_hash"] = read_quality_protocol(
                quality_protocol_path,
                scenario_key=scenario_key,
            )
        _require_input_files_unchanged(input_paths, input_hashes)
        write_json(output_path, report)
        return 0
    if not str(args.rows_csv).strip():
        raise ValueError("--rows-csv is required unless --not-evaluated-reason is used.")
    if rows_path is None:  # Defensive narrowing for type checkers and direct Namespace use.
        raise ValueError("--rows-csv is required unless --not-evaluated-reason is used.")
    if candidate_catalog_path is None:
        raise ValueError(
            "--candidate-catalog is required unless --not-evaluated-reason is used."
        )
    if quality_protocol_path is None:
        raise ValueError(
            "--quality-protocol-json is required for an evaluated performance claim."
        )
    if quality_contexts_path is None:
        raise ValueError(
            "--quality-contexts-npz is required for an evaluated performance claim."
        )
    if quality_sample_panel_path is None:
        raise ValueError(
            "--quality-sample-panel-npz is required for an evaluated performance claim."
        )
    if measurement_protocol_path is None:
        raise ValueError(
            "--measurement-protocol-json is required for an evaluated performance claim."
        )
    raw_context_binding = checkpoint_payload.get("demonstration_context_binding")
    if demonstration_manifest_path is not None:
        manifest_path = demonstration_manifest_path
        if file_sha256(manifest_path) != str(
            checkpoint_payload["demonstration_manifest_sha256"]
        ):
            raise ValueError(
                "Demonstration manifest does not match the flow-map checkpoint."
            )
        manifest = load_demonstration_manifest(manifest_path)
        manifest_metadata = dict(manifest["metadata"])
        if str(manifest_metadata.get("scenario_key", "")).strip() != scenario_key:
            raise ValueError("Demonstration manifest scenario does not match --scenario-key.")
        manifest_family = str(manifest_metadata.get("benchmark_family", "")).strip()
        if checkpoint_family and manifest_family != checkpoint_family:
            raise ValueError(
                "Demonstration manifest benchmark family does not match the flow-map checkpoint."
            )
        if expected_family and manifest_family != expected_family:
            raise ValueError(
                "Demonstration manifest benchmark family does not match --scenario-key."
            )
        manifest_binding = validate_context_binding(
            manifest_metadata.get("context_binding", {})
        )
        if raw_context_binding is not None and validate_context_binding(
            raw_context_binding
        ) != manifest_binding:
            raise ValueError(
                "Flow-map checkpoint and demonstration manifest context bindings disagree."
            )
        raw_context_binding = manifest_binding
    if not isinstance(raw_context_binding, Mapping):
        raise ValueError(
            "A verified demonstration context binding is required for a performance claim; "
            "supply --demonstration-manifest for a checkpoint without an embedded binding."
        )
    specs = _resolve_metric_specs(scenario_key, args.metrics_json)
    catalog = read_candidate_catalog(candidate_catalog_path)
    rows_sha256 = input_hashes["quality_rows"]
    quality_contexts = read_quality_contexts(quality_contexts_path)
    if quality_contexts["artifact_sha256"] != input_hashes["quality_contexts"]:
        raise ValueError("Quality contexts changed while they were being read.")
    quality_sample_panel = read_quality_sample_panel(
        quality_sample_panel_path,
        quality_context_binding=quality_contexts,
    )
    if (
        quality_sample_panel["artifact_sha256"]
        != input_hashes["quality_sample_panel"]
    ):
        raise ValueError("Quality sample panel changed while it was being read.")
    metric_payloads = _metric_spec_payloads(specs)
    measurement_protocol, measurement_digest = read_quality_measurement_protocol(
        measurement_protocol_path,
        scenario_key=scenario_key,
        candidate_catalog_sha256=candidate_catalog_sha256(catalog),
        quality_contexts_sha256=quality_contexts["artifact_sha256"],
        quality_sample_panel_sha256=quality_sample_panel["artifact_sha256"],
        artifact_binding={
            field: artifact_binding[field] for field in ARTIFACT_BINDING_FIELDS[1:]
        },
        primary_metrics=metric_payloads,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
        familywise_alpha=args.familywise_alpha,
    )
    protocol, _ = read_quality_protocol(
        quality_protocol_path,
        scenario_key=scenario_key,
    )
    report = evaluate_quality_gate(
        read_quality_rows(rows_path),
        metric_specs=specs,
        candidate_catalog=catalog,
        artifact_binding=artifact_binding,
        demonstration_context_binding=raw_context_binding,
        quality_context_binding=quality_contexts,
        quality_sample_panel_binding=quality_sample_panel,
        quality_protocol=protocol,
        quality_rows_sha256=rows_sha256,
        measurement_protocol=measurement_protocol,
        measurement_protocol_sha256=measurement_digest,
        config=QualityGateConfig(
            bootstrap_samples=args.bootstrap_samples,
            familywise_alpha=args.familywise_alpha,
            margin=0.0,
            seed=args.seed,
        ),
    )
    report["rows_sha256"] = rows_sha256
    report["candidate_catalog_file_sha256"] = input_hashes["candidate_catalog"]
    _require_input_files_unchanged(input_paths, input_hashes)
    write_json(output_path, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MetricSpec",
    "CandidateSetting",
    "QualityGateConfig",
    "build_argparser",
    "candidate_catalog_sha256",
    "evaluate_quality_gate",
    "main",
    "metric_specs_for_scenario",
    "not_evaluated_report",
    "read_quality_contexts",
    "read_quality_rows",
    "read_quality_sample_panel",
    "read_candidate_catalog",
    "read_quality_protocol",
    "quality_protocol_binding",
    "validate_quality_protocol",
    "validate_quality_context_binding",
    "validate_quality_sample_panel_binding",
]
