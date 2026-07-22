from __future__ import annotations

import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence

from genode.checkpoint_validation import validate_strict_integer
from genode.data.otflow_paths import resolve_project_path


MEASUREMENT_PROTOCOL = "flow_map_quality_measurement"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
QUALITY_EVALUATOR_SOURCE_FILES = (
    "checkpoint_validation.py",
    "data/otflow_paths.py",
    "distillation/artifacts.py",
    "distillation/checkpoint.py",
    "distillation/evaluation.py",
    "distillation/measurement_protocol.py",
    "experiment_layout.py",
    "gipo/density_representation.py",
    "gipo/models.py",
    "gipo/objectives.py",
    "path_safety.py",
    "provenance.py",
    "schedule_transfer/diffusion_flow_schedules.py",
    "solver_protocol.py",
)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    digest = value
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return digest


def _primary_metrics(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("Measurement protocol primary_metrics must be a non-empty list.")
    metrics: list[dict[str, Any]] = []
    for index, raw in enumerate(values):
        required = {"name", "direction", "weight", "applicable_key"}
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(
                f"Measurement protocol primary metric {index} has invalid fields."
            )
        name = str(raw["name"]).strip()
        direction = str(raw["direction"]).strip()
        applicable_key = str(raw["applicable_key"]).strip()
        if isinstance(raw["weight"], bool):
            raise ValueError(
                f"Measurement protocol primary metric {index} weight must be numeric."
            )
        try:
            weight = float(raw["weight"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Measurement protocol primary metric {index} weight must be numeric."
            ) from exc
        if not name or direction not in {"higher", "lower"}:
            raise ValueError(
                f"Measurement protocol primary metric {index} has an invalid name or direction."
            )
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"Measurement protocol primary metric {index} weight must be finite and positive."
            )
        metrics.append(
            {
                "name": name,
                "direction": direction,
                "weight": weight,
                "applicable_key": applicable_key,
            }
        )
    names = [metric["name"] for metric in metrics]
    if len(set(names)) != len(names):
        raise ValueError("Measurement protocol primary metric names must be unique.")
    return metrics


def _runner(values: Any) -> dict[str, str]:
    required = {
        "name",
        "release",
        "implementation_sha256",
        "environment_sha256",
    }
    if not isinstance(values, Mapping) or set(values) != required:
        raise ValueError("Measurement protocol runner fields are invalid.")
    name = str(values["name"]).strip()
    release = str(values["release"]).strip()
    if not name or not release:
        raise ValueError("Measurement protocol runner name and release must be non-empty.")
    return {
        "name": name,
        "release": release,
        "implementation_sha256": _sha256(
            values["implementation_sha256"],
            label="Measurement protocol runner implementation_sha256",
        ),
        "environment_sha256": _sha256(
            values["environment_sha256"],
            label="Measurement protocol runner environment_sha256",
        ),
    }


def _artifact_binding(values: Any) -> dict[str, str]:
    required = {
        "flow_map_checkpoint_sha256",
        "backbone_checkpoint_sha256",
        "gipo_checkpoint_sha256",
    }
    if not isinstance(values, Mapping) or set(values) != required:
        raise ValueError("Measurement protocol artifact_binding fields are invalid.")
    return {
        name: _sha256(
            values[name],
            label=f"Measurement protocol artifact_binding {name}",
        )
        for name in sorted(required)
    }


def _quality_gate(
    *,
    bootstrap_samples: Any,
    bootstrap_seed: Any,
    familywise_alpha: Any,
) -> dict[str, Any]:
    samples = validate_strict_integer(
        bootstrap_samples,
        label="Measurement protocol bootstrap_samples",
        minimum=1_000,
    )
    seed = validate_strict_integer(
        bootstrap_seed,
        label="Measurement protocol bootstrap_seed",
        minimum=0,
    )
    if isinstance(familywise_alpha, bool):
        raise ValueError(
            "Measurement protocol familywise_alpha must lie strictly between zero and one."
        )
    try:
        alpha = float(familywise_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Measurement protocol familywise_alpha must lie strictly between zero and one."
        ) from exc
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(
            "Measurement protocol familywise_alpha must lie strictly between zero and one."
        )
    return {
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "familywise_alpha": alpha,
        "margin": 0.0,
        "multiple_testing_correction": "holm",
        "minimum_paired_contexts": 20,
        "selection_split": "validation_tuning",
        "evaluation_split": "locked_test",
        "all_primary_metrics_required": True,
        "selection_protocol": "weighted_normalized_primary_metric_utility",
        "comparison_protocol": "centered_paired_bootstrap_mean_difference",
        "bootstrap_seed_protocol": (
            "base_plus_comparison_index_times_104729_plus_metric_index"
        ),
    }


def quality_evaluator_binding() -> dict[str, str]:
    """Identify the gate implementation and runtime that determine claim results."""

    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for name in QUALITY_EVALUATOR_SOURCE_FILES:
        content = (package_root / Path(name)).read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    try:
        genode_release = metadata.version("genode")
    except metadata.PackageNotFoundError:
        genode_release = "uninstalled-source-tree"
    return {
        "name": "genode-flow-map-quality-gate",
        "genode_release": genode_release,
        "implementation_sha256": digest.hexdigest(),
        "python_release": platform.python_version(),
        "numpy_release": metadata.version("numpy"),
    }


def _quality_evaluator(values: Any) -> dict[str, str]:
    required = {
        "name",
        "genode_release",
        "implementation_sha256",
        "python_release",
        "numpy_release",
    }
    if not isinstance(values, Mapping) or set(values) != required:
        raise ValueError("Measurement protocol quality_evaluator fields are invalid.")
    normalized = {name: str(values[name]).strip() for name in required}
    if normalized["name"] != "genode-flow-map-quality-gate":
        raise ValueError("Measurement protocol quality_evaluator name is invalid.")
    for name in ("genode_release", "python_release", "numpy_release"):
        if not normalized[name]:
            raise ValueError(
                f"Measurement protocol quality_evaluator {name} must be non-empty."
            )
    normalized["implementation_sha256"] = _sha256(
        values["implementation_sha256"],
        label="Measurement protocol quality_evaluator implementation_sha256",
    )
    return normalized


def quality_measurement_protocol_payload(
    *,
    scenario_key: str,
    candidate_catalog_sha256: str,
    quality_contexts_sha256: str,
    quality_sample_panel_sha256: str,
    reference_data_sha256: str,
    artifact_binding: Mapping[str, Any],
    primary_metrics: Sequence[Mapping[str, Any]],
    runner: Mapping[str, Any],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    familywise_alpha: float = 0.05,
    quality_evaluator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact protocol committed before external quality measurement."""

    scenario = str(scenario_key).strip()
    if not scenario:
        raise ValueError("Measurement protocol scenario_key must be non-empty.")
    return {
        "protocol": MEASUREMENT_PROTOCOL,
        "scenario_key": scenario,
        "candidate_catalog_sha256": _sha256(
            candidate_catalog_sha256,
            label="Measurement protocol candidate_catalog_sha256",
        ),
        "quality_contexts_sha256": _sha256(
            quality_contexts_sha256,
            label="Measurement protocol quality_contexts_sha256",
        ),
        "quality_sample_panel_sha256": _sha256(
            quality_sample_panel_sha256,
            label="Measurement protocol quality_sample_panel_sha256",
        ),
        "reference_data_sha256": _sha256(
            reference_data_sha256,
            label="Measurement protocol reference_data_sha256",
        ),
        "artifact_binding": _artifact_binding(artifact_binding),
        "primary_metrics": _primary_metrics(primary_metrics),
        "runner": _runner(runner),
        "quality_gate": _quality_gate(
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            familywise_alpha=familywise_alpha,
        ),
        "quality_evaluator": _quality_evaluator(
            quality_evaluator_binding()
            if quality_evaluator is None
            else quality_evaluator
        ),
        "candidate_execution_required": True,
        "common_sample_panel_required": True,
        "flow_map_model_evaluations": 1,
        "comparator_model_evaluations": "target_nfe",
        "locked_test_used_for_selection": False,
    }


def validate_quality_measurement_protocol(
    payload: Mapping[str, Any],
    *,
    scenario_key: str,
    candidate_catalog_sha256: str,
    quality_contexts_sha256: str,
    quality_sample_panel_sha256: str,
    artifact_binding: Mapping[str, Any],
    primary_metrics: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    familywise_alpha: float = 0.05,
) -> dict[str, Any]:
    required = {
        "protocol",
        "scenario_key",
        "candidate_catalog_sha256",
        "quality_contexts_sha256",
        "quality_sample_panel_sha256",
        "reference_data_sha256",
        "artifact_binding",
        "primary_metrics",
        "runner",
        "quality_gate",
        "quality_evaluator",
        "candidate_execution_required",
        "common_sample_panel_required",
        "flow_map_model_evaluations",
        "comparator_model_evaluations",
        "locked_test_used_for_selection",
    }
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            "Measurement protocol fields are invalid; "
            f"missing={missing}, extra={extra}."
        )
    normalized = quality_measurement_protocol_payload(
        scenario_key=str(payload["scenario_key"]),
        candidate_catalog_sha256=payload["candidate_catalog_sha256"],
        quality_contexts_sha256=payload["quality_contexts_sha256"],
        quality_sample_panel_sha256=payload["quality_sample_panel_sha256"],
        reference_data_sha256=payload["reference_data_sha256"],
        artifact_binding=payload["artifact_binding"],
        primary_metrics=payload["primary_metrics"],
        runner=payload["runner"],
        bootstrap_samples=(
            payload["quality_gate"].get("bootstrap_samples")
            if isinstance(payload["quality_gate"], Mapping)
            else None
        ),
        bootstrap_seed=(
            payload["quality_gate"].get("bootstrap_seed")
            if isinstance(payload["quality_gate"], Mapping)
            else None
        ),
        familywise_alpha=(
            payload["quality_gate"].get("familywise_alpha")
            if isinstance(payload["quality_gate"], Mapping)
            else None
        ),
        quality_evaluator=payload["quality_evaluator"],
    )
    if payload.get("protocol") != MEASUREMENT_PROTOCOL:
        raise ValueError("Unsupported quality measurement protocol.")
    for field in (
        "candidate_execution_required",
        "common_sample_panel_required",
    ):
        if payload.get(field) is not True:
            raise ValueError(f"Measurement protocol requires {field}=true.")
    if payload.get("locked_test_used_for_selection") is not False:
        raise ValueError(
            "Measurement protocol requires locked_test_used_for_selection=false."
        )
    if payload.get("flow_map_model_evaluations") != 1:
        raise ValueError("Measurement protocol requires one flow-map model evaluation.")
    if payload.get("comparator_model_evaluations") != "target_nfe":
        raise ValueError(
            "Measurement protocol comparator evaluations must equal target_nfe."
        )
    if not isinstance(payload["quality_gate"], Mapping) or dict(
        payload["quality_gate"]
    ) != normalized["quality_gate"]:
        raise ValueError("Measurement protocol quality_gate fields are invalid.")
    expected = quality_measurement_protocol_payload(
        scenario_key=scenario_key,
        candidate_catalog_sha256=candidate_catalog_sha256,
        quality_contexts_sha256=quality_contexts_sha256,
        quality_sample_panel_sha256=quality_sample_panel_sha256,
        reference_data_sha256=normalized["reference_data_sha256"],
        artifact_binding=artifact_binding,
        primary_metrics=primary_metrics,
        runner=normalized["runner"],
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        familywise_alpha=familywise_alpha,
    )
    if normalized != expected:
        raise ValueError(
            "Measurement protocol does not match the bound scenario, artifacts, "
            "candidates, contexts, sample panel, primary metrics, or quality gate."
        )
    return normalized


def measurement_protocol_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a normalized measurement protocol independently of JSON formatting."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def read_quality_measurement_protocol(
    path: str | Path,
    *,
    scenario_key: str,
    candidate_catalog_sha256: str,
    quality_contexts_sha256: str,
    quality_sample_panel_sha256: str,
    artifact_binding: Mapping[str, Any],
    primary_metrics: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    familywise_alpha: float = 0.05,
) -> tuple[dict[str, Any], str]:
    input_path = resolve_project_path(path)
    try:
        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(
                        f"Quality measurement protocol contains duplicate key {key!r}."
                    )
                result[key] = value
            return result

        payload = json.loads(
            input_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except OSError as exc:
        raise ValueError(
            f"Could not read quality measurement protocol {input_path.name!r}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Quality measurement protocol {input_path.name!r} is invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Quality measurement protocol JSON must contain an object.")
    normalized = validate_quality_measurement_protocol(
        payload,
        scenario_key=scenario_key,
        candidate_catalog_sha256=candidate_catalog_sha256,
        quality_contexts_sha256=quality_contexts_sha256,
        quality_sample_panel_sha256=quality_sample_panel_sha256,
        artifact_binding=artifact_binding,
        primary_metrics=primary_metrics,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        familywise_alpha=familywise_alpha,
    )
    return normalized, measurement_protocol_sha256(normalized)


__all__ = [
    "MEASUREMENT_PROTOCOL",
    "QUALITY_EVALUATOR_SOURCE_FILES",
    "measurement_protocol_sha256",
    "quality_measurement_protocol_payload",
    "quality_evaluator_binding",
    "read_quality_measurement_protocol",
    "validate_quality_measurement_protocol",
]
