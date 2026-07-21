from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from genode.distillation.artifacts import write_json
from genode.distillation.checkpoint import load_flow_map_checkpoint
from genode.gipo.objectives import (
    METRIC_DIRECTION_HIGHER,
    METRIC_DIRECTION_LOWER,
    teacher_objective_specs_for_scenario,
)
from genode.provenance import file_sha256
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
        if int(self.bootstrap_samples) < 1_000:
            raise ValueError("bootstrap_samples must be at least 1,000.")
        if not 0.0 < float(self.familywise_alpha) < 1.0:
            raise ValueError("familywise_alpha must lie strictly between zero and one.")
        if not math.isfinite(float(self.margin)) or float(self.margin) != 0.0:
            raise ValueError("The release gate uses a fixed zero non-inferiority margin.")


@dataclass(frozen=True, order=True)
class CandidateSetting:
    method: str
    candidate_key: str
    solver_key: str
    target_nfe: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "candidate_key": self.candidate_key,
            "solver_key": self.solver_key,
            "target_nfe": int(self.target_nfe),
        }


def metric_specs_for_scenario(scenario_key: str) -> tuple[MetricSpec, ...]:
    return tuple(
        MetricSpec(spec.metric_key, spec.direction, spec.weight, spec.applicable_key)
        for spec in teacher_objective_specs_for_scenario(str(scenario_key))
    )


def _finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return number


def _normalize_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_specs: Sequence[MetricSpec],
    *,
    artifact_binding: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Quality-gate evaluation requires non-empty metric rows.")
    names = [spec.name for spec in metric_specs]
    if not names or len(set(names)) != len(names):
        raise ValueError("Primary metric specifications must be non-empty and unique.")
    normalized: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    candidate_settings: dict[tuple[str, str, str], tuple[str, int]] = {}
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
        nfe = normalize_solver_nfe_fields(
            str(row.get("solver_key", "")),
            int(row.get("target_nfe", 0)),
            source=f"quality row {row_index}",
        )
        row_key = (phase, method, candidate_key, context_id)
        if row_key in seen_rows:
            raise ValueError(
                "Quality rows must be unique by split, method, candidate, and context."
            )
        seen_rows.add(row_key)
        candidate_identity = (phase, method, candidate_key)
        setting = (nfe.solver_key, nfe.target_nfe)
        previous_setting = candidate_settings.setdefault(candidate_identity, setting)
        if previous_setting != setting:
            raise ValueError("A candidate_key may not identify multiple solver/NFE settings.")
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
                "context_id": context_id,
                "selection_utility": _finite_float(
                    row.get("selection_utility"),
                    label=f"quality row {row_index} selection_utility",
                ),
                "metric_applicability": applicability,
                **metric_values,
            }
        )
    phases = {row["split_phase"] for row in normalized}
    if phases != {VALIDATION_PHASE, LOCKED_TEST_PHASE}:
        raise ValueError("Quality-gate rows must include validation_tuning and locked_test phases.")
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
    return normalized


def _candidate(row: Mapping[str, Any]) -> CandidateSetting:
    return CandidateSetting(
        method=str(row["method"]),
        candidate_key=str(row["candidate_key"]),
        solver_key=str(row["solver_key"]),
        target_nfe=int(row["target_nfe"]),
    )


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
            str(row["context_id"])
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
    utility_means: dict[CandidateSetting, float] = {}
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
            means[candidate][spec.name] = float(
                np.mean([float(row[spec.name]) for row in applicable_rows])
            )
        utility_means[candidate] = float(
            np.mean([float(row["selection_utility"]) for row in selected_rows])
        )
    for spec in metric_specs:
        metric_context_sets = {
            candidate: {
                str(row["context_id"])
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
    best_utility = max(utility_means.values())
    winners = [
        candidate
        for candidate, value in utility_means.items()
        if math.isclose(value, best_utility, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if len(winners) != 1:
        raise ValueError(
            f"Validation selection for method {method!r} is tied; provide a prespecified "
            "utility that uniquely freezes one candidate."
        )
    selected = winners[0]
    return selected, {
        "selection_split": VALIDATION_PHASE,
        "selection_utility": utility_means[selected],
        "selection_protocol": "prespecified_scenario_utility",
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
            grouped.setdefault(str(row["context_id"]), []).append(float(row[metric.name]))
        return {context_id: float(np.mean(values)) for context_id, values in grouped.items()}

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
    differences = np.asarray(
        [
            metric.sign * (flow_values[context_id] - comparator_values[context_id])
            - float(margin)
            for context_id in common
        ],
        dtype=np.float64,
    )
    return differences, common


def _bootstrap_test(
    differences: np.ndarray,
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    observed = float(differences.mean())
    centered = differences - observed
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
    artifact_binding: Mapping[str, str],
    config: QualityGateConfig | None = None,
) -> dict[str, Any]:
    """Select settings on validation, then gate every locked-test primary metric.

    The zero-margin test is intentionally conservative: a result must provide
    evidence that the flow map is better than both independently selected
    comparators. Merely failing to detect a difference does not pass.
    """

    gate_config = config or QualityGateConfig()
    specs = tuple(metric_specs)
    binding = {field: str(artifact_binding.get(field, "")).strip() for field in ARTIFACT_BINDING_FIELDS}
    if not binding["scenario_key"]:
        raise ValueError("artifact_binding requires a scenario_key.")
    for field in ARTIFACT_BINDING_FIELDS[1:]:
        value = binding[field]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"artifact_binding {field!r} must be a lowercase SHA-256 digest.")
    normalized = _normalize_rows(rows, specs, artifact_binding=binding)
    validation_panels = {
        method: {
            str(row["context_id"])
            for row in normalized
            if row["split_phase"] == VALIDATION_PHASE and row["method"] == method
        }
        for method in SUPPORTED_METHODS
    }
    if len({frozenset(panel) for panel in validation_panels.values()}) != 1:
        raise ValueError("Every method must use the same validation context panel.")
    selections: dict[str, CandidateSetting] = {}
    selection_metadata: dict[str, Any] = {}
    for method in SUPPORTED_METHODS:
        selection, metadata = _select_on_validation(normalized, specs, method=method)
        selections[method] = selection
        selection_metadata[method] = {**selection.to_payload(), **metadata}

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
        "artifact_binding": binding,
        "selection": selection_metadata,
        "primary_metrics": [
            {
                "name": spec.name,
                "direction": spec.direction,
                "weight": float(spec.weight),
                "applicable_key": spec.applicable_key,
            }
            for spec in specs
        ],
        "comparisons": comparisons,
        "bootstrap_samples": int(gate_config.bootstrap_samples),
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
    input_path = Path(path).expanduser().resolve()
    try:
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ValueError(f"Could not read quality rows {input_path.name!r}: {exc}") from exc


def _metric_specs_from_json(text: str) -> tuple[MetricSpec, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--metrics-json is invalid JSON: {exc}") from exc
    if isinstance(payload, Mapping):
        return tuple(MetricSpec(str(name), str(direction)) for name, direction in payload.items())
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return tuple(
            MetricSpec(
                str(item["name"]),
                str(item["direction"]),
                float(item.get("weight", 1.0)),
                str(item.get("applicable_key", "")),
            )
            for item in payload
            if isinstance(item, Mapping)
        )
    raise ValueError("--metrics-json must be an object or a list of metric objects.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the validation-frozen, familywise flow-map quality gate."
    )
    parser.add_argument("--rows-csv", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--scenario-key", default="")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--flow-map-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--gipo-checkpoint", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--not-evaluated-reason",
        default="",
        help="Write an explicit code-only status without reading benchmark rows.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    scenario_key = str(args.scenario_key).strip()
    if not scenario_key:
        raise ValueError("--scenario-key is required to bind the quality report.")
    flow_map_path = Path(args.flow_map_checkpoint).expanduser().resolve()
    backbone_path = Path(args.backbone_checkpoint).expanduser().resolve()
    gipo_path = Path(args.gipo_checkpoint).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    rows_path = (
        Path(args.rows_csv).expanduser().resolve()
        if str(args.rows_csv).strip()
        else None
    )
    protected_inputs = {flow_map_path, backbone_path, gipo_path}
    if rows_path is not None:
        protected_inputs.add(rows_path)
    if output_path in protected_inputs:
        raise ValueError("Quality report output must differ from every input artifact path.")
    _, checkpoint_payload = load_flow_map_checkpoint(
        flow_map_path,
        backbone_checkpoint=backbone_path,
        gipo_checkpoint=gipo_path,
    )
    artifact_binding = {
        "scenario_key": scenario_key,
        "flow_map_checkpoint_sha256": file_sha256(flow_map_path),
        "backbone_checkpoint_sha256": str(checkpoint_payload["backbone_checkpoint_sha256"]),
        "gipo_checkpoint_sha256": str(checkpoint_payload["gipo_checkpoint_sha256"]),
    }
    if str(args.not_evaluated_reason).strip():
        report = not_evaluated_report(
            reason=args.not_evaluated_reason,
            artifact_binding=artifact_binding,
        )
        write_json(output_path, report)
        return 0
    if not str(args.rows_csv).strip():
        raise ValueError("--rows-csv is required unless --not-evaluated-reason is used.")
    if rows_path is None:  # Defensive narrowing for type checkers and direct Namespace use.
        raise ValueError("--rows-csv is required unless --not-evaluated-reason is used.")
    if str(args.metrics_json).strip():
        specs = _metric_specs_from_json(args.metrics_json)
    else:
        specs = metric_specs_for_scenario(scenario_key)
    report = evaluate_quality_gate(
        read_quality_rows(rows_path),
        metric_specs=specs,
        artifact_binding=artifact_binding,
        config=QualityGateConfig(
            bootstrap_samples=args.bootstrap_samples,
            familywise_alpha=args.familywise_alpha,
            margin=0.0,
            seed=args.seed,
        ),
    )
    report["rows_sha256"] = file_sha256(rows_path)
    write_json(output_path, report)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MetricSpec",
    "QualityGateConfig",
    "evaluate_quality_gate",
    "main",
    "metric_specs_for_scenario",
    "not_evaluated_report",
    "read_quality_rows",
]
