from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from genode.distillation import measurement_protocol as measurement_protocol_module
from genode.distillation.measurement_protocol import (
    QUALITY_EVALUATOR_SOURCE_FILES,
    measurement_protocol_sha256,
    quality_evaluator_binding,
    quality_measurement_protocol_payload,
    read_quality_measurement_protocol,
    validate_quality_measurement_protocol,
)


def test_quality_evaluator_binding_covers_behavior_dependencies() -> None:
    required = {
        "checkpoint_validation.py",
        "distillation/artifacts.py",
        "distillation/evaluation.py",
        "distillation/measurement_protocol.py",
        "experiment_layout.py",
        "gipo/models.py",
        "gipo/objectives.py",
        "schedule_transfer/diffusion_flow_schedules.py",
        "solver_protocol.py",
    }
    assert required <= set(QUALITY_EVALUATOR_SOURCE_FILES)
    assert tuple(sorted(QUALITY_EVALUATOR_SOURCE_FILES)) == QUALITY_EVALUATOR_SOURCE_FILES

    package_root = Path(measurement_protocol_module.__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in QUALITY_EVALUATOR_SOURCE_FILES:
        path = package_root / Path(name)
        assert path.is_file(), name
        content = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    assert quality_evaluator_binding()["implementation_sha256"] == digest.hexdigest()


def _metrics() -> list[dict[str, object]]:
    return [
        {
            "name": "forecast_crps",
            "direction": "lower",
            "weight": 1.0,
            "applicable_key": "",
        }
    ]


def _payload() -> dict[str, object]:
    return quality_measurement_protocol_payload(
        scenario_key="cryptos",
        candidate_catalog_sha256="a" * 64,
        quality_contexts_sha256="b" * 64,
        quality_sample_panel_sha256="c" * 64,
        reference_data_sha256="f" * 64,
        artifact_binding={
            "flow_map_checkpoint_sha256": "1" * 64,
            "backbone_checkpoint_sha256": "2" * 64,
            "gipo_checkpoint_sha256": "3" * 64,
        },
        primary_metrics=_metrics(),
        runner={
            "name": "example-runner",
            "release": "2026-07",
            "implementation_sha256": "d" * 64,
            "environment_sha256": "e" * 64,
        },
    )


def test_measurement_protocol_round_trip_binds_external_runner(tmp_path: Path) -> None:
    path = tmp_path / "measurement_protocol.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    protocol, digest = read_quality_measurement_protocol(
        path,
        scenario_key="cryptos",
        candidate_catalog_sha256="a" * 64,
        quality_contexts_sha256="b" * 64,
        quality_sample_panel_sha256="c" * 64,
        artifact_binding=_payload()["artifact_binding"],
        primary_metrics=_metrics(),
    )

    assert protocol["runner"]["name"] == "example-runner"
    assert protocol["quality_evaluator"]["name"] == "genode-flow-map-quality-gate"
    assert digest == measurement_protocol_sha256(protocol)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update({"candidate_catalog_sha256": "e" * 64}), "does not match"),
        (lambda payload: payload["runner"].update({"implementation_sha256": "bad"}), "SHA-256"),
        (
            lambda payload: payload["artifact_binding"].update(
                {"gipo_checkpoint_sha256": "e" * 64}
            ),
            "does not match",
        ),
        (
            lambda payload: payload["quality_gate"].update(
                {"bootstrap_seed": 7}
            ),
            "does not match",
        ),
        (
            lambda payload: payload["quality_evaluator"].update(
                {"implementation_sha256": "e" * 64}
            ),
            "does not match",
        ),
        (lambda payload: payload.update({"locked_test_used_for_selection": True}), "locked_test"),
        (lambda payload: payload.update({"flow_map_model_evaluations": 2}), "one flow-map"),
    ),
)
def test_measurement_protocol_rejects_unbound_or_weakened_claims(
    mutation,
    message: str,
) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_quality_measurement_protocol(
            payload,
            scenario_key="cryptos",
            candidate_catalog_sha256="a" * 64,
            quality_contexts_sha256="b" * 64,
            quality_sample_panel_sha256="c" * 64,
            artifact_binding=_payload()["artifact_binding"],
            primary_metrics=_metrics(),
        )


def test_measurement_protocol_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "measurement_protocol.json"
    encoded = json.dumps(_payload())
    path.write_text(
        '{"protocol":"duplicate",' + encoded.removeprefix("{"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        read_quality_measurement_protocol(
            path,
            scenario_key="cryptos",
            candidate_catalog_sha256="a" * 64,
            quality_contexts_sha256="b" * 64,
            quality_sample_panel_sha256="c" * 64,
            artifact_binding=_payload()["artifact_binding"],
            primary_metrics=_metrics(),
        )
