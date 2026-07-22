from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import genode.artifact_bundle as artifact_bundle_module
from genode.artifact_bundle import (
    bundle_journal_path,
    bundle_lock_path,
    exclusive_bundle_lock,
    recover_artifact_bundle,
    validate_artifact_bundle_layout,
)
import genode.distillation.demonstrations as demonstration_module
import genode.distillation.checkpoint as checkpoint_module
from genode.distillation.artifacts import (
    DEMONSTRATION_MANIFEST_NAME,
    context_binding,
    context_fingerprint,
    load_demonstration_manifest,
    write_demonstration_manifest,
    write_npz_shard,
)
from genode.distillation.checkpoint import (
    QUALITY_STATUS_NOT_EVALUATED,
    load_flow_map_checkpoint,
    load_flow_map_sampler,
    save_flow_map_checkpoint,
)
from genode.distillation.demonstrations import (
    DEFAULT_DISTILLATION_NFES,
    DistillationContexts,
    DistillationSetting,
    collect_flow_map_demonstrations,
    default_distillation_settings,
    load_distillation_contexts,
    parse_distillation_settings,
)
from genode.distillation.evaluation import (
    QualityGateConfig,
    _holm_adjust,
    _normalize_candidate_catalog,
    _resolve_metric_specs,
    _time_grid_sha256,
    candidate_catalog_sha256,
    evaluate_quality_gate as _evaluate_quality_gate_impl,
    metric_specs_for_scenario,
    not_evaluated_report,
    quality_protocol_binding,
    read_quality_contexts,
    read_quality_protocol,
    read_quality_rows,
    read_quality_sample_panel,
    validate_quality_context_binding,
    validate_quality_sample_panel_binding,
)
from genode.distillation.gipo_policy import (
    GIPOSchedule,
    GIPOSchedulePolicy,
    load_gipo_schedule_policy,
)
from genode.distillation.model import (
    EndpointFlowMap,
    FlowMapSampler,
    endpoint_consistency_loss,
)
from genode.distillation.measurement_protocol import (
    measurement_protocol_sha256,
    quality_measurement_protocol_payload,
)
from genode.distillation.training import (
    DemonstrationStore,
    _write_flow_map_bundle,
    main as train_flow_map_main,
    recover_flow_map_bundle,
    train_endpoint_flow_map,
    validate_flow_map_bundle,
)
from genode.gipo.density_representation import (
    DENSITY_BIN_COUNT,
    DENSITY_PROTOCOL,
    uniform_reference_grid,
)
from genode.gipo.models import build_setting_encoder_config, setting_feature_dim
from genode.gipo.policy import (
    ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
    GIPO_PROTOCOL,
    MODEL_PAYLOAD_VERSION,
    EmbeddingNormalizer,
    build_gipo_student_model,
    normalize_gipo_checkpoint_payload,
)
from genode.models.conditioning import ConditioningState
from genode.models.config import OTFlowConfig
from genode.models.otflow_model import OTFlow
from genode.provenance import file_sha256
from genode.schedule_transfer.diffusion_flow_schedules import build_schedule_grid
from genode.solver_protocol import SUPPORTED_SOLVER_KEYS


def _hold_demonstration_collection_lock(
    target: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with demonstration_module._exclusive_collection_lock(Path(target)):
        ready.set()
        if not release.wait(timeout=30):
            raise RuntimeError("Timed out waiting to release demonstration collection lock.")


def _hold_artifact_bundle_lock(
    target: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with exclusive_bundle_lock(Path(target)):
        ready.set()
        if not release.wait(timeout=30):
            raise RuntimeError("Timed out waiting to release artifact bundle lock.")


def _tiny_config() -> OTFlowConfig:
    return OTFlowConfig(
        device=torch.device("cpu"),
        levels=1,
        token_dim=4,
        history_len=2,
        hidden_dim=8,
        dropout=0.0,
        ctx_heads=4,
        ctx_layers=1,
        fu_net_layers=1,
        fu_net_heads=4,
        use_amp=False,
        use_minibatch_ot=False,
    )


def _setting_config():
    return build_setting_encoder_config(observed_target_nfes=(2, 4, 6, 8))


def _flow_map() -> EndpointFlowMap:
    config = _setting_config()
    return EndpointFlowMap(
        _tiny_config(),
        setting_dim=setting_feature_dim(config=config),
        density_dim=DENSITY_BIN_COUNT,
    )


def _test_context_fingerprint(label: str) -> str:
    values = np.frombuffer(label.encode("utf-8"), dtype=np.uint8).astype(np.float64)
    return context_fingerprint(values)


def _manifest_metadata(*, context_count: int, split_phase: str = "train") -> dict[str, object]:
    return {
        "artifact_version": 1,
        "split_phase": split_phase,
        "scenario_key": "synthetic",
        "benchmark_family": "synthetic",
        "locked_test_used": False,
        "context_count": context_count,
        "rollouts_per_context": 1,
        "density_bin_count": DENSITY_BIN_COUNT,
        "density_reference_time_grid": list(uniform_reference_grid()),
        "context_embedding_kind": "ctx_summary",
        "collection_seed": 0,
        "sample_state_dim": 4,
        "settings": [{"solver_key": "euler", "target_nfe": 2}],
        "setting_encoder_config": _setting_config().to_payload(),
        "backbone_checkpoint_sha256": "a" * 64,
        "gipo_checkpoint_sha256": "b" * 64,
    }


def _write_minimal_manifest(root: Path, *, split_phase: str = "train") -> Path:
    context = write_npz_shard(
        root,
        "shards/contexts_00000.npz",
        {
            "context_index": np.asarray([0], dtype=np.int64),
            "context_id": np.asarray(["context-0"], dtype=np.str_),
            "context_fingerprint": np.asarray(
                [_test_context_fingerprint("context-0-physical")],
                dtype=np.str_,
            ),
            "ctx_summary": np.zeros((1, 8), dtype=np.float32),
        },
    )
    trajectory = write_npz_shard(
        root,
        "shards/trajectories_euler_nfe2_00000.npz",
        {
            "context_index": np.asarray([0], dtype=np.int64),
            "states": np.zeros((1, 3, 4), dtype=np.float32),
            "density_mass": np.full((1, DENSITY_BIN_COUNT), 1.0 / DENSITY_BIN_COUNT, dtype=np.float32),
        },
    )
    return write_demonstration_manifest(
        root,
        context_shards=[context],
        trajectory_shards=[trajectory],
        metadata=_manifest_metadata(context_count=1, split_phase=split_phase),
    )


def _write_training_manifest(root: Path, *, duplicate_context_ids: bool = False) -> Path:
    context_ids = ["context-0", "context-0" if duplicate_context_ids else "context-1"]
    context = write_npz_shard(
        root,
        "shards/contexts_00000.npz",
        {
            "context_index": np.asarray([0, 1], dtype=np.int64),
            "context_id": np.asarray(context_ids, dtype=np.str_),
            "context_fingerprint": np.asarray(
                [
                    _test_context_fingerprint("context-0-physical"),
                    _test_context_fingerprint("context-1-physical"),
                ],
                dtype=np.str_,
            ),
            "ctx_tokens": np.zeros((2, 2, 8), dtype=np.float32),
            "ctx_summary": np.zeros((2, 8), dtype=np.float32),
            "summary": np.zeros((2, 8), dtype=np.float32),
        },
    )
    initial = np.asarray(
        [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]],
        dtype=np.float32,
    )
    states = np.stack((initial, initial + 0.25, initial + 0.5), axis=1)
    trajectory = write_npz_shard(
        root,
        "shards/trajectories_euler_nfe2_00000.npz",
        {
            "context_index": np.asarray([0, 1], dtype=np.int64),
            "noise_seed": np.asarray([101, 102], dtype=np.int64),
            "initial_state": initial,
            "time_grid": np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32),
            "states": states,
            "density_mass": np.full((2, DENSITY_BIN_COUNT), 1.0 / DENSITY_BIN_COUNT, dtype=np.float32),
        },
    )
    trajectory.update(
        {
            "solver_key": "euler",
            "target_nfe": 2,
            "context_start": 0,
            "context_stop": 2,
        }
    )
    return write_demonstration_manifest(
        root,
        context_shards=[context],
        trajectory_shards=[trajectory],
        metadata=_manifest_metadata(context_count=2),
    )


def _quality_binding() -> dict[str, str]:
    return {
        "scenario_key": "cryptos",
        "flow_map_checkpoint_sha256": "c" * 64,
        "backbone_checkpoint_sha256": "d" * 64,
        "gipo_checkpoint_sha256": "e" * 64,
    }


def _quality_candidate_catalog() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for candidate_key, target_nfe in (("flow-selected", 4), ("flow-decoy", 6)):
        candidates.append(
            {
                "method": "flow_map",
                "candidate_key": candidate_key,
                "solver_key": "euler",
                "target_nfe": target_nfe,
                "execution": {
                    "kind": "endpoint_flow_map",
                    "density_source": "bound_gipo_checkpoint",
                },
            }
        )
    for candidate_key, target_nfe in (("gipo-selected", 4), ("gipo-decoy", 6)):
        candidates.append(
            {
                "method": "gipo",
                "candidate_key": candidate_key,
                "solver_key": "euler",
                "target_nfe": target_nfe,
                "execution": {
                    "kind": "gipo_ode_rollout",
                    "policy_sha256": "e" * 64,
                },
            }
        )
    for candidate_key, target_nfe, scheduler_key in (
        ("fixed-selected", 4, "uniform"),
        ("fixed-decoy", 6, "late_power_3"),
    ):
        time_grid = build_schedule_grid(scheduler_key, target_nfe)
        assert time_grid is not None
        candidates.append(
            {
                "method": "fixed",
                "candidate_key": candidate_key,
                "solver_key": "euler",
                "target_nfe": target_nfe,
                "execution": {
                    "kind": "fixed_time_grid",
                    "scheduler_key": scheduler_key,
                    "density_source_key": scheduler_key,
                    "time_grid": list(time_grid),
                    "time_grid_sha256": _time_grid_sha256(time_grid),
                },
            }
        )
    return candidates


def _quality_context_binding() -> dict[str, object]:
    contexts = [
        {
            "split_phase": "validation_tuning",
            "context_id": context_id,
            "context_fingerprint": _test_context_fingerprint(context_id),
        }
        for context_id in ("validation-0", "validation-1")
    ] + [
        {
            "split_phase": "locked_test",
            "context_id": context_id,
            "context_fingerprint": _test_context_fingerprint(context_id),
        }
        for context_id in (f"locked-{index}" for index in range(20))
    ]
    contexts = sorted(
        contexts, key=lambda row: (str(row["split_phase"]), str(row["context_id"]))
    )
    return validate_quality_context_binding(
        {
            "protocol": "flow_map_quality_contexts",
            "artifact_sha256": "1" * 64,
            "context_count": len(contexts),
            "contexts": contexts,
            "set_sha256": hashlib.sha256(
                json.dumps(contexts, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    )


def _quality_sample_panel_binding() -> dict[str, object]:
    contexts = _quality_context_binding()
    panels = [
        {
            **row,
            "sample_panel_sha256": _test_context_fingerprint(
                f"sample-panel:{row['context_id']}"
            ),
            "replicate_count": 2,
        }
        for row in contexts["contexts"]
    ]
    return validate_quality_sample_panel_binding(
        {
            "protocol": "flow_map_quality_sample_panel",
            "artifact_sha256": "2" * 64,
            "context_count": len(panels),
            "sample_count": 2 * len(panels),
            "replicate_count": 2,
            "panels": panels,
            "set_sha256": hashlib.sha256(
                json.dumps(panels, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        quality_context_binding=contexts,
    )


_AUTO_MEASUREMENT_PROTOCOL = "test-auto-measurement-protocol"


def _quality_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = {
        "flow_map": (("flow-selected", 4, 1.0, 9.0), ("flow-decoy", 6, 2.0, 8.0)),
        "gipo": (("gipo-selected", 4, 1.0, 9.0), ("gipo-decoy", 6, 2.0, 8.0)),
        "fixed": (("fixed-selected", 4, 1.0, 9.0), ("fixed-decoy", 6, 2.0, 8.0)),
    }
    normalized_candidates = {
        (candidate.method, candidate.candidate_key): candidate
        for candidate in _normalize_candidate_catalog(_quality_candidate_catalog())
    }
    sample_panels = {
        (row["split_phase"], row["context_id"]): row
        for row in _quality_sample_panel_binding()["panels"]
    }

    def add_row(
        *,
        phase: str,
        method: str,
        candidate_key: str,
        target_nfe: int,
        context_id: str,
        error: float,
        score: float,
    ) -> None:
        candidate = normalized_candidates[(method, candidate_key)]
        panel = sample_panels[(phase, context_id)]
        rows.append(
            {
                "split_phase": phase,
                "method": method,
                "candidate_key": candidate_key,
                "candidate_execution_sha256": candidate.execution_sha256,
                "solver_key": "euler",
                "target_nfe": target_nfe,
                "model_evaluations": 1 if method == "flow_map" else target_nfe,
                "context_id": context_id,
                "context_fingerprint": _test_context_fingerprint(context_id),
                "measurement_protocol_sha256": _AUTO_MEASUREMENT_PROTOCOL,
                "sample_panel_sha256": panel["sample_panel_sha256"],
                "replicate_count": panel["replicate_count"],
                **_quality_binding(),
                "temporal_uw1": error,
                "temporal_cw1": error,
                "temporal_tstr_f1": score,
                "temporal_tstr_f1_applicable": True,
            }
        )

    for method, method_candidates in candidates.items():
        for candidate_key, target_nfe, error, score in method_candidates:
            for context_id in ("validation-0", "validation-1"):
                add_row(
                    phase="validation_tuning",
                    method=method,
                    candidate_key=candidate_key,
                    target_nfe=target_nfe,
                    context_id=context_id,
                    error=error,
                    score=score,
                )
    locked_values = {
        "flow_map": ("flow-selected", 0.0, 10.0),
        "gipo": ("gipo-selected", 1.0, 9.0),
        "fixed": ("fixed-selected", 2.0, 8.0),
    }
    for method, (candidate_key, error, score) in locked_values.items():
        for context_id in (f"locked-{index}" for index in range(20)):
            add_row(
                phase="locked_test",
                method=method,
                candidate_key=candidate_key,
                target_nfe=4,
                context_id=context_id,
                error=error,
                score=score,
            )
    return rows


def _quality_protocol() -> dict[str, object]:
    body = {
        "scenario_key": "cryptos",
        "flow_map": {
            "quality_candidate_catalog_sha256": candidate_catalog_sha256(
                _quality_candidate_catalog()
            ),
            "quality_rows_sha256": "9" * 64,
            "quality_contexts_sha256": "1" * 64,
            "quality_sample_panel_sha256": "2" * 64,
            "quality_measurement_protocol_sha256": (
                _AUTO_MEASUREMENT_PROTOCOL
            ),
        },
    }
    return {
        **body,
        "protocol_hash": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _quality_measurement_protocol(
    config: QualityGateConfig,
) -> dict[str, object]:
    binding = _quality_binding()
    return quality_measurement_protocol_payload(
        scenario_key=str(binding["scenario_key"]),
        candidate_catalog_sha256=candidate_catalog_sha256(
            _quality_candidate_catalog()
        ),
        quality_contexts_sha256="1" * 64,
        quality_sample_panel_sha256="2" * 64,
        reference_data_sha256="f" * 64,
        artifact_binding={
            name: binding[name]
            for name in (
                "flow_map_checkpoint_sha256",
                "backbone_checkpoint_sha256",
                "gipo_checkpoint_sha256",
            )
        },
        primary_metrics=[
            {
                "name": spec.name,
                "direction": spec.direction,
                "weight": float(spec.weight),
                "applicable_key": spec.applicable_key,
            }
            for spec in metric_specs_for_scenario("cryptos")
        ],
        runner={
            "name": "test-quality-runner",
            "release": "test-release",
            "implementation_sha256": "4" * 64,
            "environment_sha256": "5" * 64,
        },
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.seed,
        familywise_alpha=config.familywise_alpha,
    )


def evaluate_quality_gate(rows, *args, **kwargs):
    config = kwargs.get("config") or QualityGateConfig()
    protocol = kwargs.setdefault(
        "measurement_protocol", _quality_measurement_protocol(config)
    )
    digest = kwargs.setdefault(
        "measurement_protocol_sha256", measurement_protocol_sha256(protocol)
    )
    quality_protocol = json.loads(
        json.dumps(kwargs.get("quality_protocol") or _quality_protocol())
    )
    flow_map_protocol = quality_protocol.get("flow_map", {})
    if (
        flow_map_protocol.get("quality_measurement_protocol_sha256")
        == _AUTO_MEASUREMENT_PROTOCOL
    ):
        flow_map_protocol["quality_measurement_protocol_sha256"] = digest
        quality_protocol["protocol_hash"] = hashlib.sha256(
            json.dumps(
                {
                    name: value
                    for name, value in quality_protocol.items()
                    if name != "protocol_hash"
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    kwargs["quality_protocol"] = quality_protocol
    kwargs.setdefault("quality_rows_sha256", "9" * 64)
    bound_rows = []
    for raw in rows:
        row = dict(raw)
        if row.get("measurement_protocol_sha256") == _AUTO_MEASUREMENT_PROTOCOL:
            row["measurement_protocol_sha256"] = digest
        bound_rows.append(row)
    kwargs.setdefault("quality_context_binding", _quality_context_binding())
    kwargs.setdefault(
        "quality_sample_panel_binding", _quality_sample_panel_binding()
    )
    return _evaluate_quality_gate_impl(bound_rows, *args, **kwargs)


def test_demonstration_manifest_round_trip_uses_portable_verified_shards(tmp_path: Path) -> None:
    manifest_path = _write_minimal_manifest(tmp_path)

    manifest = load_demonstration_manifest(manifest_path)

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["path_base"] == "manifest_parent"
    assert raw["context_shards"][0]["path"] == "shards/contexts_00000.npz"
    assert not Path(raw["context_shards"][0]["path"]).is_absolute()
    assert manifest["root"] == tmp_path.resolve()
    assert manifest["context_shards"][0]["resolved_path"].is_file()
    assert manifest["trajectory_shards"][0]["resolved_path"].is_file()

    write_demonstration_manifest(
        tmp_path,
        context_shards=manifest["context_shards"],
        trajectory_shards=manifest["trajectory_shards"],
        metadata=manifest["metadata"],
    )
    rewritten = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "resolved_path" not in rewritten["context_shards"][0]
    assert "resolved_path" not in rewritten["trajectory_shards"][0]
    assert str(tmp_path.resolve()) not in manifest_path.read_text(encoding="utf-8")


def test_context_fingerprint_is_label_independent_and_dtype_stable() -> None:
    history32 = np.asarray([[1.0, -2.0], [3.5, 4.0]], dtype=np.float32)
    condition32 = np.asarray([0.25, 0.5], dtype=np.float32)

    first = context_fingerprint(history32, condition32)
    second = context_fingerprint(history32.astype(np.float64), condition32.astype(np.float64))

    assert first == second
    assert context_binding((first,))["context_fingerprints"] == [first]


@pytest.mark.parametrize("relative_path", ["../outside.npz", "/absolute.npz", "shards/not_npz.bin"])
def test_demonstration_shards_reject_unsafe_or_wrong_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    with pytest.raises(ValueError):
        write_npz_shard(tmp_path, relative_path, {"value": np.zeros(1, dtype=np.float32)})


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([object()], dtype=object),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
    ],
)
def test_demonstration_shards_reject_unsafe_array_payloads(tmp_path: Path, array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        write_npz_shard(tmp_path, "shards/bad.npz", {"value": array})


def test_demonstration_manifest_detects_tampering(tmp_path: Path) -> None:
    manifest_path = _write_minimal_manifest(tmp_path)
    shard = tmp_path / "shards" / "contexts_00000.npz"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        load_demonstration_manifest(manifest_path)


def test_demonstration_manifest_rejects_fractional_shape_metadata(tmp_path: Path) -> None:
    manifest_path = _write_minimal_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["context_shards"][0]["arrays"]["context_id"]["shape"] = [1.0]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be an integer"):
        load_demonstration_manifest(manifest_path)


def test_demonstration_manifest_requires_literal_false_locked_test_flag(
    tmp_path: Path,
) -> None:
    manifest_path = _write_minimal_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["locked_test_used"] = 0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="locked_test_used=false"):
        load_demonstration_manifest(manifest_path)


def test_demonstration_manifest_rejects_locked_test_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-test"):
        _write_minimal_manifest(tmp_path, split_phase="locked_test")


def test_distillation_setting_parser_covers_the_supported_matrix() -> None:
    defaults = default_distillation_settings()

    assert len(defaults) == len(SUPPORTED_SOLVER_KEYS) * len(DEFAULT_DISTILLATION_NFES)
    assert {(setting.solver_key, setting.target_nfe) for setting in defaults} == {
        (solver_key, target_nfe)
        for solver_key in SUPPORTED_SOLVER_KEYS
        for target_nfe in DEFAULT_DISTILLATION_NFES
    }
    assert parse_distillation_settings("euler:4,heun:6") == (
        DistillationSetting("euler", 4),
        DistillationSetting("heun", 6),
    )


@pytest.mark.parametrize(
    "text, error",
    [
        ("rk4:4", "Unknown solver_key"),
        ("heun:5", "even target NFE"),
        ("euler:4,euler:4", "duplicates"),
        ("euler", "solver_key:target_nfe"),
    ],
)
def test_distillation_setting_parser_rejects_invalid_settings(text: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_distillation_settings(text)


def test_endpoint_flow_map_enforces_terminal_boundary_and_validates_density() -> None:
    torch.manual_seed(7)
    flow_map = _flow_map().eval()
    cfg = flow_map.cfg
    batch = 2
    state = torch.randn(batch, cfg.sample_state_dim)
    conditioning = ConditioningState(
        ctx=torch.randn(batch, cfg.hidden_dim),
        ctx_summary=torch.randn(batch, cfg.hidden_dim),
        t_emb=torch.randn(batch, cfg.hidden_dim),
        cond_emb=None,
        ctx_tokens=torch.randn(batch, cfg.history_len, cfg.hidden_dim),
    )
    setting = torch.randn(batch, flow_map.setting_dim)
    density = torch.full((batch, flow_map.density_dim), 1.0 / flow_map.density_dim)

    at_endpoint = flow_map(
        state,
        torch.ones(batch, 1),
        conditioning,
        setting,
        density,
    )
    before_endpoint = flow_map(
        state,
        torch.zeros(batch, 1),
        conditioning,
        setting,
        density,
    )

    assert torch.equal(at_endpoint, state)
    assert before_endpoint.shape == state.shape
    assert torch.isfinite(before_endpoint).all()
    with pytest.raises(ValueError, match="sum to one"):
        flow_map(
            state,
            torch.zeros(batch, 1),
            conditioning,
            setting,
            torch.ones_like(density),
            validate_values=True,
        )


def test_endpoint_consistency_loss_is_zero_at_target_symmetric_and_differentiable() -> None:
    prediction = torch.tensor([[0.5, -0.25]], requires_grad=True)
    target = torch.zeros_like(prediction)

    zero = endpoint_consistency_loss(target, target)
    positive = endpoint_consistency_loss(prediction, target, delta=0.1)
    negative = endpoint_consistency_loss(-prediction, target, delta=0.1)
    positive.backward()

    assert zero.item() == pytest.approx(0.0)
    assert positive.item() > 0.0
    assert positive.item() == pytest.approx(negative.item())
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_flow_map_checkpoint_round_trip_and_source_hash_compatibility(tmp_path: Path) -> None:
    torch.manual_seed(11)
    flow_map = _flow_map()
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    incompatible_gipo = tmp_path / "other-gipo.pt"
    backbone_checkpoint.write_bytes(b"backbone checkpoint")
    gipo_checkpoint.write_bytes(b"gipo checkpoint")
    incompatible_gipo.write_bytes(b"different checkpoint")
    checkpoint = tmp_path / "flow-map.pt"

    save_flow_map_checkpoint(
        checkpoint,
        flow_map,
        backbone_checkpoint=backbone_checkpoint,
        gipo_checkpoint=gipo_checkpoint,
        setting_encoder_config=_setting_config(),
        training_summary={"locked_test_used_for_selection": False, "epochs": 1},
        demonstration_manifest_sha256="a" * 64,
    )
    loaded, payload = load_flow_map_checkpoint(
        checkpoint,
        backbone_checkpoint=backbone_checkpoint,
        gipo_checkpoint=gipo_checkpoint,
    )

    assert payload["quality_gate"]["status"] == QUALITY_STATUS_NOT_EVALUATED
    assert payload["backbone_checkpoint_sha256"]
    assert payload["gipo_checkpoint_sha256"]
    assert "backbone_checkpoint" not in payload
    assert "gipo_checkpoint" not in payload
    assert loaded.model_config() == flow_map.model_config()
    for name, expected in flow_map.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected)

    with pytest.raises(ValueError, match="not compatible.*GIPO"):
        load_flow_map_checkpoint(checkpoint, gipo_checkpoint=incompatible_gipo)


def test_flow_map_checkpoint_rejects_changed_expected_source(tmp_path: Path) -> None:
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    gipo_checkpoint.write_bytes(b"original gipo")
    expected_gipo_hash = file_sha256(gipo_checkpoint)
    gipo_checkpoint.write_bytes(b"replacement gipo")
    output = tmp_path / "flow-map.pt"

    with pytest.raises(ValueError, match="GIPO checkpoint changed"):
        save_flow_map_checkpoint(
            output,
            _flow_map(),
            backbone_checkpoint=backbone_checkpoint,
            gipo_checkpoint=gipo_checkpoint,
            setting_encoder_config=_setting_config(),
            training_summary={"locked_test_used_for_selection": False},
            demonstration_manifest_sha256="b" * 64,
            expected_backbone_checkpoint_sha256=file_sha256(backbone_checkpoint),
            expected_gipo_checkpoint_sha256=expected_gipo_hash,
        )

    assert not output.exists()


def test_flow_map_checkpoint_rejects_lexical_link_without_touching_referent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    referent = tmp_path / "referent.pt"
    output = tmp_path / "flow-map.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    gipo_checkpoint.write_bytes(b"gipo")
    referent.write_bytes(b"referent must remain unchanged")
    try:
        output.symlink_to(referent)
    except OSError:
        monkeypatch.setattr(
            checkpoint_module,
            "is_link_or_reparse_point",
            lambda path: Path(path) == output,
        )

    with pytest.raises(ValueError, match="symlink, junction, or reparse"):
        save_flow_map_checkpoint(
            output,
            _flow_map(),
            backbone_checkpoint=backbone_checkpoint,
            gipo_checkpoint=gipo_checkpoint,
            setting_encoder_config=_setting_config(),
            training_summary={"locked_test_used_for_selection": False},
            demonstration_manifest_sha256="b" * 64,
        )

    assert referent.read_bytes() == b"referent must remain unchanged"


def test_artifact_bundle_layout_rejects_final_symlink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    referent = tmp_path / "referent.pt"
    target = tmp_path / "student.pt"
    referent.write_bytes(b"referent must remain unchanged")
    try:
        target.symlink_to(referent)
    except OSError:
        monkeypatch.setattr(
            artifact_bundle_module,
            "is_link_or_reparse_point",
            lambda path: Path(path) == target,
        )

    with pytest.raises(ValueError, match="symlink, junction, or reparse"):
        validate_artifact_bundle_layout(target, {"student": target})

    assert referent.read_bytes() == b"referent must remain unchanged"


def test_flow_map_checkpoint_no_overwrite_preserves_concurrent_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    output = tmp_path / "flow-map.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    gipo_checkpoint.write_bytes(b"gipo")
    real_link = checkpoint_module.os.link

    def create_concurrent_output(source, destination, **kwargs):
        destination_path = Path(destination)
        if destination_path == output:
            output.write_bytes(b"concurrent owner")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(checkpoint_module.os, "link", create_concurrent_output)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_flow_map_checkpoint(
            output,
            _flow_map(),
            backbone_checkpoint=backbone_checkpoint,
            gipo_checkpoint=gipo_checkpoint,
            setting_encoder_config=_setting_config(),
            training_summary={"locked_test_used_for_selection": False},
            demonstration_manifest_sha256="b" * 64,
        )

    assert output.read_bytes() == b"concurrent owner"
    assert not list(tmp_path.glob(".flow-map.pt.*.tmp"))


def test_flow_map_checkpoint_rejects_locked_test_selection(tmp_path: Path) -> None:
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    gipo_checkpoint.write_bytes(b"gipo")

    with pytest.raises(ValueError, match="locked_test_used_for_selection"):
        save_flow_map_checkpoint(
            tmp_path / "flow-map.pt",
            _flow_map(),
            backbone_checkpoint=backbone_checkpoint,
            gipo_checkpoint=gipo_checkpoint,
            setting_encoder_config=_setting_config(),
            training_summary={"locked_test_used_for_selection": True},
            demonstration_manifest_sha256="b" * 64,
        )

    with pytest.raises(ValueError, match="nested.locked_test_used_for_selection"):
        save_flow_map_checkpoint(
            tmp_path / "nested-flow-map.pt",
            _flow_map(),
            backbone_checkpoint=backbone_checkpoint,
            gipo_checkpoint=gipo_checkpoint,
            setting_encoder_config=_setting_config(),
            training_summary={
                "locked_test_used_for_selection": False,
                "nested": {"locked_test_used_for_selection": True},
            },
            demonstration_manifest_sha256="b" * 64,
        )


def test_flow_map_checkpoint_rejects_fractional_config_and_model_dimensions(
    tmp_path: Path,
) -> None:
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    checkpoint = tmp_path / "flow-map.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    gipo_checkpoint.write_bytes(b"gipo")
    save_flow_map_checkpoint(
        checkpoint,
        _flow_map(),
        backbone_checkpoint=backbone_checkpoint,
        gipo_checkpoint=gipo_checkpoint,
        setting_encoder_config=_setting_config(),
        training_summary={"locked_test_used_for_selection": False},
        demonstration_manifest_sha256="b" * 64,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["otflow_config"]["data"]["levels"] = 1.5
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="data.levels must be an integer"):
        load_flow_map_checkpoint(checkpoint)

    payload["otflow_config"]["data"]["levels"] = 1
    payload["model_config"]["hidden_dim"] = 8.5
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="hidden_dim must be an integer"):
        load_flow_map_checkpoint(checkpoint)

    payload["model_config"]["hidden_dim"] = 8
    payload["otflow_config"]["data"]["standardize"] = "false"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="data.standardize must be a boolean"):
        load_flow_map_checkpoint(checkpoint)

    payload["otflow_config"]["data"]["standardize"] = True
    payload["otflow_config"]["data"].pop("cond_vol_window")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="must be complete"):
        load_flow_map_checkpoint(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["otflow_config"]["data"]["cond_vol_window"] = 50
    payload["model_config"]["unknown"] = "retired"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="model_config fields are invalid"):
        load_flow_map_checkpoint(checkpoint)

    payload["model_config"].pop("unknown")
    payload["model_config"]["fu_net_type"] = "mlp"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="fu_net_type"):
        load_flow_map_checkpoint(checkpoint)

    payload["model_config"]["fu_net_type"] = "transformer"
    payload["setting_encoder_config"]["nfe_reference"] = 8.5
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="nfe_reference must be an integer"):
        load_flow_map_checkpoint(checkpoint)


def _uniform_gipo_policy(*, context_dim: int = 8) -> GIPOSchedulePolicy:
    encoder_config = _setting_config()
    setting_dim = setting_feature_dim(config=encoder_config)
    student = build_gipo_student_model(
        setting_dim=setting_dim,
        density_dim=DENSITY_BIN_COUNT,
        context_dim=context_dim,
        model_config={
            "hidden_dim": 8,
            "num_layers": 1,
            "attention_heads": 4,
            "dropout": 0.0,
        },
    )
    with torch.no_grad():
        student.head.weight.zero_()
        student.head.bias.zero_()
    return GIPOSchedulePolicy(
        student,
        embedding_normalizer=EmbeddingNormalizer(
            mean=np.zeros(context_dim, dtype=np.float32),
            std=np.ones(context_dim, dtype=np.float32),
        ),
        reference_time_grid=uniform_reference_grid(),
        setting_encoder_config=encoder_config,
        checkpoint_payload={"protocol": "synthetic-test"},
    )


def _write_gipo_checkpoint(path: Path, policy: GIPOSchedulePolicy) -> None:
    torch.save(
        {
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": MODEL_PAYLOAD_VERSION,
            "student_policy_type": "continuous_density",
            "locked_test_used_for_selection": False,
            "context_embedding_kind": "ctx_summary",
            "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
            "density_representation": {
                "density_protocol": DENSITY_PROTOCOL,
                "reference_time_grid": list(policy.reference_time_grid),
            },
            "density_dim": policy.density_dim,
            "setting_dim": policy.setting_dim,
            "setting_encoder_config": policy.setting_encoder_config.to_payload(),
            "embedding_normalizer": policy.embedding_normalizer.to_payload(),
            "context_dim": len(policy.embedding_normalizer.mean),
            "student_model_config": policy.student.model_config(),
            "student_state": policy.student.state_dict(),
            "teacher_training": {
                "teacher_target": "metric_vector",
                "teacher_metric_targets": ["u_comp_uniform"],
                "teacher_metric_target_protocol": "family_metric_utility_vector",
                "teacher_metric_mask_protocol": "row_valid_component_mask",
                "teacher_scalarization": "weighted_metric_average",
                "teacher_checkpoint_selection": {
                    "selection_protocol": "weighted_normalized_regret",
                    "locked_test_used_for_selection": False,
                },
            },
        },
        path,
    )


@pytest.mark.parametrize(
    "solver_key, target_nfe, expected_macro_steps",
    [("euler", 4, 4), ("heun", 4, 2)],
)
def test_gipo_schedule_policy_returns_normalized_setting_specific_schedules(
    solver_key: str,
    target_nfe: int,
    expected_macro_steps: int,
) -> None:
    policy = _uniform_gipo_policy()

    schedule = policy.predict(
        torch.zeros(2, 8),
        solver_key=solver_key,
        target_nfe=target_nfe,
    )

    assert schedule.solver_key == solver_key
    assert schedule.target_nfe == target_nfe
    assert schedule.macro_steps == expected_macro_steps
    assert schedule.realized_nfe == target_nfe
    assert schedule.density_mass.shape == (2, DENSITY_BIN_COUNT)
    assert torch.allclose(schedule.density_mass.sum(dim=-1), torch.ones(2))
    assert schedule.time_grid.shape == (2, expected_macro_steps + 1)
    assert torch.equal(schedule.time_grid[:, 0], torch.zeros(2))
    assert torch.equal(schedule.time_grid[:, -1], torch.ones(2))
    assert torch.all(torch.diff(schedule.time_grid, dim=-1) > 0)


def test_gipo_schedule_policy_rejects_incompatible_context_width() -> None:
    policy = _uniform_gipo_policy(context_dim=8)

    with pytest.raises(ValueError, match="normalizer is incompatible"):
        policy.predict(torch.zeros(1, 7), solver_key="euler", target_nfe=4)


def test_gipo_schedule_loader_rejects_nested_locked_test_selection(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gipo.pt"
    _write_gipo_checkpoint(checkpoint, _uniform_gipo_policy())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["teacher_training"]["teacher_checkpoint_selection"][
        "locked_test_used_for_selection"
    ] = True
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="locked_test_used_for_selection"):
        load_gipo_schedule_policy(checkpoint)


def test_gipo_schedule_loader_requires_nested_locked_test_selection_flag(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "gipo.pt"
    _write_gipo_checkpoint(checkpoint, _uniform_gipo_policy())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["teacher_training"]["teacher_checkpoint_selection"].pop(
        "locked_test_used_for_selection"
    )
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="locked_test_used_for_selection"):
        load_gipo_schedule_policy(checkpoint)


def test_gipo_schedule_loader_requires_explicit_context_embedding_kind(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "gipo.pt"
    _write_gipo_checkpoint(checkpoint, _uniform_gipo_policy())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload.pop("context_embedding_kind")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="explicit context_embedding_kind"):
        load_gipo_schedule_policy(checkpoint)


def test_gipo_schedule_loader_requires_complete_setting_encoder_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "gipo.pt"
    _write_gipo_checkpoint(checkpoint, _uniform_gipo_policy())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["setting_encoder_config"].pop("rope_frequencies")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="missing=.*rope_frequencies"):
        load_gipo_schedule_policy(checkpoint)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda config: config.update({"unknown": 1}),
        lambda config: config.pop("hidden_dim"),
        lambda config: config.update({"setting_dim": 999}),
        lambda config: config.update({"dropout": float("nan")}),
    ),
)
def test_gipo_schedule_loader_requires_complete_student_model_config(
    tmp_path: Path,
    mutation,
) -> None:
    checkpoint = tmp_path / "gipo.pt"
    _write_gipo_checkpoint(checkpoint, _uniform_gipo_policy())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mutation(payload["student_model_config"])
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="model configuration|model_config"):
        load_gipo_schedule_policy(checkpoint)


def test_gipo_schedule_policy_rejects_overflowing_normalization() -> None:
    policy = _uniform_gipo_policy()

    with pytest.raises(ValueError, match="normalization produced non-finite"):
        policy.predict(
            torch.full((1, 8), 1e308, dtype=torch.float64),
            solver_key="euler",
            target_nfe=4,
        )


def test_gipo_schedule_policy_rejects_invalid_student_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _uniform_gipo_policy()
    monkeypatch.setattr(
        policy.student,
        "density_mass",
        lambda setting, context: torch.full(
            (int(context.shape[0]), DENSITY_BIN_COUNT),
            float("nan"),
            device=context.device,
            dtype=context.dtype,
        ),
    )

    with pytest.raises(ValueError, match="density must be finite"):
        policy.predict(torch.zeros(1, 8), solver_key="euler", target_nfe=4)


def test_gipo_schedule_loader_migrates_compatible_student(tmp_path: Path) -> None:
    policy = _uniform_gipo_policy()
    model_config = dict(policy.student.model_config())
    model_config["hidden_layers"] = model_config.pop("num_layers")
    model_config.update(
        {
            "num_series": 0,
            "series_feature_dim": 0,
            "series_conditioning": "none_context_only",
        }
    )
    checkpoint = tmp_path / "gipo-compatibility.pt"
    torch.save(
        {
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": 6,
            "student_policy_type": "continuous_density",
            "locked_test_used_for_selection": False,
            "context_embedding_kind": "ctx_summary",
            "student_architecture": ARCHITECTURE_DENSITY_QUERY_TRANSFORMER,
            "density_representation": {
                "density_protocol": DENSITY_PROTOCOL,
                "reference_time_grid": list(policy.reference_time_grid),
            },
            "density_dim": policy.density_dim,
            "setting_dim": policy.setting_dim,
            "setting_encoder_config": policy.setting_encoder_config.to_payload(),
            "embedding_normalizer": policy.embedding_normalizer.to_payload(),
            "context_dim": 8,
            "student_model_config": model_config,
            "student_state": policy.student.state_dict(),
            "pseudo_target_nfe_values": [6, 10],
            "student_training_mode": "seen_plus_unseen_pseudo",
            "student_pseudo_distillation": {
                "enabled": True,
                "pseudo_target_weight": 0.25,
                "student_teacher_score_include_pseudo": True,
                "locked_test_used_for_pseudo": False,
            },
            "student_objective_settings": {
                "student_teacher_score_include_pseudo": True,
            },
            "nfe_sequence_diagnostics": {"student_pseudo_rows": {"row_count": 2}},
            "teacher_training": {
                "series_conditioning": "none_context_only",
                "teacher_target": "metric_vector",
                "teacher_metric_targets": ["u_comp_uniform"],
                "teacher_metric_target_protocol": "family_metric_utility_vector",
                "teacher_metric_mask_protocol": "row_valid_component_mask",
                "teacher_scalarization": "weighted_metric_average",
                "teacher_checkpoint_selection": {
                    "selection_protocol": "weighted_normalized_regret",
                    "locked_test_used_for_selection": False,
                },
            },
            "student_target_summary": {"series_conditioning": "none_context_only"},
            "student_training": {
                "series_conditioning": "none_context_only",
                "student_pseudo_target_summary": {
                    "pseudo_distillation_used": True,
                    "pseudo_target_weight": 0.25,
                    "pseudo_context_setting_count": 2,
                    "pseudo_target_nfes": [6, 10],
                    "pseudo_split_phases": ["train_tuning"],
                    "series_conditioning": "none_context_only",
                },
                "pseudo_distillation_used": True,
                "pseudo_target_weight": 0.25,
                "student_teacher_score_include_pseudo": True,
                "student_target_summary": {"series_conditioning": "none_context_only"},
                "student_validation_target_summary": {
                    "series_conditioning": "none_context_only"
                },
                "student_optimizer": {"student_teacher_score_include_pseudo": True},
                "losses": [
                    {
                        "student_pseudo_kl_ce_loss": 0.1,
                        "student_pseudo_weighted_loss": 0.025,
                        "student_pseudo_teacher_score_z_mean": 0.2,
                        "student_pseudo_teacher_score_mean": 0.3,
                    }
                ],
            },
            "series_conditioning": "none_context_only",
            "series_index_map": {"unused": 0},
        },
        checkpoint,
    )

    loaded = load_gipo_schedule_policy(checkpoint)
    expected = policy.predict(torch.zeros(2, 8), solver_key="euler", target_nfe=4)
    actual = loaded.predict(torch.zeros(2, 8), solver_key="euler", target_nfe=4)

    metadata = loaded.checkpoint_payload
    assert metadata["source_model_payload_version"] == 6
    assert metadata["student_training_mode"] == "seen_plus_unseen_target"
    assert metadata["unseen_target_nfe_values"] == [6, 10]
    assert metadata["student_unseen_target_distillation"]["unseen_target_weight"] == 0.25
    assert metadata["nfe_sequence_diagnostics"]["student_unseen_target_rows"] == {
        "row_count": 2
    }
    unseen_summary = metadata["student_training"]["student_unseen_target_summary"]
    assert unseen_summary["unseen_context_setting_count"] == 2
    assert metadata["student_training"]["losses"][0][
        "student_unseen_target_weighted_loss"
    ] == 0.025
    assert "pseudo" not in json.dumps(metadata, sort_keys=True)
    assert "series_conditioning" not in json.dumps(metadata, sort_keys=True)
    assert torch.equal(actual.density_mass, expected.density_mass)
    assert torch.equal(actual.time_grid, expected.time_grid)


def test_gipo_compatibility_migration_rejects_ambiguous_or_invalid_fields() -> None:
    base = {
        "model_payload_version": 6,
        "student_model_config": {
            "hidden_layers": 1,
            "num_series": 0,
            "series_feature_dim": 0,
            "series_conditioning": "none_context_only",
        },
    }
    with pytest.raises(ValueError, match="both"):
        normalize_gipo_checkpoint_payload(
            {
                **base,
                "pseudo_target_nfe_values": [6],
                "unseen_target_nfe_values": [6],
            }
        )
    with pytest.raises(ValueError, match="integer"):
        normalize_gipo_checkpoint_payload(
            {
                **base,
                "student_model_config": {
                    **base["student_model_config"],
                    "hidden_layers": True,
                },
            }
        )


class _UniformSchedulePolicy:
    def __init__(self):
        self.student = torch.nn.Identity()
        self.density_dim = DENSITY_BIN_COUNT
        self.reference_time_grid = uniform_reference_grid()
        self.setting_encoder_config = _setting_config()
        self.context_embedding_kind = "ctx_summary"
        self.context_summaries: list[torch.Tensor] = []

    def context_embedding_from_cache(self, cache):
        if isinstance(cache, dict):
            return cache[self.context_embedding_kind]
        return getattr(cache, self.context_embedding_kind)

    def predict(
        self,
        context_summary: torch.Tensor,
        *,
        solver_key: str,
        target_nfe: int,
    ) -> GIPOSchedule:
        self.context_summaries.append(context_summary.detach().clone())
        setting = DistillationSetting(solver_key, target_nfe)
        macro_steps = target_nfe if solver_key not in {"heun", "midpoint_rk2"} else target_nfe // 2
        batch = int(context_summary.shape[0])
        density = torch.full(
            (batch, DENSITY_BIN_COUNT),
            1.0 / DENSITY_BIN_COUNT,
            device=context_summary.device,
            dtype=context_summary.dtype,
        )
        grid = torch.linspace(
            0.0,
            1.0,
            macro_steps + 1,
            device=context_summary.device,
            dtype=context_summary.dtype,
        ).unsqueeze(0).expand(batch, -1)
        return GIPOSchedule(
            solver_key=setting.solver_key,
            target_nfe=setting.target_nfe,
            macro_steps=macro_steps,
            realized_nfe=target_nfe,
            density_mass=density,
            time_grid=grid,
        )


def test_collect_flow_map_demonstrations_with_tiny_frozen_models(tmp_path: Path) -> None:
    torch.manual_seed(13)
    backbone = OTFlow(_tiny_config()).eval()
    contexts = DistillationContexts(
        context_ids=("context-0",),
        histories=torch.randn(1, 2, 4),
    )

    manifest_path = collect_flow_map_demonstrations(
        backbone,
        _UniformSchedulePolicy(),  # type: ignore[arg-type]
        contexts,
        settings=(DistillationSetting("euler", 2),),
        output_dir=tmp_path / "demonstrations",
        split_phase="train",
        scenario_key="synthetic",
        benchmark_family="synthetic",
        backbone_checkpoint_sha256="c" * 64,
        gipo_checkpoint_sha256="d" * 64,
        rollouts_per_context=1,
        context_batch_size=1,
        shard_contexts=1,
        seed=17,
    )

    manifest = load_demonstration_manifest(manifest_path)
    assert manifest_path.name == DEMONSTRATION_MANIFEST_NAME
    assert manifest["metadata"]["locked_test_used"] is False
    assert manifest["metadata"]["settings"] == [{"solver_key": "euler", "target_nfe": 2}]
    assert len(manifest["context_shards"]) == 1
    assert len(manifest["trajectory_shards"]) == 1
    trajectory_path = manifest["trajectory_shards"][0]["resolved_path"]
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        assert trajectory["initial_state"].shape == (1, 4)
        assert trajectory["states"].shape == (1, 3, 4)
        assert trajectory["time_grid"].shape == (1, 3)
        assert trajectory["density_mass"].shape == (1, DENSITY_BIN_COUNT)
        assert np.array_equal(trajectory["initial_state"], trajectory["states"][:, 0])

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_flow_map_demonstrations(
            backbone,
            _UniformSchedulePolicy(),  # type: ignore[arg-type]
            contexts,
            settings=(DistillationSetting("euler", 2),),
            output_dir=tmp_path / "demonstrations",
            split_phase="train",
            scenario_key="synthetic",
            benchmark_family="synthetic",
            backbone_checkpoint_sha256="c" * 64,
            gipo_checkpoint_sha256="d" * 64,
            rollouts_per_context=1,
            context_batch_size=1,
            shard_contexts=1,
            seed=17,
        )


def test_collection_uses_common_noise_across_settings_and_builds_store(
    tmp_path: Path,
) -> None:
    torch.manual_seed(131)
    backbone = OTFlow(_tiny_config()).eval()
    settings = (
        DistillationSetting("euler", 2),
        DistillationSetting("heun", 2),
    )
    manifest_path = collect_flow_map_demonstrations(
        backbone,
        _UniformSchedulePolicy(),  # type: ignore[arg-type]
        DistillationContexts(
            context_ids=("context-0", "context-1"),
            histories=torch.randn(2, 2, 4),
        ),
        settings=settings,
        output_dir=tmp_path / "multi-setting-demonstrations",
        split_phase="train_tuning",
        scenario_key="synthetic",
        benchmark_family="synthetic",
        backbone_checkpoint_sha256="c" * 64,
        gipo_checkpoint_sha256="d" * 64,
        rollouts_per_context=2,
        context_batch_size=1,
        shard_contexts=1,
        seed=37,
    )

    manifest = load_demonstration_manifest(manifest_path)
    seeds_by_context: dict[int, list[tuple[int, ...]]] = {}
    for record in manifest["trajectory_shards"]:
        with np.load(record["resolved_path"], allow_pickle=False) as shard:
            context_index = int(shard["context_index"][0])
            seeds_by_context.setdefault(context_index, []).append(
                tuple(int(value) for value in shard["noise_seed"].tolist())
            )
    assert set(seeds_by_context) == {0, 1}
    assert all(len(seed_rows) == len(settings) for seed_rows in seeds_by_context.values())
    assert all(len(set(seed_rows)) == 1 for seed_rows in seeds_by_context.values())

    store = DemonstrationStore(
        manifest_path,
        validation_fraction=0.5,
        split_seed=41,
    )
    assert store.expected_settings == {
        (setting.solver_key, setting.target_nfe) for setting in settings
    }
    assert store.split_context_count("train") == 1
    assert store.split_context_count("validation") == 1


def test_demonstration_collection_rejects_locked_test_before_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "locked-test-demonstrations"
    with pytest.raises(ValueError, match="non-test"):
        collect_flow_map_demonstrations(
            OTFlow(_tiny_config()).eval(),
            _UniformSchedulePolicy(),  # type: ignore[arg-type]
            DistillationContexts(("context-0",), torch.zeros(1, 2, 4)),
            settings=(DistillationSetting("euler", 2),),
            output_dir=output_dir,
            split_phase="locked_test",
            scenario_key="synthetic",
            benchmark_family="synthetic",
            backbone_checkpoint_sha256="c" * 64,
            gipo_checkpoint_sha256="d" * 64,
        )
    assert not output_dir.exists()


def test_flow_map_sampler_reports_exactly_one_map_evaluation() -> None:
    torch.manual_seed(19)
    backbone = OTFlow(_tiny_config()).eval()
    flow_map = _flow_map().eval()
    policy = _UniformSchedulePolicy()
    sampler = FlowMapSampler(
        backbone,
        flow_map,
        setting_encoder_config=_setting_config(),
        gipo_policy=policy,  # type: ignore[arg-type]
    )
    initial = torch.randn(2, backbone.cfg.sample_state_dim)
    history = torch.randn(2, backbone.cfg.history_len, backbone.cfg.context_dim)
    expected_cache = backbone.backbone.precompute(history)

    result = sampler.map_state(
        initial,
        solver_key="euler",
        target_nfe=4,
        hist=history,
    )

    assert result.sample.shape == initial.shape
    assert result.model_evaluations == 1
    assert result.teacher_realized_nfe == 4
    assert len(policy.context_summaries) == 1
    assert torch.equal(policy.context_summaries[0], expected_cache.ctx_summary)
    assert not torch.equal(policy.context_summaries[0], expected_cache.summary)

    manual_density = torch.full(
        (2, DENSITY_BIN_COUNT),
        1.0 / DENSITY_BIN_COUNT,
    )
    sampler.map_state(
        initial,
        solver_key="euler",
        target_nfe=4,
        density_mass=manual_density,
        hist=history,
    )
    assert len(policy.context_summaries) == 1


def test_load_flow_map_sampler_binds_and_runs_verified_gipo(tmp_path: Path) -> None:
    torch.manual_seed(191)
    backbone = OTFlow(_tiny_config()).eval()
    backbone_checkpoint = tmp_path / "backbone.pt"
    torch.save(
        {"cfg": backbone.cfg.to_dict(), "model_state": backbone.state_dict()},
        backbone_checkpoint,
    )
    policy = _uniform_gipo_policy()
    gipo_checkpoint = tmp_path / "gipo.pt"
    _write_gipo_checkpoint(gipo_checkpoint, policy)
    flow_map_checkpoint = tmp_path / "flow-map.pt"
    save_flow_map_checkpoint(
        flow_map_checkpoint,
        _flow_map().eval(),
        backbone_checkpoint=backbone_checkpoint,
        gipo_checkpoint=gipo_checkpoint,
        setting_encoder_config=_setting_config(),
        training_summary={"locked_test_used_for_selection": False},
        demonstration_manifest_sha256="a" * 64,
    )

    sampler, metadata = load_flow_map_sampler(
        flow_map_checkpoint,
        backbone_checkpoint=backbone_checkpoint,
        gipo_checkpoint=gipo_checkpoint,
    )
    future = sampler.sample_future(
        torch.randn(2, backbone.cfg.history_len, backbone.cfg.context_dim),
        solver_key="heun",
        target_nfe=4,
    )

    assert sampler.gipo_policy is not None
    assert future.shape == (2, 1, backbone.cfg.snapshot_dim)
    assert metadata["gipo_checkpoint_sha256"]
    assert "model_state" not in metadata


def test_flow_map_sampler_reshapes_non_autoregressive_future_blocks() -> None:
    torch.manual_seed(20)
    cfg = _tiny_config()
    cfg.apply_overrides(rollout_mode="non_ar", future_block_len=2)
    backbone = OTFlow(cfg).eval()
    flow_map = EndpointFlowMap(
        cfg,
        setting_dim=setting_feature_dim(config=_setting_config()),
        density_dim=DENSITY_BIN_COUNT,
    ).eval()
    sampler = FlowMapSampler(
        backbone,
        flow_map,
        setting_encoder_config=_setting_config(),
    )
    density = torch.full((2, DENSITY_BIN_COUNT), 1.0 / DENSITY_BIN_COUNT)

    future = sampler.sample_future(
        torch.randn(2, cfg.history_len, cfg.context_dim),
        solver_key="euler",
        target_nfe=4,
        density_mass=density,
    )

    assert future.shape == (2, 2, cfg.snapshot_dim)


def test_demonstration_store_uses_context_disjoint_splits_and_samples_trajectory_states(
    tmp_path: Path,
) -> None:
    store = DemonstrationStore(
        _write_training_manifest(tmp_path),
        validation_fraction=0.5,
        split_seed=23,
    )

    assert store.split_context_count("train") == 1
    assert store.split_context_count("validation") == 1
    assert set(store.split_by_context.values()) == {"train", "validation"}
    train_contexts = {
        context_index
        for context_index, split in store.split_by_context.items()
        if split == "train"
    }
    validation_contexts = {
        context_index
        for context_index, split in store.split_by_context.items()
        if split == "validation"
    }
    assert train_contexts.isdisjoint(validation_contexts)

    batch = store.sample_batch(
        16,
        split="train",
        generator=np.random.default_rng(29),
        device=torch.device("cpu"),
    )
    assert batch.state.shape == (16, 4)
    assert batch.teacher_endpoint.shape == (16, 4)
    assert batch.density_mass.shape == (16, DENSITY_BIN_COUNT)
    assert batch.setting.shape == (16, setting_feature_dim(config=_setting_config()))
    assert set(batch.source_time.flatten().tolist()) <= {0.0, 0.5}
    assert torch.allclose(batch.density_mass.sum(dim=-1), torch.ones(16))

    with pytest.raises(ValueError, match="No train rows"):
        store.sample_batch(
            1,
            split="train",
            generator=np.random.default_rng(31),
            device=torch.device("cpu"),
            setting=("heun", 2),
        )


def test_demonstration_manifest_rejects_duplicate_context_ids_before_splitting(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unique"):
        _write_training_manifest(tmp_path, duplicate_context_ids=True)


def test_demonstration_store_requires_complete_setting_encoder_config(
    tmp_path: Path,
) -> None:
    path = _write_training_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["setting_encoder_config"].pop("rope_frequencies")
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing=.*rope_frequencies"):
        DemonstrationStore(path)


@pytest.mark.parametrize(
    "histories",
    (
        np.zeros((1, 2, 4), dtype=np.int64),
        np.zeros((1, 2, 4), dtype=np.bool_),
        np.zeros((1, 2, 4), dtype=np.complex64),
    ),
)
def test_context_npz_rejects_non_floating_histories(
    tmp_path: Path,
    histories: np.ndarray,
) -> None:
    path = tmp_path / "contexts.npz"
    np.savez(path, context_ids=np.asarray(["context-0"]), histories=histories)

    with pytest.raises(ValueError, match="histories must contain floating-point"):
        load_distillation_contexts(path)


def test_context_npz_rejects_non_floating_conditions(tmp_path: Path) -> None:
    path = tmp_path / "contexts.npz"
    np.savez(
        path,
        context_ids=np.asarray(["context-0"]),
        histories=np.zeros((1, 2, 4), dtype=np.float32),
        conditions=np.zeros((1, 2), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="conditions must contain floating-point"):
        load_distillation_contexts(path)


def test_demonstration_output_rejects_indirect_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indirect = tmp_path / "junction"
    monkeypatch.setattr(
        demonstration_module,
        "first_link_or_reparse_component",
        lambda path, *, root: indirect,
    )

    with pytest.raises(ValueError, match="symlink, junction, or reparse"):
        demonstration_module._validate_output_root(
            indirect / "demonstrations", overwrite=False
        )


def test_demonstration_collection_lock_rejects_concurrent_process(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_demonstration_collection_lock,
        args=(str(target), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=30), "Child process did not acquire collection lock."
        with pytest.raises(RuntimeError, match="Another demonstration collector"):
            collect_flow_map_demonstrations(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                DistillationContexts(("context-0",), torch.zeros(1, 2, 4)),
                settings=(DistillationSetting("euler", 2),),
                output_dir=target,
                split_phase="train",
                scenario_key="synthetic",
                benchmark_family="synthetic",
                backbone_checkpoint_sha256="c" * 64,
                gipo_checkpoint_sha256="d" * 64,
            )
    finally:
        release.set()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    assert process.exitcode == 0
    lock_path = demonstration_module._collection_lock_path(target)
    assert lock_path.parent == target.parent
    assert lock_path.is_file()
    with demonstration_module._exclusive_collection_lock(target):
        pass


def test_demonstration_collection_lock_preserves_unrecognized_sidecar(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    lock_path = demonstration_module._collection_lock_path(target)
    original = b"unmanaged lock contents\n"
    lock_path.write_bytes(original)

    with pytest.raises(ValueError, match="unrecognized"):
        with demonstration_module._exclusive_collection_lock(target):
            pytest.fail("An unrecognized lock sidecar must fail closed.")

    assert lock_path.read_bytes() == original


def test_demonstration_collection_lock_recovers_empty_interrupted_sidecar(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    lock_path = demonstration_module._collection_lock_path(target)
    lock_path.touch()

    with demonstration_module._exclusive_collection_lock(target):
        pass
    assert lock_path.read_bytes() == demonstration_module._COLLECTION_LOCK_MARKER

    with demonstration_module._exclusive_collection_lock(target):
        pass


def test_demonstration_promotion_syncs_complete_staging_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-sync"
    _write_minimal_manifest(staging, split_phase="validation_tuning")
    synced_files: list[Path] = []
    synced_directories: list[Path] = []
    real_write_journal = demonstration_module._write_promotion_journal

    monkeypatch.setattr(
        demonstration_module,
        "_fsync_regular_file",
        lambda path: synced_files.append(path),
    )
    monkeypatch.setattr(
        demonstration_module,
        "_fsync_directory",
        lambda path: synced_directories.append(path),
    )

    def assert_synced_before_journal(path, record):
        assert synced_files[-1] == staging / DEMONSTRATION_MANIFEST_NAME
        assert {path.name for path in synced_files} == {
            DEMONSTRATION_MANIFEST_NAME,
            "contexts_00000.npz",
            "trajectories_euler_nfe2_00000.npz",
        }
        assert staging / "shards" in synced_directories
        assert staging in synced_directories
        assert synced_directories.index(staging / "shards") < synced_directories.index(
            staging
        )
        real_write_journal(path, record)

    monkeypatch.setattr(
        demonstration_module,
        "_write_promotion_journal",
        assert_synced_before_journal,
    )

    demonstration_module._prepare_promotion_journal(staging, target)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory fsync semantics")
def test_posix_parent_directory_fsync_failure_is_surfaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fsync(descriptor):
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(demonstration_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="directory fsync failure"):
        demonstration_module._fsync_parent_directory(tmp_path / "entry")


def test_demonstration_promotion_recovers_previous_artifact_after_interruption(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(target, split_phase="train")
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    backup = target.parent / str(record["backup_name"])
    os.replace(target, backup)

    demonstration_module._recover_interrupted_promotion(target)

    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "train"
    assert not staging.exists()
    assert not backup.exists()
    assert not demonstration_module._promotion_journal_path(target).exists()


def test_demonstration_promotion_finishes_installed_artifact_after_interruption(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(target, split_phase="train")
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    backup = target.parent / str(record["backup_name"])
    os.replace(target, backup)
    os.replace(staging, target)

    demonstration_module._recover_interrupted_promotion(target)

    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "validation_tuning"
    assert not backup.exists()
    assert not demonstration_module._promotion_journal_path(target).exists()


def test_demonstration_promotion_rolls_back_when_prior_target_was_absent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    assert record["previous_kind"] == "absent"

    demonstration_module._recover_interrupted_promotion(target)

    assert not target.exists()
    assert not staging.exists()
    assert not demonstration_module._promotion_journal_path(target).exists()


def test_demonstration_promotion_finishes_install_when_prior_target_was_absent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    assert record["previous_kind"] == "absent"
    os.replace(staging, target)

    demonstration_module._recover_interrupted_promotion(target)

    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "validation_tuning"
    assert not demonstration_module._promotion_journal_path(target).exists()


def test_demonstration_promotion_preserves_unmanaged_backup_without_blocking(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(target, split_phase="train")
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    backup = target.parent / str(record["backup_name"])
    os.replace(target, backup)
    os.replace(staging, target)
    unmanaged = backup / "unmanaged.npy"
    np.save(unmanaged, np.asarray([1.0], dtype=np.float32))

    demonstration_module._recover_interrupted_promotion(target)

    assert unmanaged.is_file()
    assert backup.is_dir()
    assert not demonstration_module._promotion_journal_path(target).exists()


def test_demonstration_promotion_tolerates_partially_cleaned_obsolete_backup(
    tmp_path: Path,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(target, split_phase="train")
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    backup = target.parent / str(record["backup_name"])
    os.replace(target, backup)
    os.replace(staging, target)
    (backup / "shards" / "contexts_00000.npz").unlink()

    demonstration_module._recover_interrupted_promotion(target)

    assert backup.is_dir()
    assert not demonstration_module._promotion_journal_path(target).exists()
    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "validation_tuning"


def test_demonstration_promotion_retains_journal_until_obsolete_cleanup_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-interrupted"
    _write_minimal_manifest(target, split_phase="train")
    _write_minimal_manifest(staging, split_phase="validation_tuning")

    record = demonstration_module._prepare_promotion_journal(staging, target)
    backup = target.parent / str(record["backup_name"])
    os.replace(target, backup)
    os.replace(staging, target)
    journal = demonstration_module._promotion_journal_path(target)

    real_cleanup = demonstration_module._cleanup_obsolete_artifact

    def interrupt_cleanup(root, expected_manifest_sha256):
        assert journal.exists()
        raise KeyboardInterrupt("simulated interruption during transaction cleanup")

    monkeypatch.setattr(
        demonstration_module,
        "_cleanup_obsolete_artifact",
        interrupt_cleanup,
    )

    with pytest.raises(KeyboardInterrupt, match="during transaction cleanup"):
        demonstration_module._recover_interrupted_promotion(target)

    assert journal.exists()
    assert backup.is_dir()
    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "validation_tuning"
    monkeypatch.setattr(
        demonstration_module,
        "_cleanup_obsolete_artifact",
        real_cleanup,
    )
    demonstration_module._recover_interrupted_promotion(target)
    assert not journal.exists()
    assert not backup.exists()


def test_demonstration_promotion_rolls_back_when_installation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-failure"
    _write_minimal_manifest(target, split_phase="train")
    _write_minimal_manifest(staging, split_phase="validation_tuning")
    real_replace = os.replace

    def fail_staging_install(source, destination):
        if Path(source) == staging and Path(destination) == target:
            raise OSError("simulated staging installation failure")
        return real_replace(source, destination)

    monkeypatch.setattr(demonstration_module.os, "replace", fail_staging_install)

    with pytest.raises(OSError, match="staging installation failure"):
        demonstration_module._promote_staged_artifact(
            staging,
            target,
            overwrite=True,
        )

    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "train"
    assert not staging.exists()
    assert not demonstration_module._promotion_journal_path(target).exists()
    assert not list(tmp_path.glob(".demonstrations.backup-*"))


def test_demonstration_promotion_without_overwrite_preserves_concurrent_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demonstrations"
    staging = tmp_path / ".demonstrations.staging-concurrent"
    _write_minimal_manifest(staging, split_phase="validation_tuning")
    real_install = demonstration_module._install_path_no_replace

    def create_target_then_install(source: Path, destination: Path) -> None:
        _write_minimal_manifest(destination, split_phase="train")
        real_install(source, destination)

    monkeypatch.setattr(
        demonstration_module,
        "_install_path_no_replace",
        create_target_then_install,
    )

    with pytest.raises(FileExistsError, match="concurrently created"):
        demonstration_module._promote_staged_artifact(
            staging,
            target,
            overwrite=False,
        )

    assert load_demonstration_manifest(target / DEMONSTRATION_MANIFEST_NAME)[
        "metadata"
    ]["split_phase"] == "train"
    assert demonstration_module._promotion_journal_path(target).exists()


@pytest.mark.parametrize("preexisting", (False, True))
def test_flow_map_bundle_rolls_back_checkpoint_when_summary_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    from genode import artifact_bundle

    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"
    if preexisting:
        torch.save(
            {
                "label": "old checkpoint",
                "training_summary": {"locked_test_used_for_selection": False},
            },
            checkpoint,
        )
        summary.write_text(
            json.dumps(
                {
                    "locked_test_used_for_selection": False,
                    "status": "completed",
                    "checkpoint_name": checkpoint.name,
                    "checkpoint_sha256": file_sha256(checkpoint),
                }
            ),
            encoding="utf-8",
        )
    old_checkpoint = checkpoint.read_bytes() if preexisting else b""
    old_summary = summary.read_bytes() if preexisting else b""
    real_install = artifact_bundle._link_without_overwrite

    def write_checkpoint(path: Path) -> Path:
        torch.save(
            {
                "label": "new checkpoint",
                "training_summary": {"locked_test_used_for_selection": False},
            },
            path,
        )
        return path

    def fail_summary_promotion(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == summary and ".bundle-stage-" in source_path.name:
            raise OSError("simulated summary promotion failure")
        return real_install(source_path, destination_path)

    monkeypatch.setattr(
        "genode.artifact_bundle._link_without_overwrite",
        fail_summary_promotion,
    )
    with pytest.raises(OSError, match="summary promotion failure"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
            overwrite=preexisting,
        )

    if preexisting:
        assert checkpoint.read_bytes() == old_checkpoint
        assert summary.read_bytes() == old_summary
    else:
        assert not checkpoint.exists()
        assert not summary.exists()
    assert not bundle_journal_path(checkpoint).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))
    assert not list(tmp_path.glob(".*.bundle-backup-*"))


def test_flow_map_bundle_restart_finishes_committed_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from genode import artifact_bundle

    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"

    def write_checkpoint(path: Path) -> Path:
        torch.save(
            {"training_summary": {"locked_test_used_for_selection": False}},
            path,
        )
        return path

    real_cleanup = artifact_bundle._cleanup_finalized_transaction
    monkeypatch.setattr(
        artifact_bundle,
        "_cleanup_finalized_transaction",
        lambda **_: (_ for _ in ()).throw(
            OSError("simulated interruption after commit")
        ),
    )
    with pytest.raises(OSError, match="interruption after commit"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
        )

    assert bundle_journal_path(checkpoint).is_file()
    assert checkpoint.is_file()
    assert summary.is_file()
    monkeypatch.setattr(
        artifact_bundle,
        "_cleanup_finalized_transaction",
        real_cleanup,
    )
    recover_flow_map_bundle(checkpoint, summary)
    assert not bundle_journal_path(checkpoint).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


def test_flow_map_bundle_rejects_edited_summary_semantics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"

    def write_checkpoint(path: Path) -> Path:
        torch.save(
            {"training_summary": {"locked_test_used_for_selection": False}},
            path,
        )
        return path

    _write_flow_map_bundle(
        checkpoint_path=checkpoint,
        summary_path=summary,
        training_summary={"locked_test_used_for_selection": False},
        checkpoint_writer=write_checkpoint,
    )
    edited = json.loads(summary.read_text(encoding="utf-8"))
    edited["locked_test_used_for_selection"] = True
    summary.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON training summaries do not match"):
        validate_flow_map_bundle(checkpoint, summary)


def test_flow_map_bundle_cleans_owned_partial_stage_before_journaling(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"

    def fail_checkpoint_write(path: Path) -> Path:
        path.write_bytes(b"partial owned stage")
        raise OSError("simulated checkpoint writer failure")

    with pytest.raises(OSError, match="checkpoint writer failure"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=fail_checkpoint_write,
        )

    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))
    assert not bundle_journal_path(checkpoint).exists()


def test_flow_map_bundle_rechecks_source_identity_before_commit(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"
    source = tmp_path / "gipo.pt"
    replacement = tmp_path / "gipo-replacement.pt"
    source.write_bytes(b"source used for training")
    replacement.write_bytes(b"concurrent replacement")
    expected_hash = file_sha256(source)

    def write_checkpoint(path: Path) -> Path:
        torch.save(
            {"training_summary": {"locked_test_used_for_selection": False}},
            path,
        )
        os.replace(replacement, source)
        return path

    def validate_source() -> None:
        if file_sha256(source) != expected_hash:
            raise ValueError("GIPO source checkpoint changed before commit")

    with pytest.raises(ValueError, match="GIPO source checkpoint changed"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
            precommit_validator=validate_source,
        )

    assert source.read_bytes() == b"concurrent replacement"
    assert not checkpoint.exists()
    assert not summary.exists()
    assert not bundle_journal_path(checkpoint).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


@pytest.mark.parametrize(
    ("source_kind", "error_pattern"),
    (
        ("gipo", "GIPO source checkpoint changed"),
        ("demonstration_manifest", "Demonstration manifest changed"),
        ("demonstration_shard", "Demonstration shard .* changed"),
    ),
)
def test_flow_map_training_aborts_when_bound_source_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    error_pattern: str,
) -> None:
    manifest = tmp_path / "demonstrations" / DEMONSTRATION_MANIFEST_NAME
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    backbone = tmp_path / "backbone.pt"
    gipo = tmp_path / "gipo.pt"
    replacement = tmp_path / "replacement-gipo.pt"
    manifest_replacement = tmp_path / "replacement-manifest.json"
    shard = manifest.parent / "contexts.npz"
    shard_replacement = tmp_path / "replacement-contexts.npz"
    backbone.write_bytes(b"validated backbone")
    gipo.write_bytes(b"validated gipo")
    replacement.write_bytes(b"replacement gipo")
    manifest_replacement.write_text('{"replacement":true}', encoding="utf-8")
    shard.write_bytes(b"validated demonstration shard")
    shard_replacement.write_bytes(b"replacement demonstration shard")
    setting_config = _setting_config()
    reference_grid = uniform_reference_grid(DENSITY_BIN_COUNT)
    store = SimpleNamespace(
        manifest={
            "context_shards": [
                {"resolved_path": shard, "sha256": file_sha256(shard)}
            ],
            "trajectory_shards": [],
        },
        metadata={
            "backbone_checkpoint_sha256": file_sha256(backbone),
            "gipo_checkpoint_sha256": file_sha256(gipo),
            "context_embedding_kind": "ctx_summary",
            "context_binding": context_binding(("f" * 64,)),
            "scenario_key": "cryptos",
            "benchmark_family": "forecast",
        },
        setting_encoder_config=setting_config,
        density_dim=DENSITY_BIN_COUNT,
        density_reference_time_grid=reference_grid,
        state_dim=4,
    )
    gipo_policy = SimpleNamespace(
        setting_encoder_config=setting_config,
        density_dim=DENSITY_BIN_COUNT,
        reference_time_grid=reference_grid,
        context_embedding_kind="ctx_summary",
    )
    backbone_model = SimpleNamespace(cfg=SimpleNamespace(sample_state_dim=4))

    def replace_source_during_training(*args, **kwargs):
        if source_kind == "gipo":
            os.replace(replacement, gipo)
        elif source_kind == "demonstration_manifest":
            os.replace(manifest_replacement, manifest)
        else:
            os.replace(shard_replacement, shard)
        return _flow_map(), {
            "locked_test_used_for_selection": False,
            "train_context_count": 1,
            "validation_context_count": 0,
        }

    monkeypatch.setattr(
        "genode.distillation.training.DemonstrationStore",
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        "genode.distillation.training.load_gipo_schedule_policy",
        lambda *args, **kwargs: gipo_policy,
    )
    monkeypatch.setattr(
        "genode.distillation.training.load_checkpoint_model",
        lambda *args, **kwargs: (backbone_model, None),
    )
    monkeypatch.setattr(
        "genode.distillation.training.train_endpoint_flow_map",
        replace_source_during_training,
    )
    output = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"
    with pytest.raises(ValueError, match=error_pattern):
        train_flow_map_main(
            [
                "--demonstration-manifest",
                str(manifest),
                "--backbone-checkpoint",
                str(backbone),
                "--gipo-checkpoint",
                str(gipo),
                "--output-checkpoint",
                str(output),
                "--summary-json",
                str(summary),
                "--device",
                "cpu",
            ]
        )

    if source_kind == "gipo":
        assert gipo.read_bytes() == b"replacement gipo"
    elif source_kind == "demonstration_manifest":
        assert manifest.read_text(encoding="utf-8") == '{"replacement":true}'
    else:
        assert shard.read_bytes() == b"replacement demonstration shard"
    assert not output.exists()
    assert not summary.exists()
    assert not bundle_journal_path(output).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


def test_artifact_bundle_requires_deterministic_anchor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"
    with pytest.raises(ValueError, match="lexicographically first role"):
        recover_artifact_bundle(
            summary,
            {"checkpoint": checkpoint, "summary": summary},
        )


@pytest.mark.parametrize("sidecar_kind", ("lock", "journal"))
def test_flow_map_bundle_rejects_reserved_sidecar_target_without_writes(
    tmp_path: Path,
    sidecar_kind: str,
) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    summary = (
        bundle_lock_path(checkpoint)
        if sidecar_kind == "lock"
        else bundle_journal_path(checkpoint)
    )
    writer_called = False

    def write_checkpoint(path: Path) -> Path:
        nonlocal writer_called
        writer_called = True
        path.write_bytes(b"unexpected")
        return path

    with pytest.raises(ValueError, match="reserved lock or journal sidecar"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
        )

    assert writer_called is False
    assert list(tmp_path.iterdir()) == []


def test_flow_map_bundle_rejects_nested_targets_without_writes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    summary = checkpoint / "training-summary.json"
    writer_called = False

    def write_checkpoint(path: Path) -> Path:
        nonlocal writer_called
        writer_called = True
        path.write_bytes(b"unexpected")
        return path

    with pytest.raises(ValueError, match="may not contain one another"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
        )

    assert writer_called is False
    assert list(tmp_path.iterdir()) == []


def test_flow_map_bundle_restart_rolls_back_prepared_partial_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from genode import artifact_bundle

    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"
    real_install = artifact_bundle._link_without_overwrite
    real_recover = artifact_bundle._recover_locked

    def write_checkpoint(path: Path) -> Path:
        torch.save(
            {"training_summary": {"locked_test_used_for_selection": False}},
            path,
        )
        return path

    def fail_summary_install(source: Path, destination: Path) -> None:
        if destination == summary and ".bundle-stage-" in source.name:
            raise OSError("simulated process interruption")
        real_install(source, destination)

    def leave_prepared_journal(*args, force_abort: bool = False, **kwargs) -> None:
        if force_abort:
            raise OSError("simulated unavailable in-process recovery")
        real_recover(*args, force_abort=force_abort, **kwargs)

    monkeypatch.setattr(
        artifact_bundle,
        "_link_without_overwrite",
        fail_summary_install,
    )
    monkeypatch.setattr(artifact_bundle, "_recover_locked", leave_prepared_journal)
    with pytest.raises(OSError, match="process interruption"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
        )

    assert checkpoint.is_file()
    assert not summary.exists()
    assert bundle_journal_path(checkpoint).is_file()
    monkeypatch.setattr(artifact_bundle, "_link_without_overwrite", real_install)
    monkeypatch.setattr(artifact_bundle, "_recover_locked", real_recover)
    recover_flow_map_bundle(checkpoint, summary)
    assert not checkpoint.exists()
    assert not summary.exists()
    assert not bundle_journal_path(checkpoint).exists()
    assert not list(tmp_path.glob(".*.bundle-stage-*.tmp"))


def test_flow_map_bundle_lock_rejects_concurrent_process(tmp_path: Path) -> None:
    checkpoint = tmp_path / "flow-map.pt"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_artifact_bundle_lock,
        args=(str(checkpoint), ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=30), "Child process did not acquire bundle lock."
        with pytest.raises(RuntimeError, match="Another process is already writing"):
            recover_flow_map_bundle(checkpoint, tmp_path / "flow-map-training.json")
    finally:
        release.set()
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert process.exitcode == 0


def test_flow_map_bundle_preserves_concurrent_unknown_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from genode import artifact_bundle

    checkpoint = tmp_path / "flow-map.pt"
    summary = tmp_path / "flow-map-training.json"
    real_install = artifact_bundle._link_without_overwrite

    def write_checkpoint(path: Path) -> Path:
        torch.save(
            {"training_summary": {"locked_test_used_for_selection": False}},
            path,
        )
        return path

    def create_concurrent_checkpoint(source: Path, destination: Path) -> None:
        if destination == checkpoint and ".bundle-stage-" in source.name:
            checkpoint.write_bytes(b"concurrent owner")
        real_install(source, destination)

    monkeypatch.setattr(
        artifact_bundle,
        "_link_without_overwrite",
        create_concurrent_checkpoint,
    )
    with pytest.raises(FileExistsError, match="appeared concurrently"):
        _write_flow_map_bundle(
            checkpoint_path=checkpoint,
            summary_path=summary,
            training_summary={"locked_test_used_for_selection": False},
            checkpoint_writer=write_checkpoint,
        )

    assert checkpoint.read_bytes() == b"concurrent owner"
    assert bundle_journal_path(checkpoint).is_file()
    assert list(tmp_path.glob(".*.bundle-stage-*.tmp"))


def test_single_step_flow_map_training_is_synthetic_and_selects_on_validation(
    tmp_path: Path,
) -> None:
    torch.manual_seed(31)
    store = DemonstrationStore(
        _write_training_manifest(tmp_path),
        validation_fraction=0.5,
        split_seed=31,
    )
    backbone = OTFlow(_tiny_config()).eval()

    flow_map, summary = train_endpoint_flow_map(
        backbone,
        store,
        steps=1,
        batch_size=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        validation_interval=1,
        validation_shards=1,
        batches_per_shard=1,
        seed=31,
    )

    assert isinstance(flow_map, EndpointFlowMap)
    assert not flow_map.training
    assert summary["best_step"] == 1
    assert summary["train_context_count"] == 1
    assert summary["validation_context_count"] == 1
    assert summary["locked_test_used_for_selection"] is False
    assert summary["checkpoint_export"] == "final_selected_only"
    assert np.isfinite(summary["best_validation_loss"])
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_quality_gate_passes_only_with_familywise_evidence_on_every_primary_metric() -> None:
    report = evaluate_quality_gate(
        _quality_rows(),
        metric_specs=metric_specs_for_scenario("cryptos"),
        candidate_catalog=_quality_candidate_catalog(),
        artifact_binding=_quality_binding(),
        demonstration_context_binding=context_binding(
            (_test_context_fingerprint("training-0"),)
        ),
        quality_protocol=_quality_protocol(),
        config=QualityGateConfig(bootstrap_samples=1_000, seed=37),
    )

    assert report["status"] == "passed"
    assert report["performance_claim"] is True
    assert report["bootstrap_seed"] == 37
    assert report["multiple_testing_correction"] == "holm"
    assert report["all_primary_metrics_required"] is True
    assert report["locked_test_used_for_selection"] is False
    assert len(report["comparisons"]) == 6
    assert all(comparison["paired_context_count"] == 20 for comparison in report["comparisons"])
    assert all(comparison["passed"] for comparison in report["comparisons"])
    assert all(
        comparison["holm_adjusted_p_value"] >= comparison["one_sided_p_value"]
        for comparison in report["comparisons"]
    )


def test_quality_gate_rejects_measurement_row_digest_tamper() -> None:
    rows = _quality_rows()
    rows[0]["measurement_protocol_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="measurement_protocol_sha256"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=37),
        )


def test_quality_gate_rejects_measurement_protocol_artifact_substitution() -> None:
    config = QualityGateConfig(bootstrap_samples=1_000, seed=37)
    measurement_protocol = json.loads(
        json.dumps(_quality_measurement_protocol(config))
    )
    measurement_protocol["quality_contexts_sha256"] = "a" * 64

    with pytest.raises(ValueError, match="does not match"):
        evaluate_quality_gate(
            _quality_rows(),
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            measurement_protocol=measurement_protocol,
            measurement_protocol_sha256=measurement_protocol_sha256(
                measurement_protocol
            ),
            config=config,
        )


def test_quality_gate_rejects_measurement_payload_digest_mismatch() -> None:
    config = QualityGateConfig(bootstrap_samples=1_000, seed=37)
    measurement_protocol = _quality_measurement_protocol(config)

    with pytest.raises(ValueError, match="does not match the validated protocol"):
        evaluate_quality_gate(
            _quality_rows(),
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            measurement_protocol=measurement_protocol,
            measurement_protocol_sha256="0" * 64,
            config=config,
        )


def test_quality_context_and_sample_artifacts_recompute_physical_bindings(
    tmp_path: Path,
) -> None:
    context_ids = ["validation-a", "locked-a"]
    contexts_path = tmp_path / "quality_contexts.npz"
    np.savez_compressed(
        contexts_path,
        context_ids=np.asarray(context_ids, dtype=np.str_),
        split_phases=np.asarray(
            ["validation_tuning", "locked_test"], dtype=np.str_
        ),
        histories=np.asarray([[[0.1]], [[0.2]]], dtype=np.float64),
    )
    contexts = read_quality_contexts(contexts_path)

    sample_path = tmp_path / "quality_samples.npz"
    np.savez_compressed(
        sample_path,
        context_ids=np.asarray(
            ["validation-a", "validation-a", "locked-a", "locked-a"],
            dtype=np.str_,
        ),
        logical_seeds=np.asarray([10, 11, 10, 11], dtype=np.int64),
        initial_states=np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.5, -0.5], [-0.5, 0.5]],
            dtype=np.float64,
        ),
    )
    sample_panel = read_quality_sample_panel(
        sample_path,
        quality_context_binding=contexts,
    )

    assert contexts["artifact_sha256"]
    assert contexts["contexts"][0]["context_fingerprint"] == context_fingerprint(
        np.asarray([[0.2]], dtype=np.float32)
    )
    assert sample_panel["replicate_count"] == 2
    assert sample_panel["sample_count"] == 4


def test_quality_context_ids_are_globally_unique_across_splits(tmp_path: Path) -> None:
    contexts_path = tmp_path / "quality_contexts.npz"
    np.savez_compressed(
        contexts_path,
        context_ids=np.asarray(["shared", "shared"], dtype=np.str_),
        split_phases=np.asarray(
            ["validation_tuning", "locked_test"], dtype=np.str_
        ),
        histories=np.asarray([[[0.1]], [[0.2]]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="ids must be unique"):
        read_quality_contexts(contexts_path)


def test_quality_candidate_catalog_binds_registered_fixed_grid() -> None:
    catalog = _quality_candidate_catalog()
    fixed = next(candidate for candidate in catalog if candidate["method"] == "fixed")
    fixed["execution"]["time_grid"][1] = 0.123
    fixed["execution"]["time_grid_sha256"] = _time_grid_sha256(
        fixed["execution"]["time_grid"]
    )

    with pytest.raises(ValueError, match="registered scheduler"):
        _normalize_candidate_catalog(catalog)


@pytest.mark.parametrize("method", ("flow_map", "gipo", "fixed"))
def test_quality_candidate_catalog_requires_search_breadth(method: str) -> None:
    catalog = _quality_candidate_catalog()
    removed = False
    narrowed = []
    for candidate in catalog:
        if candidate["method"] == method and not removed:
            removed = True
            continue
        narrowed.append(candidate)

    with pytest.raises(ValueError, match="at least two candidates"):
        _normalize_candidate_catalog(narrowed)


def test_quality_candidate_catalog_rejects_duplicate_solver_nfe_search() -> None:
    catalog = json.loads(json.dumps(_quality_candidate_catalog()))
    second_gipo = next(
        candidate
        for candidate in catalog
        if candidate["candidate_key"] == "gipo-decoy"
    )
    second_gipo["target_nfe"] = 4

    with pytest.raises(ValueError, match="two solver/NFE settings"):
        _normalize_candidate_catalog(catalog)


def test_quality_candidate_catalog_requires_multiple_fixed_schedules() -> None:
    catalog = json.loads(json.dumps(_quality_candidate_catalog()))
    second_fixed = next(
        candidate
        for candidate in catalog
        if candidate["candidate_key"] == "fixed-decoy"
    )
    time_grid = build_schedule_grid("uniform", 6)
    assert time_grid is not None
    second_fixed["execution"] = {
        "kind": "fixed_time_grid",
        "scheduler_key": "uniform",
        "density_source_key": "uniform",
        "time_grid": list(time_grid),
        "time_grid_sha256": _time_grid_sha256(time_grid),
    }

    with pytest.raises(ValueError, match="two registered schedule and density-source"):
        _normalize_candidate_catalog(catalog)


def test_quality_gate_requires_one_flow_map_evaluation_and_common_sample_panel() -> None:
    rows = _quality_rows()
    flow_row = next(row for row in rows if row["method"] == "flow_map")
    flow_row["model_evaluations"] = 2
    with pytest.raises(ValueError, match="model_evaluations must equal 1"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=38),
        )

    rows = _quality_rows()
    rows[0]["sample_panel_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="common sample panel"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=38),
        )


def test_quality_gate_metric_override_must_match_registered_claim_metrics() -> None:
    registered = metric_specs_for_scenario("cryptos")
    payload = [
        {
            "name": spec.name,
            "direction": spec.direction,
            "weight": spec.weight,
            "applicable_key": spec.applicable_key,
        }
        for spec in registered
    ]
    assert _resolve_metric_specs("cryptos", json.dumps(payload)) == registered

    payload[-1]["applicable_key"] = ""
    with pytest.raises(ValueError, match="exactly match"):
        _resolve_metric_specs("cryptos", json.dumps(payload))
    with pytest.raises(ValueError, match="metric objects"):
        _resolve_metric_specs("cryptos", json.dumps([*payload[:-1], "ignored"]))
    with pytest.raises(KeyError):
        _resolve_metric_specs(
            "unregistered_scenario",
            json.dumps([{"name": "easy_metric", "direction": "lower"}]),
        )


def test_quality_gate_public_api_rejects_unregistered_claim_metrics() -> None:
    from genode.distillation.evaluation import MetricSpec

    with pytest.raises(ValueError, match="exactly match"):
        evaluate_quality_gate(
            _quality_rows(),
            metric_specs=(MetricSpec("easy_metric", "lower"),),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=39),
        )


def test_quality_gate_requires_exact_prespecified_candidate_coverage() -> None:
    rows = [
        row
        for row in _quality_rows()
        if not (
            row["split_phase"] == "validation_tuning"
            and row["candidate_key"] == "fixed-decoy"
        )
    ]

    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
        )


def test_quality_gate_requires_selected_candidates_to_cover_bound_locked_panel() -> None:
    quality_contexts = json.loads(json.dumps(_quality_context_binding()))
    extra_context = {
        "split_phase": "locked_test",
        "context_id": "locked-20",
        "context_fingerprint": _test_context_fingerprint("locked-20"),
    }
    quality_contexts["contexts"].append(extra_context)
    quality_contexts["contexts"].sort(
        key=lambda row: (str(row["split_phase"]), str(row["context_id"]))
    )
    quality_contexts["context_count"] = len(quality_contexts["contexts"])
    quality_contexts["set_sha256"] = hashlib.sha256(
        json.dumps(
            quality_contexts["contexts"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    quality_contexts = validate_quality_context_binding(quality_contexts)

    quality_sample_panel = json.loads(json.dumps(_quality_sample_panel_binding()))
    extra_panel = {
        **extra_context,
        "sample_panel_sha256": _test_context_fingerprint(
            "sample-panel:locked-20"
        ),
        "replicate_count": 2,
    }
    quality_sample_panel["panels"].append(extra_panel)
    quality_sample_panel["panels"].sort(
        key=lambda row: (str(row["split_phase"]), str(row["context_id"]))
    )
    quality_sample_panel["context_count"] = len(quality_sample_panel["panels"])
    quality_sample_panel["sample_count"] = 2 * len(
        quality_sample_panel["panels"]
    )
    quality_sample_panel["set_sha256"] = hashlib.sha256(
        json.dumps(
            quality_sample_panel["panels"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    quality_sample_panel = validate_quality_sample_panel_binding(
        quality_sample_panel,
        quality_context_binding=quality_contexts,
    )

    rows = _quality_rows()
    for method, candidate_key in (
        ("flow_map", "flow-decoy"),
        ("gipo", "gipo-decoy"),
        ("fixed", "fixed-decoy"),
    ):
        template = next(
            row
            for row in rows
            if row["split_phase"] == "validation_tuning"
            and row["method"] == method
            and row["candidate_key"] == candidate_key
        )
        rows.append(
            {
                **template,
                **extra_context,
                "sample_panel_sha256": extra_panel["sample_panel_sha256"],
            }
        )

    with pytest.raises(ValueError, match="entire bound locked-test physical context panel"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_context_binding=quality_contexts,
            quality_sample_panel_binding=quality_sample_panel,
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
        )


def test_quality_gate_rejects_post_hoc_candidate_catalog_substitution() -> None:
    rows = _quality_rows()
    catalog = json.loads(json.dumps(_quality_candidate_catalog()))
    catalog[-1]["candidate_key"] = "fixed-replacement"

    with pytest.raises(ValueError, match="bound pipeline protocol"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=catalog,
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
        )


def test_quality_protocol_binds_its_payload_rows_and_catalog(tmp_path: Path) -> None:
    catalog = _quality_candidate_catalog()
    body = {
        "scenario_key": "cryptos",
        "flow_map": {
            "quality_candidate_catalog_sha256": candidate_catalog_sha256(catalog),
            "quality_rows_sha256": "9" * 64,
            "quality_contexts_sha256": "1" * 64,
            "quality_sample_panel_sha256": "2" * 64,
            "quality_measurement_protocol_sha256": "3" * 64,
        },
    }
    protocol_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps({**body, "protocol_hash": protocol_hash}),
        encoding="utf-8",
    )

    protocol, loaded_hash = read_quality_protocol(path, scenario_key="cryptos")
    binding = quality_protocol_binding(
        protocol,
        candidate_catalog=catalog,
        rows_sha256="9" * 64,
        quality_contexts_sha256="1" * 64,
        quality_sample_panel_sha256="2" * 64,
        measurement_protocol_sha256="3" * 64,
    )

    assert loaded_hash == protocol_hash
    assert binding["protocol_hash"] == protocol_hash
    with pytest.raises(ValueError, match="rows"):
        quality_protocol_binding(
            protocol,
            candidate_catalog=catalog,
            rows_sha256="8" * 64,
            quality_contexts_sha256="1" * 64,
            quality_sample_panel_sha256="2" * 64,
            measurement_protocol_sha256="3" * 64,
        )


def test_public_quality_gate_rejects_fabricated_pipeline_protocol_hash() -> None:
    config = QualityGateConfig(bootstrap_samples=1_000, seed=41)
    measurement_protocol = _quality_measurement_protocol(config)
    measurement_digest = measurement_protocol_sha256(measurement_protocol)
    protocol = _quality_protocol()
    protocol["flow_map"][
        "quality_measurement_protocol_sha256"
    ] = measurement_digest
    protocol["protocol_hash"] = "0" * 64

    with pytest.raises(ValueError, match="hash does not match"):
        evaluate_quality_gate(
            _quality_rows(),
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=protocol,
            measurement_protocol=measurement_protocol,
            measurement_protocol_sha256=measurement_digest,
            config=config,
        )


def test_quality_gate_requires_shared_metric_applicability_by_context() -> None:
    rows = _quality_rows()
    changed = next(
        row
        for row in rows
        if row["split_phase"] == "validation_tuning"
        and row["method"] == "flow_map"
        and row["candidate_key"] == "flow-selected"
        and row["context_id"] == "validation-0"
    )
    changed["temporal_tstr_f1_applicable"] = False

    with pytest.raises(ValueError, match="shared context property"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
        )


def test_read_quality_rows_parses_plain_csv_target_nfe(tmp_path: Path) -> None:
    path = tmp_path / "quality.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_nfe",
                "replicate_count",
                "model_evaluations",
                "method",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "target_nfe": "4",
                "replicate_count": "2",
                "model_evaluations": "1",
                "method": "flow_map",
            }
        )

    assert read_quality_rows(path)[0]["target_nfe"] == 4

    path.write_text(
        "target_nfe,replicate_count,model_evaluations,method\n"
        "4.0,2,1,flow_map\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="plain positive decimal"):
        read_quality_rows(path)


def test_quality_gate_rejects_finite_values_whose_difference_overflows() -> None:
    rows = _quality_rows()
    for row in rows:
        if row["split_phase"] != "locked_test":
            continue
        row["temporal_uw1"] = -1e308 if row["method"] == "flow_map" else 1e308

    with pytest.raises(ValueError, match="overflow"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
        )


def test_quality_gate_uses_overflow_safe_validation_means() -> None:
    rows = _quality_rows()
    for row in rows:
        if row["split_phase"] == "validation_tuning":
            row["temporal_uw1"] = 1e308

    report = evaluate_quality_gate(
        rows,
        metric_specs=metric_specs_for_scenario("cryptos"),
        candidate_catalog=_quality_candidate_catalog(),
        artifact_binding=_quality_binding(),
        demonstration_context_binding=context_binding(
            (_test_context_fingerprint("training-0"),)
        ),
        quality_protocol=_quality_protocol(),
        config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
    )

    assert report["status"] == "passed"
    assert all(
        np.isfinite(selection["primary_metric_means"]["temporal_uw1"])
        for selection in report["selection"].values()
    )


def test_quality_gate_rejects_renamed_demonstration_context() -> None:
    rows = _quality_rows()
    training_fingerprint = _test_context_fingerprint("training-0")
    for row in rows:
        if row["context_id"] == "validation-0":
            row["context_fingerprint"] = training_fingerprint
    quality_contexts = json.loads(json.dumps(_quality_context_binding()))
    for row in quality_contexts["contexts"]:
        if row["context_id"] == "validation-0":
            row["context_fingerprint"] = training_fingerprint
    quality_contexts["set_sha256"] = hashlib.sha256(
        json.dumps(
            quality_contexts["contexts"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    quality_contexts = validate_quality_context_binding(quality_contexts)
    quality_sample_panel = json.loads(json.dumps(_quality_sample_panel_binding()))
    for row in quality_sample_panel["panels"]:
        if row["context_id"] == "validation-0":
            row["context_fingerprint"] = training_fingerprint
    quality_sample_panel["set_sha256"] = hashlib.sha256(
        json.dumps(
            quality_sample_panel["panels"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    quality_sample_panel = validate_quality_sample_panel_binding(
        quality_sample_panel,
        quality_context_binding=quality_contexts,
    )

    with pytest.raises(ValueError, match="demonstration context"):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding((training_fingerprint,)),
            quality_context_binding=quality_contexts,
            quality_sample_panel_binding=quality_sample_panel,
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=40),
        )


def test_quality_gate_freezes_validation_choice_and_is_conservative_at_equality() -> None:
    rows = _quality_rows()
    for row in rows:
        if row["split_phase"] == "locked_test" and row["method"] == "flow_map":
            row["temporal_uw1"] = 1.0
            row["temporal_cw1"] = 1.0
            row["temporal_tstr_f1"] = 9.0
        if row["split_phase"] == "locked_test" and row["candidate_key"] == "flow-decoy":
            row["temporal_uw1"] = -100.0
            row["temporal_cw1"] = -100.0
            row["temporal_tstr_f1"] = 100.0

    report = evaluate_quality_gate(
        rows,
        metric_specs=metric_specs_for_scenario("cryptos"),
        candidate_catalog=_quality_candidate_catalog(),
        artifact_binding=_quality_binding(),
        demonstration_context_binding=context_binding(
            (_test_context_fingerprint("training-0"),)
        ),
        quality_protocol=_quality_protocol(),
        config=QualityGateConfig(bootstrap_samples=1_000, seed=41),
    )

    assert report["status"] == "failed"
    assert report["performance_claim"] is False
    assert report["selection"]["flow_map"]["candidate_key"] == "flow-selected"
    assert report["selection"]["flow_map"]["selection_split"] == "validation_tuning"
    assert report["selection"]["flow_map"]["locked_test_used_for_selection"] is False
    gipo_comparisons = [
        comparison
        for comparison in report["comparisons"]
        if comparison["comparator_method"] == "gipo"
    ]
    assert all(comparison["mean_difference"] == pytest.approx(0.0) for comparison in gipo_comparisons)
    assert all(not comparison["passed"] for comparison in gipo_comparisons)


def test_quality_gate_requires_paired_locked_test_contexts() -> None:
    rows = [
        row
        for row in _quality_rows()
        if not (
            row["split_phase"] == "locked_test"
            and row["method"] == "gipo"
            and row["context_id"] != "locked-0"
        )
    ]

    with pytest.raises(
        ValueError,
        match="entire bound locked-test physical context panel",
    ):
        evaluate_quality_gate(
            rows,
            metric_specs=metric_specs_for_scenario("cryptos"),
            candidate_catalog=_quality_candidate_catalog(),
            artifact_binding=_quality_binding(),
            demonstration_context_binding=context_binding(
                (_test_context_fingerprint("training-0"),)
            ),
            quality_protocol=_quality_protocol(),
            config=QualityGateConfig(bootstrap_samples=1_000, seed=43),
        )


def test_holm_adjustment_is_monotone_in_ranked_p_values() -> None:
    adjusted = _holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert all(adjusted[index] >= raw for index, raw in enumerate((0.01, 0.04, 0.03)))


def test_quality_gate_configuration_and_not_evaluated_status_are_explicit() -> None:
    with pytest.raises(ValueError, match="at least 1,000"):
        QualityGateConfig(bootstrap_samples=999)
    with pytest.raises(ValueError, match="zero"):
        QualityGateConfig(margin=0.01)

    report = not_evaluated_report(reason="Code-only release; benchmarks were not run.")
    assert report["status"] == "not_evaluated"
    assert report["performance_claim"] is False
    assert report["locked_test_used_for_selection"] is False
