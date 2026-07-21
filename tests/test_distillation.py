from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from genode.distillation.artifacts import (
    DEMONSTRATION_MANIFEST_NAME,
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
    parse_distillation_settings,
)
from genode.distillation.evaluation import (
    MetricSpec,
    QualityGateConfig,
    _holm_adjust,
    evaluate_quality_gate,
    not_evaluated_report,
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
from genode.distillation.training import DemonstrationStore, train_endpoint_flow_map
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
from genode.solver_protocol import SUPPORTED_SOLVER_KEYS


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


def _quality_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = {
        "flow_map": (("flow-selected", 4, 1.0, 9.0), ("flow-decoy", 6, 2.0, 8.0)),
        "gipo": (("gipo-selected", 4, 1.0, 9.0), ("gipo-decoy", 6, 2.0, 8.0)),
        "fixed": (("fixed-selected", 4, 1.0, 9.0), ("fixed-decoy", 6, 2.0, 8.0)),
    }
    for method, method_candidates in candidates.items():
        for candidate_key, target_nfe, error, score in method_candidates:
            for context_id in ("validation-0", "validation-1"):
                rows.append(
                    {
                        "split_phase": "validation_tuning",
                        "method": method,
                        "candidate_key": candidate_key,
                        "solver_key": "euler",
                        "target_nfe": target_nfe,
                        "context_id": context_id,
                        "selection_utility": 1.0 if candidate_key.endswith("selected") else 0.0,
                        **_quality_binding(),
                        "error": error,
                        "score": score,
                    }
                )
    locked_values = {
        "flow_map": ("flow-selected", 0.0, 10.0),
        "gipo": ("gipo-selected", 1.0, 9.0),
        "fixed": ("fixed-selected", 2.0, 8.0),
    }
    for method, (candidate_key, error, score) in locked_values.items():
        for context_id in (f"locked-{index}" for index in range(20)):
            rows.append(
                {
                    "split_phase": "locked_test",
                    "method": method,
                    "candidate_key": candidate_key,
                    "solver_key": "euler",
                    "target_nfe": 4,
                    "context_id": context_id,
                    "selection_utility": 0.0,
                    **_quality_binding(),
                    "error": error,
                    "score": score,
                }
            )
    return rows


def _quality_binding() -> dict[str, str]:
    return {
        "scenario_key": "synthetic",
        "flow_map_checkpoint_sha256": "c" * 64,
        "backbone_checkpoint_sha256": "d" * 64,
        "gipo_checkpoint_sha256": "e" * 64,
    }


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


def test_flow_map_checkpoint_rejects_locked_test_selection(tmp_path: Path) -> None:
    backbone_checkpoint = tmp_path / "backbone.pt"
    gipo_checkpoint = tmp_path / "gipo.pt"
    backbone_checkpoint.write_bytes(b"backbone")
    gipo_checkpoint.write_bytes(b"gipo")

    with pytest.raises(ValueError, match="Locked-test"):
        save_flow_map_checkpoint(
            tmp_path / "flow-map.pt",
            _flow_map(),
            backbone_checkpoint=backbone_checkpoint,
            gipo_checkpoint=gipo_checkpoint,
            setting_encoder_config=_setting_config(),
            training_summary={"locked_test_used_for_selection": True},
            demonstration_manifest_sha256="b" * 64,
        )


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


def test_gipo_schedule_loader_migrates_safe_version_six_student(tmp_path: Path) -> None:
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
    checkpoint = tmp_path / "gipo-v6.pt"
    torch.save(
        {
            "protocol": GIPO_PROTOCOL,
            "model_payload_version": 6,
            "student_policy_type": "continuous_density",
            "locked_test_used_for_selection": False,
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
            "teacher_training": {"series_conditioning": "none_context_only"},
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


def test_gipo_version_six_migration_rejects_ambiguous_or_invalid_fields() -> None:
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
        self.context_summaries: list[torch.Tensor] = []

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


def test_demonstration_store_rejects_duplicate_context_ids_before_splitting(
    tmp_path: Path,
) -> None:
    manifest = _write_training_manifest(tmp_path, duplicate_context_ids=True)

    with pytest.raises(ValueError, match="Duplicate context_id"):
        DemonstrationStore(manifest, validation_fraction=0.5, split_seed=23)


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
        metric_specs=(MetricSpec("error", "lower"), MetricSpec("score", "higher")),
        artifact_binding=_quality_binding(),
        config=QualityGateConfig(bootstrap_samples=1_000, seed=37),
    )

    assert report["status"] == "passed"
    assert report["performance_claim"] is True
    assert report["multiple_testing_correction"] == "holm"
    assert report["all_primary_metrics_required"] is True
    assert report["locked_test_used_for_selection"] is False
    assert len(report["comparisons"]) == 4
    assert all(comparison["paired_context_count"] == 20 for comparison in report["comparisons"])
    assert all(comparison["passed"] for comparison in report["comparisons"])
    assert all(
        comparison["holm_adjusted_p_value"] >= comparison["one_sided_p_value"]
        for comparison in report["comparisons"]
    )


def test_quality_gate_freezes_validation_choice_and_is_conservative_at_equality() -> None:
    rows = _quality_rows()
    for row in rows:
        if row["split_phase"] == "locked_test" and row["method"] == "flow_map":
            row["error"] = 1.0
            row["score"] = 9.0
        if row["split_phase"] == "locked_test" and row["candidate_key"] == "flow-decoy":
            row["error"] = -100.0
            row["score"] = 100.0

    report = evaluate_quality_gate(
        rows,
        metric_specs=(MetricSpec("error", "lower"), MetricSpec("score", "higher")),
        artifact_binding=_quality_binding(),
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

    with pytest.raises(ValueError, match="identical locked-test context coverage"):
        evaluate_quality_gate(
            rows,
            metric_specs=(MetricSpec("error", "lower"),),
            artifact_binding=_quality_binding(),
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
