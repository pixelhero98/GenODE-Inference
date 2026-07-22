from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from genode.distillation import demonstrations
from genode.distillation.artifacts import (
    context_fingerprint,
    in_memory_context_source_sha256,
    load_demonstration_manifest,
    write_demonstration_manifest,
    write_npz_shard,
)
from genode.distillation.demonstrations import _noise_seed
from genode.distillation.training import DemonstrationStore
from genode.gipo.density_representation import DENSITY_BIN_COUNT, uniform_reference_grid
from genode.gipo.models import build_setting_encoder_config
from genode.provenance import file_sha256


def _setting_encoder_payload() -> dict[str, object]:
    return build_setting_encoder_config(observed_target_nfes=(2, 4, 6, 8)).to_payload()


def _write_training_artifact(
    root: Path,
    *,
    context_ids: tuple[str, str],
    context_fingerprints: tuple[str, str],
) -> Path:
    context_record = write_npz_shard(
        root,
        "shards/contexts_00000.npz",
        {
            "context_index": np.asarray([0, 1], dtype=np.int64),
            "context_id": np.asarray(context_ids, dtype=np.str_),
            "context_fingerprint": np.asarray(context_fingerprints, dtype=np.str_),
            "ctx_tokens": np.zeros((2, 2, 8), dtype=np.float32),
            "ctx_summary": np.zeros((2, 8), dtype=np.float32),
            "summary": np.zeros((2, 8), dtype=np.float32),
        },
    )
    initial_state = np.asarray(
        [[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]],
        dtype=np.float32,
    )
    trajectory_record = write_npz_shard(
        root,
        "shards/trajectories_euler_nfe2_00000.npz",
        {
            "context_index": np.asarray([0, 1], dtype=np.int64),
            "noise_seed": np.asarray([101, 102], dtype=np.int64),
            "initial_state": initial_state,
            "time_grid": np.asarray(
                [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
                dtype=np.float32,
            ),
            "states": np.stack(
                (initial_state, initial_state + 0.25, initial_state + 0.5),
                axis=1,
            ),
            "density_mass": np.full(
                (2, DENSITY_BIN_COUNT),
                1.0 / DENSITY_BIN_COUNT,
                dtype=np.float32,
            ),
        },
    )
    trajectory_record.update(
        {
            "solver_key": "euler",
            "target_nfe": 2,
            "context_start": 0,
            "context_stop": 2,
        }
    )
    return write_demonstration_manifest(
        root,
        context_shards=[context_record],
        trajectory_shards=[trajectory_record],
        metadata={
            "artifact_version": 1,
            "split_phase": "train_tuning",
            "scenario_key": "synthetic",
            "benchmark_family": "synthetic",
            "locked_test_used": False,
            "context_count": 2,
            "rollouts_per_context": 1,
            "collection_seed": 0,
            "density_bin_count": DENSITY_BIN_COUNT,
            "density_reference_time_grid": list(uniform_reference_grid()),
            "context_embedding_kind": "ctx_summary",
            "sample_state_dim": 4,
            "settings": [{"solver_key": "euler", "target_nfe": 2}],
            "setting_encoder_config": _setting_encoder_payload(),
            "backbone_checkpoint_sha256": "a" * 64,
            "gipo_checkpoint_sha256": "b" * 64,
        },
    )


def test_context_fingerprint_uses_float32_and_normalized_zero_without_mutation() -> None:
    history32 = np.asarray([[0.1, -0.0], [1.25, -2.5]], dtype=np.float32)
    history64 = np.asarray([[0.1, 0.0], [1.25, -2.5]], dtype=np.float64)
    condition32 = np.asarray([-0.0, 0.3], dtype=np.float32)
    condition64 = np.asarray([0.0, 0.3], dtype=np.float64)
    original_bytes = (history32.tobytes(), history64.tobytes(), condition32.tobytes())

    assert context_fingerprint(history32, condition32) == context_fingerprint(
        history64, condition64
    )
    assert (history32.tobytes(), history64.tobytes(), condition32.tobytes()) == original_bytes
    assert bool(np.signbit(history32[0, 1]))
    assert bool(np.signbit(condition32[0]))

    with pytest.raises(ValueError, match="float32 model precision"):
        context_fingerprint(np.asarray([np.finfo(np.float64).max], dtype=np.float64))


def test_store_split_and_rollout_seed_depend_on_physical_identity(tmp_path: Path) -> None:
    fingerprints = (
        context_fingerprint(np.asarray([[0.1]], dtype=np.float32)),
        context_fingerprint(np.asarray([[0.2]], dtype=np.float32)),
    )
    first_manifest = _write_training_artifact(
        tmp_path / "first",
        context_ids=("alpha", "beta"),
        context_fingerprints=fingerprints,
    )
    renamed_manifest = _write_training_artifact(
        tmp_path / "renamed",
        context_ids=("beta", "alpha"),
        context_fingerprints=fingerprints,
    )

    first = DemonstrationStore(first_manifest, validation_fraction=0.5, split_seed=17)
    renamed = DemonstrationStore(renamed_manifest, validation_fraction=0.5, split_seed=17)

    assert first.split_by_context == renamed.split_by_context
    assert first.context_fingerprints == renamed.context_fingerprints
    first_metadata = load_demonstration_manifest(first_manifest)["metadata"]
    assert first_metadata["contexts_source_kind"] == "in_memory"
    assert first_metadata["contexts_source_sha256"] == in_memory_context_source_sha256(
        ("alpha", "beta"), fingerprints
    )

    expected_seed = int.from_bytes(
        hashlib.sha256(f"23\0{fingerprints[0]}\0{0}".encode("utf-8")).digest()[:8],
        "big",
    ) % (2**63 - 1)
    assert (
        _noise_seed(23, context_fingerprint=fingerprints[0], rollout_index=0)
        == expected_seed
    )


@pytest.mark.parametrize("field", ["contexts_source_kind", "contexts_source_sha256"])
def test_manifest_requires_context_source_provenance(
    tmp_path: Path,
    field: str,
) -> None:
    fingerprints = (
        context_fingerprint(np.asarray([[0.1]], dtype=np.float32)),
        context_fingerprint(np.asarray([[0.2]], dtype=np.float32)),
    )
    manifest_path = _write_training_artifact(
        tmp_path,
        context_ids=("alpha", "beta"),
        context_fingerprints=fingerprints,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["metadata"][field]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_demonstration_manifest(manifest_path)


def test_collection_cli_binds_the_context_npz_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backbone_path = tmp_path / "backbone.pt"
    gipo_path = tmp_path / "gipo.pt"
    contexts_path = tmp_path / "contexts.npz"
    backbone_path.write_bytes(b"backbone")
    gipo_path.write_bytes(b"gipo")
    np.savez(
        contexts_path,
        context_ids=np.asarray(["context-0"], dtype=np.str_),
        histories=np.zeros((1, 2, 4), dtype=np.float32),
    )
    captured: dict[str, object] = {}

    def fake_collect(*args: object, **kwargs: object) -> Path:
        captured.update(kwargs)
        return Path(str(kwargs["output_dir"])) / "flow_map_demonstrations.json"

    monkeypatch.setattr(demonstrations, "resolve_torch_device", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(demonstrations, "load_checkpoint_model", lambda *_args: (object(), {}))
    monkeypatch.setattr(demonstrations, "load_gipo_schedule_policy", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(demonstrations, "collect_flow_map_demonstrations", fake_collect)

    result = demonstrations.main(
        [
            "--backbone-checkpoint",
            str(backbone_path),
            "--gipo-checkpoint",
            str(gipo_path),
            "--contexts-npz",
            str(contexts_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--split-phase",
            "train_tuning",
            "--scenario-key",
            "synthetic",
            "--benchmark-family",
            "synthetic",
        ]
    )

    assert result == 0
    assert captured["contexts_source_kind"] == "npz"
    assert captured["contexts_source_sha256"] == file_sha256(contexts_path)
