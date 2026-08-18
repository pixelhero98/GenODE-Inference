from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import pytest

from genode.gico import image_artifact_io
from genode.gico.image_artifact_io import (
    load_float64_npy,
    staged_image_gico_directory,
    validate_image_gico_directory,
    write_float64_npy,
)
from genode.gico.image_causal_artifacts import (
    load_image_gico_causal_artifact,
    save_image_gico_causal_artifact,
)
from genode.gico.image_causal_training import (
    ImageGICOCausalTrainingConfig,
    train_image_gico_causal_student,
)
from genode.gico.image_students import (
    load_image_gico_deterministic_artifact,
    save_image_gico_deterministic_artifact,
    train_image_gico_deterministic_student,
)
from genode.gico.image_supervision import load_image_gico_supervision, save_image_gico_supervision
from genode.provenance import file_sha256
from tests.test_image_gico_supervision import _unconditional_supervision

_MEMBERS = {"manifest.json", "payload.bin"}


def _write_minimal_stage(stage: Path, *, marker: str = "payload") -> None:
    (stage / "payload.bin").write_bytes(marker.encode("ascii"))
    (stage / "manifest.json").write_text('{"artifact":"test"}\n', encoding="utf-8", newline="\n")


def _race_publisher(
    output: str,
    marker: str,
    barrier: Any,
    results: Any,
) -> None:
    try:
        with staged_image_gico_directory(output, expected_members=_MEMBERS) as stage:
            _write_minimal_stage(stage, marker=marker)
            barrier.wait(timeout=30)
        results.put(("published", marker))
    except FileExistsError:
        results.put(("exists", marker))


def test_concurrent_empty_target_is_preserved_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact"
    real_reserve = image_artifact_io._reserve_target_directory

    def concurrent_mkdir(path: Path):
        path.mkdir()
        return real_reserve(path)

    monkeypatch.setattr(image_artifact_io, "_reserve_target_directory", concurrent_mkdir)
    with (
        pytest.raises(FileExistsError),
        staged_image_gico_directory(
            target,
            expected_members=_MEMBERS,
        ) as stage,
    ):
        _write_minimal_stage(stage)

    assert target.is_dir()
    assert not list(target.iterdir())
    assert not list(tmp_path.glob(".artifact.stage-*"))


def test_two_process_publication_race_has_one_complete_winner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    target = tmp_path / "artifact"
    processes = [
        context.Process(target=_race_publisher, args=(str(target), marker, barrier, results))
        for marker in ("first", "second")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0

    observed = []
    for _ in processes:
        try:
            observed.append(results.get(timeout=10))
        except Empty as exc:  # pragma: no cover - diagnostic for a broken child process
            raise AssertionError("Publication worker did not report its outcome.") from exc
    assert sorted(status for status, _ in observed) == ["exists", "published"]
    winner = next(marker for status, marker in observed if status == "published")
    root = validate_image_gico_directory(target, expected_members=_MEMBERS)
    assert (root / "payload.bin").read_text(encoding="ascii") == winner
    assert not list(tmp_path.glob(".artifact.stage-*"))


def test_keyboard_interrupt_during_install_removes_only_owned_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact"
    real_link = image_artifact_io._link_regular_file_no_replace

    def interrupt_after_link(source: Path, destination: Path, *, expected: object) -> None:
        real_link(source, destination, expected=expected)
        raise KeyboardInterrupt

    monkeypatch.setattr(image_artifact_io, "_link_regular_file_no_replace", interrupt_after_link)
    with (
        pytest.raises(KeyboardInterrupt),
        staged_image_gico_directory(
            target,
            expected_members=_MEMBERS,
        ) as stage,
    ):
        _write_minimal_stage(stage)

    assert not os.path.lexists(target)
    assert not list(tmp_path.glob(".artifact.stage-*"))


def test_manifest_is_installed_only_after_incomplete_directory_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact"
    real_link = image_artifact_io._link_regular_file_no_replace
    observed_incomplete = False

    def inspect_before_manifest(source: Path, destination: Path, *, expected: object):
        nonlocal observed_incomplete
        if destination.name == "manifest.json":
            with pytest.raises(ValueError, match="incomplete or unexpected"):
                validate_image_gico_directory(target, expected_members=_MEMBERS)
            observed_incomplete = True
        return real_link(source, destination, expected=expected)

    monkeypatch.setattr(image_artifact_io, "_link_regular_file_no_replace", inspect_before_manifest)
    with staged_image_gico_directory(target, expected_members=_MEMBERS) as stage:
        _write_minimal_stage(stage)

    assert observed_incomplete
    validate_image_gico_directory(target, expected_members=_MEMBERS)


def test_dangling_link_leaf_is_rejected_without_creating_referent(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    missing = tmp_path / "missing"
    try:
        target.symlink_to(missing, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with (
        pytest.raises(ValueError, match="symlink, junction, or reparse"),
        staged_image_gico_directory(
            target,
            expected_members=_MEMBERS,
        ),
    ):
        pass
    assert not missing.exists()


def test_all_strict_image_gico_loaders_reject_linked_roots(tmp_path: Path) -> None:
    probe_target = tmp_path / "probe-target"
    probe_target.mkdir()
    probe = tmp_path / "probe-link"
    try:
        probe.symlink_to(probe_target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    probe.unlink()

    supervision = _unconditional_supervision()
    supervision_root = tmp_path / "supervision"
    save_image_gico_supervision(supervision, supervision_root)
    deterministic_root = tmp_path / "deterministic"
    save_image_gico_deterministic_artifact(
        train_image_gico_deterministic_student(supervision),
        supervision,
        deterministic_root,
    )
    causal_root = tmp_path / "causal"
    save_image_gico_causal_artifact(
        train_image_gico_causal_student(
            supervision,
            device="cpu",
            config=ImageGICOCausalTrainingConfig(updates=1, batch_size=2, seed=29),
        ),
        supervision,
        causal_root,
    )

    cases = (
        ("supervision-link", supervision_root, load_image_gico_supervision),
        ("deterministic-link", deterministic_root, load_image_gico_deterministic_artifact),
        ("causal-link", causal_root, load_image_gico_causal_artifact),
    )
    for name, root, loader in cases:
        link = tmp_path / name
        link.symlink_to(root, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink, junction, or reparse"):
            loader(link)
        link.unlink()


@pytest.mark.parametrize(
    ("relative_output", "unexpected_parent"),
    [
        (Path("created-by-bug") / ".." / "artifact", "created-by-bug"),
        (Path("created-by-bug") / " artifact ", "created-by-bug"),
    ],
)
def test_invalid_lexical_leaf_does_not_create_parent(
    tmp_path: Path,
    relative_output: Path,
    unexpected_parent: str,
) -> None:
    with (
        pytest.raises(ValueError),
        staged_image_gico_directory(
            tmp_path / relative_output,
            expected_members=_MEMBERS,
        ),
    ):
        pass

    assert not (tmp_path / unexpected_parent).exists()


def test_float64_npy_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptor = write_float64_npy(tmp_path / "array.npy", source)
    loaded = load_float64_npy(tmp_path, descriptor, field="density")

    assert descriptor["dtype"] == "float64-le"
    assert loaded.dtype == np.float64
    assert np.array_equal(loaded, source.astype(np.float64))
    assert not loaded.flags.writeable

    path = tmp_path / "array.npy"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="hash changed"):
        load_float64_npy(tmp_path, descriptor, field="density")


def test_float64_npy_post_write_validation_failure_removes_owned_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "array.npy"

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise ValueError("injected post-write validation failure")

    monkeypatch.setattr(image_artifact_io, "_sha256_regular_file", fail_validation)
    with pytest.raises(ValueError, match="injected post-write"):
        write_float64_npy(target, np.ones((2, 2), dtype=np.float64))

    assert not os.path.lexists(target)


def test_float64_npy_rejects_link_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real.npy"
    descriptor = write_float64_npy(real, np.ones((2, 2), dtype=np.float64))
    link = tmp_path / "linked.npy"
    try:
        link.symlink_to(real)
    except (NotImplementedError, OSError):
        monkeypatch.setattr(
            image_artifact_io,
            "is_link_or_reparse_point",
            lambda path: Path(path) == link,
        )
        link.write_bytes(real.read_bytes())
    linked_descriptor = {**descriptor, "file": link.name}

    with pytest.raises(ValueError, match="regular file"):
        load_float64_npy(tmp_path, linked_descriptor, field="density")


def test_float64_npy_rejects_nonfinite_write_and_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonempty and finite"):
        write_float64_npy(tmp_path / "write.npy", np.asarray([np.nan], dtype=np.float64))

    path = tmp_path / "load.npy"
    np.save(path, np.asarray([np.inf], dtype="<f8"), allow_pickle=False)
    descriptor = {
        "file": path.name,
        "sha256": file_sha256(path),
        "shape": [1],
        "dtype": "float64-le",
    }
    with pytest.raises(ValueError, match="metadata or values changed"):
        load_float64_npy(tmp_path, descriptor, field="density")


def test_float64_npy_rejects_empty_load(tmp_path: Path) -> None:
    path = tmp_path / "empty.npy"
    np.save(path, np.empty((0, 2), dtype="<f8"), allow_pickle=False)
    descriptor = {
        "file": path.name,
        "sha256": file_sha256(path),
        "shape": [0, 2],
        "dtype": "float64-le",
    }

    with pytest.raises(ValueError, match="metadata or values changed"):
        load_float64_npy(tmp_path, descriptor, field="density")
