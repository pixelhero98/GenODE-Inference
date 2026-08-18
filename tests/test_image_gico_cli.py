from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from genode.gico.image_cli import main


def _portable_inputs(root: Path) -> Path:
    support = np.full((3, 2, 64), 1.0 / 64.0, dtype=np.float64)
    weights = np.asarray(
        [
            [[0.25, 0.75]],
            [[0.50, 0.50]],
            [[0.75, 0.25]],
        ],
        dtype=np.float64,
    )
    np.save(root / "support.npy", support, allow_pickle=False)
    np.save(root / "weights.npy", weights, allow_pickle=False)
    manifest = root / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "unconditional_mixture",
                "target_nfes": [2, 4, 8],
                "schedule_keys": ["uniform", "uniform_alias"],
                "fixed_density_mass": "support.npy",
                "mixture_weights": "weights.npy",
                "source_identities": {"evidence": "sha256:" + "1" * 64},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_image_gico_cli_end_to_end_has_portable_lineage(tmp_path: Path) -> None:
    supervision = tmp_path / "supervision"
    assert (
        main(
            [
                "build-targets",
                "--manifest",
                str(_portable_inputs(tmp_path)),
                "--output",
                str(supervision),
            ]
        )
        == 0
    )

    deterministic = tmp_path / "deterministic"
    assert (
        main(
            [
                "train-deterministic",
                "--supervision",
                str(supervision),
                "--output",
                str(deterministic),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                "--supervision",
                str(supervision),
                "--deterministic",
                str(deterministic),
            ]
        )
        == 0
    )
    deterministic_schedule = tmp_path / "deterministic-schedule"
    assert (
        main(
            [
                "materialize",
                "--student",
                "deterministic_barycenter",
                "--artifact",
                str(deterministic),
                "--target-nfe",
                "4",
                "--context-indices",
                "0",
                "--output",
                str(deterministic_schedule),
            ]
        )
        == 0
    )

    config = tmp_path / "causal-config.json"
    config.write_text(json.dumps({"updates": 1, "batch_size": 1}), encoding="utf-8")
    stochastic = tmp_path / "stochastic"
    assert (
        main(
            [
                "train-stochastic",
                "--supervision",
                str(supervision),
                "--config",
                str(config),
                "--output",
                str(stochastic),
            ]
        )
        == 0
    )
    stochastic_schedule = tmp_path / "stochastic-schedule"
    assert (
        main(
            [
                "materialize",
                "--student",
                "stochastic_causal_ar",
                "--artifact",
                str(stochastic),
                "--target-nfe",
                "8",
                "--context-indices",
                "0",
                "--request-sha256",
                "2" * 64,
                "--sample-keys",
                "sample-0",
                "--output",
                str(stochastic_schedule),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "validate",
                "--supervision",
                str(supervision),
                "--deterministic",
                str(deterministic),
                "--stochastic",
                str(stochastic),
            ]
        )
        == 0
    )

    for manifest_path in tmp_path.rglob("manifest.json"):
        text = manifest_path.read_text(encoding="utf-8")
        assert str(tmp_path).replace("\\", "/") not in text.replace("\\", "/")
        assert ":\\" not in text


def _materialize_arguments(tmp_path: Path) -> list[str]:
    return [
        "materialize",
        "--student",
        "stochastic_causal_ar",
        "--artifact",
        str(tmp_path / "artifact"),
        "--target-nfe",
        "4",
        "--context-indices",
        "0",
        "--output",
        str(tmp_path / "schedule"),
    ]


def test_materialize_rejects_two_random_sources_at_argument_boundary(tmp_path: Path) -> None:
    uniforms = tmp_path / "uniforms.npy"
    with pytest.raises(SystemExit):
        main(
            [
                *_materialize_arguments(tmp_path),
                "--uniforms",
                str(uniforms),
                "--request-sha256",
                "1" * 64,
            ]
        )


@pytest.mark.parametrize("random_arguments", [[], ["--uniforms", "uniforms.npy"]])
def test_materialize_sample_keys_require_request_sha256(
    tmp_path: Path,
    random_arguments: list[str],
) -> None:
    with pytest.raises(ValueError, match="--sample-keys requires --request-sha256"):
        main(
            [
                *_materialize_arguments(tmp_path),
                *random_arguments,
                "--sample-keys",
                "sample-0",
            ]
        )
