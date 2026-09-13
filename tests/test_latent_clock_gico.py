from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from genode.gico.clocks import materialize, reference_densities
from genode.latent_clock.clocks import Clock, reference_clocks
from genode.latent_clock.contracts import ExecutionTrace, FrozenContext
from genode.latent_clock.gico import fit_gico, prepare_gico_rows, runtime_binding
from genode.latent_clock.protocol import NOISE_SEEDS, PILOT_CLOCKS


def raw_fixture():
    manifest = {
        "records": [
            {"prompt_id": phase, "split": phase, "image_id": i, "caption_id": i, "prompt": f"A distinct image {i}."}
            for i, phase in enumerate(("pilot", "calibration", "validation"))
        ]
    }
    rows = []
    binding = {"task": "sana", "backbone_revision": "weights", "solver": "euler"}
    for prompt in manifest["records"]:
        for nfe in (4, 6, 8):
            for clock in reference_clocks(nfe):
                if prompt["split"] == "pilot" and (nfe == 6 or clock.key not in PILOT_CLOCKS):
                    continue
                for seed in NOISE_SEEDS:
                    rows.append(
                        {
                            **prompt,
                            "nfe": nfe,
                            "clock_key": clock.key,
                            "noise_seed": seed,
                            "completed": True,
                            "realized_nfe": nfe,
                            "checkpoint_id": "weights",
                            "solver_key": "euler",
                            "density_mass": list(clock.density_mass),
                            "nodes": list(clock.nodes),
                            "backbone_binding": binding,
                            "measurement_protocol": "frozen-text-scorers-v1",
                            "context_embedding_sha256": "a" * 64,
                            "preference": float("nan") if nfe == 6 else float(nfe),
                            "alignment": 0.5,
                        }
                    )
    return rows, manifest


def test_native_transfer_excludes_nfe6_before_metric_handling_and_keeps_pilot_separate():
    raw, manifest = raw_fixture()
    rows, pilot = prepare_gico_rows(raw, manifest=manifest, task="sana", nfes=(4, 8), budget="25")
    assert len(rows) == 2 * 2 * 25 * 2 and len(pilot) == 2 * 3 * 2
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert {row["split"] for row in pilot} == {"calibration"}
    assert all(np.isfinite(row["metrics"]["preference"]) for row in rows)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        prepare_gico_rows(raw + [raw[0]], manifest=manifest, task="sana", nfes=(4, 8))


def test_historical_direct_grid_evidence_and_locked_rows_are_rejected():
    raw, manifest = raw_fixture()
    raw[0].pop("density_mass")
    with pytest.raises(ValueError, match="recollect"):
        prepare_gico_rows(raw, manifest=manifest, task="sana", nfes=(4, 8))
    raw[0]["split"] = "locked_test"
    with pytest.raises(ValueError, match="Locked-test"):
        prepare_gico_rows(raw, manifest=manifest, task="sana", nfes=(4, 8))


def test_every_latent_reference_executes_common_density_realization():
    for nfe in (4, 6, 8):
        for clock in reference_clocks(nfe):
            assert np.array_equal(clock.nodes, materialize(clock.density_mass, "euler", nfe))
            assert np.array_equal(clock.nodes, materialize(clock.density_mass, "ipndm", nfe))
    reference = reference_clocks(4)[0]
    with pytest.raises(ValueError, match="density realization"):
        Clock("bad", 4, (0, 0.1, 0.2, 0.8, 1), density_mass=reference.density_mass)


def test_native_fit_delegates_to_common_config(tmp_path):
    config = tmp_path / "train.json"
    config.write_text(json.dumps({"rows": "rows.jsonl", "contexts": "contexts.npz", "output": "policy"}))
    with patch("genode.latent_clock.gico.run_config", return_value={"task": "sana"}) as run:
        assert fit_gico(
            config_path=str(config), student_kind="GICO-sto-policy", teacher_score_weight=0.1, dry_run=True
        ) == {"task": "sana"}
    assert run.call_args.args[0]["student_kind"] == "GICO-sto-policy"
    assert run.call_args.args[0]["teacher_score_weight"] == 0.1
    assert run.call_args.kwargs["dry_run"] is True


@pytest.mark.parametrize("kind", ["GICO-det-policy", "GICO-sto-policy"])
def test_collection_samples_one_clock_per_image_and_reuses_complete_solver_grid(tmp_path, kind):
    from genode.latent_clock.collection import collect

    calls = []

    class Adapter:
        backbone_revision = "weights"
        solver_key = "euler"

        def encode_context(self, prompt_id, prompt):
            return FrozenContext(prompt_id, np.asarray([1.0, 2.0]), "weights")

        def sample(self, *, noise_seed, context, clock):
            calls.append((noise_seed, clock))
            return torch.zeros((1, 3, 512, 512)), ExecutionTrace("gico", clock.key, "euler", 4, 4, 4, 8, 0.0)

    runtime = SimpleNamespace(
        adapter=Adapter(),
        metadata={"backbone": "sana"},
        fingerprints={"model": "immutable"},
        verify_frozen=lambda: None,
    )
    sampled = []

    class Policy:
        artifact_sha256 = "a" * 64
        metadata = {"task": "sana", "solvers": ["euler"], "backbone_binding": runtime_binding(runtime)}

        def density(self, embedding, solver, nfe, *, seed, request_id):
            sampled.append((embedding.copy(), solver, nfe, seed, request_id))
            return np.asarray(reference_densities("euler", nfe)["late_p_3"])

    request = {
        "prompt_id": "p",
        "prompt": "A blue cube",
        "split": "validation",
        "request_id": "image-one",
        "method": "gico",
        "clock": None,
        "nfe": 4,
        "noise_seed": 17011,
    }
    plan = {
        "method": "gico",
        "checkpoint": "policy",
        "checkpoint_sha256": "a" * 64,
        "student_kind": kind,
        "clock_seed": 23,
        "requests": [request],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text("{}")
    with (
        patch("genode.latent_clock.collection.load_runtime", return_value=runtime),
        patch("genode.gico.policy.load_policy", return_value=Policy()),
    ):
        collect(runtime_config=str(runtime_path), plan_path=str(path), output=str(tmp_path / "images"))
    assert len(sampled) == len(calls) == 1
    assert sampled[0][3:] == (23, "image-one") and calls[0][0] == 17011
    row = json.loads((tmp_path / "images" / "image-one.json").read_text())
    assert row["density_mass"] == list(calls[0][1].density_mass)
    assert row["clock_student_kind"] == kind and row["clock_seed"] == 23
    assert row["clock_key"] == kind and calls[0][1].key == kind
