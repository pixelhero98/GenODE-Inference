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


def raw_fixture():
    from genode.gico.collection import functional_manifest
    from tests.test_unified_gico_rewards import reference_evidence

    rows, _ = reference_evidence(task="sana")
    return functional_manifest(rows)


def test_native_preparation_keeps_the_complete_collected_scope_and_split():
    raw, manifest = raw_fixture()
    assert prepare_gico_rows(raw, manifest=manifest, task="sana", nfes=(4,)) == raw
    with pytest.raises(ValueError, match="scope"):
        prepare_gico_rows(raw, manifest=manifest, task="sana", nfes=(4, 8))
    with pytest.raises(ValueError, match="measurements changed"):
        prepare_gico_rows(raw + [raw[0]], manifest=manifest, task="sana", nfes=(4,))


def test_changed_clock_or_locked_rows_invalidate_the_collection():
    raw, manifest = raw_fixture()
    raw[0]["split"] = "locked_test"
    with pytest.raises(ValueError, match="measurements changed"):
        prepare_gico_rows(raw, manifest=manifest, task="sana", nfes=(4,))


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
        assert fit_gico(config_path=str(config), policy_kind="stochastic", dry_run=True) == {"task": "sana"}
    assert run.call_args.args[0]["policy_kind"] == "stochastic"
    assert "refinement_weight" not in run.call_args.args[0]
    assert run.call_args.kwargs["dry_run"] is True


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
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
        "policy_kind": kind,
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
    assert row["clock_policy_kind"] == kind and row["clock_seed"] == 23
    assert row["clock_key"] == kind and calls[0][1].key == kind


def test_preparation_uses_canonical_native_table_and_preserves_manifest(tmp_path):
    from genode.gico.collection import functional_manifest
    from genode.gico.evidence import content_hash
    from genode.gico.policy import load_context_embedding_table, save_context_embedding_table
    from genode.latent_clock.gico import prepare_gico
    from tests.test_unified_gico_rewards import reference_evidence

    rows, contexts = reference_evidence(task="sana")
    for row in rows:
        row["context_embedding_sha256"] = content_hash(contexts[row["context_id"]])
    rows, manifest = functional_manifest(rows)
    source = tmp_path / "collected.json"
    source.write_text(json.dumps({"rows": rows, "collection_manifest": manifest}))
    table = tmp_path / "native.npz"
    save_context_embedding_table(table, contexts)
    destination = tmp_path / "prepared"
    prepare_gico(
        rows_paths=[str(source)],
        embeddings_paths=[str(table)],
        manifest_path=str(source),
        task="sana",
        nfes=(4,),
        output=str(destination),
    )
    config = json.loads((destination / "train_config.json").read_text())
    assert config["policy_kind"] == "deterministic"
    assert json.loads((destination / "collection.json").read_text()) == manifest
    assert set(load_context_embedding_table(destination / "contexts.npz")) == set(contexts)
