"""Portable evaluator integration with explicitly synthetic frozen runtimes."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from genode.gico.clocks import materialize
from genode.gico.evaluators import build_evaluator, measurement_identity
from genode.gico.image_evaluator import ImageEvaluator
from genode.gico.policy import save_context_embedding_table
from genode.gico.task_evaluators import SequenceEvaluator, tensor_digest
from tests.test_native_image_kid import native_rows
from tests.test_runtime_policy_clocks import ForecastFixture, StubSampler


class FixtureField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x, t, labels=None):
        return -x * self.scale * (1 + t.reshape(-1, 1, 1, 1))


class FixtureFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(255.0))

    def forward(self, images, return_features=True):
        return images.float().mean(dim=(2, 3)) / self.scale


def image_runtime():
    runtime = ImageEvaluator.__new__(ImageEvaluator)
    runtime.device = "cpu"
    runtime.bezier = None
    runtime.table = [[0.0]]
    runtime.model, runtime.detector = FixtureField(), FixtureFeatures()
    runtime.bind_modules([runtime.model, runtime.detector])
    return runtime


@pytest.mark.parametrize("kind", ["GICO-det-policy", "GICO-sto-policy"])
def test_builtin_factory_executes_paired_blocks_and_seed_specific_assets(tmp_path, monkeypatch, kind):
    from genode.benchmarks.image.noise import generate_seeded_image_noise
    from genode.latent_clock.artifacts import sha256_file

    runtime = image_runtime()
    anchors = [r for r in native_rows() if r["split"] == "validation" and r["schedule_key"] == "uniform"]
    runtime.objective, runtime.binding = anchors[0]["image_objective"], anchors[0]["backbone_binding"]
    contexts, cases = {}, {}
    for index, row in enumerate(anchors):
        noise = generate_seeded_image_noise("cifar10", row["sample_block"]["seeds"]).values
        row["sample_block"]["noise_sha256"] = [tensor_digest(x) for x in noise]
        features = tmp_path / f"reference-{index}.npy"
        np.save(features, np.array([[0.2, 0.3, 0.1], [0.3, 0.2, 0.4]]) + index / 10)
        case = {
            "reference_features": str(features),
            "reference_features_sha256": sha256_file(features),
            "reference_block": row["reference_block"],
        }
        contexts[row["context_id"]] = [0.0]
        cases[measurement_identity(row)] = case
        row["metrics"] = runtime.measure(row, [row] * 2, [0.0], case, tmp_path / f"measured-{index}")
        row["metrics"]["optional_diagnostic"] = 123.0
    assert len(cases) == 2 and len(contexts) == 1
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in anchors), encoding="utf-8")
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "runtime.json").write_text(json.dumps({"task": "cifar10"}), encoding="utf-8")
    save_context_embedding_table(tmp_path / "contexts.npz", contexts)
    config = {
        key: str(tmp_path / filename)
        for key, filename in {
            "rows": "rows.jsonl",
            "contexts": "contexts.npz",
            "runtime": "runtime.json",
            "cases": "cases.json",
            "output": "selection",
        }.items()
    }
    config.update(clock_seed=41, clock_replicates=2)
    monkeypatch.setattr("genode.gico.task_evaluators.load_task_evaluator", lambda config: runtime)
    requests = []

    def density(context, solver, nfe, *, seed, request_id):
        requests.append((seed, request_id))
        mass = np.arange(1, 65, dtype=float)
        return mass / mass.sum()

    evaluator = build_evaluator(config)
    candidate = SimpleNamespace(checkpoint_id="fixture-first", student_kind=kind, density=density)
    rows = evaluator(candidate)
    first = requests[:]
    requests.clear()
    candidate.checkpoint_id = "fixture-second"
    evaluator(candidate)
    assert requests == first and len(first) == len(set(first))
    assert len(rows) == (8 if kind == "GICO-sto-policy" else 4)
    assert list((tmp_path / "selection").glob("*/measurements.json"))
    assert all(np.isfinite(r["metrics"]["kid"]) for r in rows)
    runtime.verify_frozen()


def test_forecast_replays_recorded_nonfirst_physical_seed(tmp_path):
    model = StubSampler(1)
    runtime = SequenceEvaluator.__new__(SequenceEvaluator)
    runtime.device, runtime.molecule = torch.device("cpu"), False
    runtime.config = {"backbones": {"fixture": {"backbone_id": "fixture-backbone"}}}
    runtime.loaded = {"fixture": {"model": model, "cfg": model.cfg, "splits": {"val": ForecastFixture()}}}
    runtime.bind_modules([model])
    mass = np.full(64, 1 / 64)
    row = {
        "task": "traffic_hourly",
        "backbone": "fixture-backbone",
        "reference_id": "target",
        "seed": 31,
        "solver": "euler",
        "nfe": 2,
        "ensemble_size": 2,
        "schedule_key": "student",
    }
    clocks = [{"density_mass": mass.tolist(), "time_grid": list(materialize(mass, "euler", 2))}] * 2
    case = {
        "backbone": "fixture",
        "reference_id": "target",
        "example_idx": 0,
        "target_sha256": tensor_digest(torch.tensor([[4.0]])),
        "collection_batch_size": 1,
        "sample_seed_values": [1000031, 1000032],
    }
    runtime.measure(row, clocks, [4.0] * 4, case, tmp_path / "first")
    with torch.random.fork_rng(devices=[]):
        for seed, observed in zip(case["sample_seed_values"], model.draws, strict=True):
            torch.manual_seed(seed)
            torch.testing.assert_close(observed, torch.rand((1, 1, 1)), atol=0, rtol=0)
    case["collection_batch_size"] = 2
    with pytest.raises(ValueError, match="one context per batch"):
        runtime.measure(row, clocks, [4.0] * 4, case, tmp_path / "invalid")


def test_image_runtime_rejects_solver_mislabel_before_generation(tmp_path):
    runtime = image_runtime()
    row = deepcopy(native_rows()[0])
    runtime.objective, runtime.binding = row["image_objective"], row["backbone_binding"]
    row["solver"] = "heun"
    with pytest.raises(ValueError, match="binding changed"):
        runtime.measure(row, [row] * 2, [0.0], {}, tmp_path / "invalid")


def test_forecast_imputation_is_causal():
    from genode.data.otflow_forecast_data import _fill_missing_values

    left = _fill_missing_values(np.array([np.nan, 2, np.nan, 4]))
    right = _fill_missing_values(np.array([np.nan, 2, np.nan, 400]))
    np.testing.assert_array_equal(left[:3], [0, 2, 2])
    np.testing.assert_array_equal(left[:3], right[:3])
    with pytest.raises(ValueError, match="infinity"):
        _fill_missing_values(np.array([1, np.inf]))


def test_text_solver_identity_is_checked_before_prompt_execution(tmp_path):
    from genode.gico.task_evaluators import TextEvaluator
    from genode.latent_clock.gico import runtime_binding

    evaluator = TextEvaluator.__new__(TextEvaluator)
    evaluator.runtime = SimpleNamespace(
        adapter=SimpleNamespace(solver_key="ipndm", backbone_revision="fixture"),
        metadata={"backbone": "sd15"},
        fingerprints={"generator": "fixture"},
    )
    binding = runtime_binding(evaluator.runtime)
    with pytest.raises(ValueError, match="exact single-image frozen runtime"):
        evaluator.measure({"ensemble_size": 1, "backbone_binding": binding, "solver": "ipndm_v"}, [], [], {}, tmp_path)


def test_lpips_implementation_mismatch_fails_before_scorer_construction(tmp_path, monkeypatch):
    import sys

    from tests.test_image_gico_supervision import image_manifest

    manifest = image_manifest("cifar10")
    path = tmp_path / "backbone.json"
    path.write_text(json.dumps(manifest["backbone_manifest"]), encoding="utf-8")
    monkeypatch.setattr("genode.backbones.loading.load_verified_image_backbone", lambda *a, **kw: FixtureField())
    monkeypatch.setattr("genode.gico.image_conditional_context.native_contexts", lambda model: ([[0.0]], {}))
    monkeypatch.setattr("genode.gico.image_evaluator.verify_file", lambda *a: None)
    monkeypatch.setattr("genode.gico.image_evaluator.importlib.metadata.version", lambda name: "0.1.4")
    monkeypatch.setattr("genode.gico.image_evaluator.lpips_implementation_identity", lambda: "changed")
    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace())
    with pytest.raises(ValueError, match="LPIPS package version"):
        ImageEvaluator(
            {
                "task": "cifar10",
                "device": "cpu",
                "objective": manifest["rows"][0]["image_objective"],
                "backbone_manifest": str(path),
                "checkpoint": "fixture.pt",
                "source_root": "fixture",
                "lpips_checkpoint": "fixture-lpips.pt",
            }
        )
