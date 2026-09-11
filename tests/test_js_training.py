import copy
import json

import numpy as np
import pytest
import torch

from genode.latent_clock.artifacts import canonical_sha256
from genode.latent_clock.js_experiment import _same_generator, training_panel, verify_anchor_scores
from genode.latent_clock.js_reinforce import DirichletSchedulePolicy, JSArchitecture
from genode.latent_clock.js_training import JSFit, fit_js, load_js, save_js


def test_budgeted_updates_and_artifact_replay(tmp_path):
    torch.manual_seed(101)
    model = DirichletSchedulePolicy(JSArchitecture(2, 6, 4, blocks=1, conv_width=8))
    before = copy.deepcopy(model.state_dict())
    contexts = [
        {"noise": torch.randn(1, 2, 4, 4), "text": torch.randn(1, 3, 6), "pooled": None, "padding_mask": None}
        for _ in range(3)
    ]
    calls = []

    def reward(index, schedule, update, member):
        calls.append((index, update, member))
        assert schedule.target_nfe == 4
        return (index + 1) * schedule.nodes[0]

    config = JSFit(15, 3, 41, batch_contexts=3)
    history = fit_js(model, context_count=3, inputs=contexts.__getitem__, evaluate=reward, config=config)
    assert len(history) == 2 and len(calls) == 12
    assert set(calls) == {(i, u, m) for i in range(3) for u in range(2) for m in range(2)}
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(np.isfinite(list(row.values())).all() for row in history)
    save_js(tmp_path / "artifact", model, {"training_contexts": ["a", "b", "c"]})
    state = torch.random.get_rng_state().clone()
    loaded, metadata = load_js(tmp_path / "artifact/policy.pt")
    assert torch.equal(state, torch.random.get_rng_state())
    assert not any(p.requires_grad for p in loaded.parameters())
    assert metadata["training_contexts"] == ["a", "b", "c"]
    torch.testing.assert_close(model(**contexts[0]), loaded(**contexts[0]), rtol=0, atol=0)
    manifest = tmp_path / "artifact/manifest.json"
    manifest.write_text(json.dumps({"protocol": "old", "policy_sha256": "wrong"}))
    with pytest.raises(ValueError, match="Incompatible or modified"):
        load_js(tmp_path / "artifact/policy.pt")
    with pytest.raises(ValueError, match="complete balanced"):
        JSFit(14, 3, 1, batch_contexts=3).updates(3)


def test_training_anchors_pairing_and_split_isolation():
    prompts = [
        {"prompt_id": f"p{i}", "image_id": i, "caption_id": i, "prompt": str(i), "split": "calibration"}
        for i in range(3)
    ]
    rows = [
        {
            "split": "train",
            "schedule_key": "uniform",
            "nfe": 4,
            "context_id": p["prompt_id"],
            "seed": seed,
            "task": "sana",
            "solver": "euler",
            "ensemble_size": 1,
            "time_grid": np.linspace(0, 1, 5).tolist(),
            "reference_id": canonical_sha256(
                {key: p[key] for key in ("prompt_id", "image_id", "caption_id", "prompt")}
            ),
        }
        for p in prompts
        for seed in (1, 2)
    ]
    anchors, _ = training_panel(rows, {"records": prompts}, nfe=4, noise_seeds=[1, 2])
    assert len(anchors) == 6
    with pytest.raises(ValueError, match="complete, unique"):
        training_panel(rows[:-1], {"records": prompts}, nfe=4, noise_seeds=[1, 2])
    altered = copy.deepcopy(prompts)
    altered[0]["split"] = "validation"
    with pytest.raises(ValueError, match="designated"):
        training_panel(rows, {"records": altered}, nfe=4, noise_seeds=[1, 2])
    altered[0]["split"] = "calibration"
    altered[0]["prompt"] = "Different text"
    with pytest.raises(ValueError, match="different prompt"):
        training_panel(rows, {"records": altered}, nfe=4, noise_seeds=[1, 2])


def test_anchor_scorer_and_audited_generator_provenance():
    scoring = {
        "scorer_versions": {"metric": "1"},
        "weight_fingerprints": ["weights"],
        "asset_manifest_sha256": "assets",
    }
    raw = {
        "prompt_id": "p",
        "noise_seed": 1,
        "clock_key": "uniform",
        "nfe": 4,
        "completed": True,
        "realized_nfe": 4,
        "measurement_protocol": "generator-protocol",
        "preference": 1.0,
        "alignment": 0.5,
    }
    anchors = {
        ("p", 1): {
            "nfe": 4,
            "metrics": {"preference": 1.0, "alignment": 0.5},
            "measurement_protocol": canonical_sha256(["generator-protocol", canonical_sha256(scoring)]),
        }
    }
    verify_anchor_scores(anchors, [raw], scoring)
    with pytest.raises(ValueError, match="scorer provenance"):
        verify_anchor_scores(anchors, [raw], {**scoring, "weight_fingerprints": ["changed"]})
    with pytest.raises(ValueError, match="metrics differ"):
        verify_anchor_scores(anchors, [{**raw, "preference": 2}], scoring)
    source = {"weights": "frozen", "runtime": {"cfg": 4.5, "implementation_sha256": "old"}}
    target = {"weights": "frozen", "runtime": {"cfg": 4.5, "implementation_sha256": "new"}}
    assert _same_generator(source, target, ["old", "new"])
    assert not _same_generator(source, target, ["old", "unreviewed"])
    assert not _same_generator(source, {**target, "weights": "changed"}, ["old", "new"])
