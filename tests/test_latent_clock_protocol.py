import json

import numpy as np
import pytest

from genode.latent_clock.adapters.ipndm import compile_ipndm_times
from genode.latent_clock.clocks import (
    REFERENCE_CLOCK_KEYS,
    bo_bounds,
    clock_from_interval_logits,
    interval_logits,
    reference_clocks,
)
from genode.latent_clock.contracts import BudgetLedger, ExecutionTrace, FrozenContext
from genode.latent_clock.protocol import (
    NOISE_SEEDS,
    PILOT_CLOCKS,
    PILOT_NFES,
    RewardScales,
    build_prompt_splits,
    estimate_reward_scales,
)


def test_reference_support_includes_late_p3_pair_at_all_nfes():
    assert len(REFERENCE_CLOCK_KEYS) == 25
    assert {"late_p_3", "late_p_3_reversed"} <= set(REFERENCE_CLOCK_KEYS)
    for nfe in (4, 6, 8):
        clocks = reference_clocks(nfe)
        assert len(clocks) == 25
        assert all(clock.target_nfe == nfe and len(clock.nodes) == nfe + 1 for clock in clocks)
        p3 = next(clock for clock in clocks if clock.key == "late_p_3")
        reverse = next(clock for clock in clocks if clock.key == "late_p_3_reversed")
        assert np.allclose(reverse.nodes, 1 - np.asarray(p3.nodes)[::-1])


def test_logit_clock_round_trip_and_bounds_cover_reference_support():
    for nfe in (4, 6, 8):
        lower, upper = bo_bounds(nfe)
        for clock in reference_clocks(nfe):
            logits = interval_logits(clock)
            assert np.all(logits >= lower) and np.all(logits <= upper)
            reconstructed = clock_from_interval_logits("round_trip", logits)
            assert np.allclose(reconstructed.nodes, clock.nodes, atol=1e-14)


@pytest.mark.parametrize("nfe", [4, 6, 8])
def test_pg_extreme_actions_keep_complete_float32_clocks(nfe):
    from genode.latent_clock.clocks import sana_grid_is_representable

    rng = np.random.default_rng(319)
    actions = [np.full(nfe - 1, -1000.0), np.full(nfe - 1, 1000.0)]
    actions.extend(rng.normal(0, 100, size=(200, nfe - 1)))
    actions.append(np.asarray([np.finfo(float).max, -np.finfo(float).max] + [0.0] * (nfe - 3)))
    for logits in actions:
        clock = clock_from_interval_logits("pg_extreme", logits, pg_precision=True)
        assert len(clock.nodes) == nfe + 1
        assert clock.nodes[0] == 0 and clock.nodes[-1] == 1
        assert sana_grid_is_representable(np.asarray(clock.nodes))


def test_pg_precision_map_leaves_valid_clocks_exactly_unchanged():
    from genode.latent_clock.clocks import sana_grid_is_representable

    for nfe in (4, 6, 8):
        for reference in reference_clocks(nfe):
            logits = interval_logits(reference)
            old = clock_from_interval_logits("original", logits)
            if sana_grid_is_representable(np.asarray(old.nodes)):
                new = clock_from_interval_logits("new", logits, pg_precision=True)
                assert old.nodes == new.nodes


def test_pg_evaluation_uses_same_precision_map_as_training():
    import torch

    from genode.latent_clock.pg import ClockPolicy, sample_policy_clock

    policy = ClockPolicy(3, 8)
    with torch.no_grad():
        policy.network[-1].bias.fill_(-1000)
    context = np.zeros(3, dtype=np.float32)
    torch.manual_seed(123)
    with torch.no_grad():
        action = policy.distribution(torch.tensor(context)[None]).sample()[0].numpy()
    expected = clock_from_interval_logits("training", action, pg_precision=True)
    actual = sample_policy_clock(policy, context, seed=123)
    assert actual.nodes == expected.nodes
    assert actual.source_kind == "pg_precision_corrected"


def test_ipndm_compiler_uses_complete_descending_time_grid():
    clock = clock_from_interval_logits("candidate", [0.2, -0.3, 0.4])
    times = compile_ipndm_times(clock)
    assert len(times) == 5
    assert times[0] == 1 and times[-1] == pytest.approx(0.001)
    assert np.all(np.diff(times) < 0)


def test_prompt_split_is_image_disjoint_caption_unique_and_stable(tmp_path):
    annotations = [
        {"id": index * 2, "image_id": index, "caption": f"A unique caption {index}"} for index in range(900)
    ] + [{"id": 1, "image_id": 0, "caption": "unused second caption"}]
    source = tmp_path / "captions.json"
    source.write_text(json.dumps({"annotations": annotations}), encoding="utf-8")
    left = build_prompt_splits(source)
    right = build_prompt_splits(source)
    assert left == right
    assert len(left["records"]) == 800
    assert len({row["image_id"] for row in left["records"]}) == 800
    assert len({row["prompt"].casefold() for row in left["records"]}) == 800


def test_reward_scale_pilot_requires_exact_coverage_and_balanced_utility():
    rows = []
    for prompt_index in range(32):
        for nfe in PILOT_NFES:
            for clock_index, clock in enumerate(PILOT_CLOCKS):
                for seed_index, seed in enumerate(NOISE_SEEDS):
                    rows.append(
                        {
                            "prompt_id": f"p{prompt_index}",
                            "nfe": nfe,
                            "clock_key": clock,
                            "noise_seed": seed,
                            "preference": float(clock_index * (prompt_index + 1) + seed_index),
                            "alignment": float(clock_index * nfe + prompt_index * 0.1 + seed_index),
                        }
                    )
    scales = estimate_reward_scales(rows)
    assert scales.preference > 0 and scales.alignment > 0
    utility = scales.utility(2, 4, 1, 2)
    assert utility["utility"] == pytest.approx(0.5 / scales.preference + 1 / scales.alignment)
    with pytest.raises(ValueError, match="complete"):
        estimate_reward_scales(rows[:-1])
    with pytest.raises(ValueError, match="Degenerate"):
        RewardScales(0, 1)


def test_budget_ledger_distinguishes_logical_and_physical_work():
    ledger = BudgetLedger()
    trace = ExecutionTrace("gico", "uniform", "euler", 4, 4, 4, 8, 1.5)
    for method in ("global", "prompt"):
        ledger.record(trace, prompt_id=method, noise_seed=1, completed=True, physical_reuse_key="same")
    assert ledger.summary()["logical_trajectories"] == 2
    assert ledger.summary()["physical_trajectories"] == 1
    assert ledger.summary()["realized_nfe"] == 8
    assert ledger.summary()["physical_realized_nfe"] == 4
    assert ledger.summary()["physical_gpu_seconds"] == 1.5


def test_frozen_context_copies_and_seals_embedding():
    original = np.asarray([1.0, 2.0])
    context = FrozenContext("p", original, "revision")
    original[0] = 5
    assert context.embedding.tolist() == [1, 2]
    with pytest.raises(ValueError):
        context.embedding[0] = 4


def test_external_ipndm_features_have_order2_and_exact_nfe():
    from genode.gico.conditioning import Conditioning
    from genode.solver_protocol import normalize_solver_nfe_fields, solver_effective_order, solver_runtime_name

    rows = [{"split": "train", "context_id": "p", "solver": "ipndm", "nfe": nfe} for nfe in (4, 8)]
    conditioning = Conditioning.fit(rows, {"p": [1.0, 2.0]})
    fields = normalize_solver_nfe_fields("ipndm", 6, realized_nfe=6)
    assert fields.macro_steps == 6
    assert solver_effective_order("ipndm") == 2
    features = conditioning.transform([1.0, 2.0], "ipndm", 6)
    np.testing.assert_array_equal(features[:3], [0, 0, 1])
    np.testing.assert_allclose(conditioning.budget("ipndm", 6), np.log1p([6, 6]))
    np.testing.assert_allclose(features[-2:], conditioning.settings.transform_one(np.log1p([6, 6])))
    with pytest.raises(ValueError, match="external"):
        solver_runtime_name("ipndm")


def test_paired_statistics_keep_prompt_pairing_and_optimizer_variation():
    from genode.latent_clock.report import paired_bootstrap

    result = paired_bootstrap(np.tile([1.0, 2.0, 3.0], (512, 1)), replicates=100)
    assert result["mean_difference"] == 2.0
    assert result["ci95_low"] == 2.0 and result["ci95_high"] == 2.0
    assert result["optimizer_std"] == 1.0


def test_locked_collection_requires_frozen_method_registry(tmp_path):
    from genode.latent_clock.collection import prepare_collection

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"records": []}))
    with pytest.raises(ValueError, match="method-freeze"):
        prepare_collection(
            manifest_path=str(path), output=str(tmp_path / "plan.json"), phase="locked_test", nfes=[6], method="uniform"
        )


def test_weight_fingerprint_detects_mutation_without_large_python_copy():
    import torch

    from genode.latent_clock.rewards import _module_fingerprint

    module = torch.nn.Linear(4, 3)
    before = _module_fingerprint(module)
    with torch.no_grad():
        module.weight[0, 0].add_(1)
    assert _module_fingerprint(module) != before


def test_all_reference_clocks_survive_actual_sana_scheduler_precision():
    diffusers = pytest.importorskip("diffusers")
    from genode.latent_clock.adapters.sana import _prepare_scheduler

    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
    for nfe in (4, 6, 8):
        for clock in reference_clocks(nfe):
            _prepare_scheduler(scheduler, clock, "cpu")
            realized = scheduler.sigmas.numpy()
            assert len(realized) == nfe + 1
            assert np.all(np.diff(realized) < 0)
            assert np.all(np.diff(scheduler.timesteps.numpy()) < 0)


def test_scorer_resume_keeps_successful_observations_and_records_failure(tmp_path, monkeypatch):
    import torch

    from genode.latent_clock import rewards
    from genode.latent_clock.artifacts import read_jsonl, sha256_file, write_new_jsonl

    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")
    monkeypatch.setenv("LATENT_CLOCK_ASSETS", str(manifest))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    calls = []

    class Scorer:
        versions = {"mock": "1"}
        asset_manifest_sha256 = sha256_file(manifest)
        _fingerprints = ("a", "b")

        def __init__(self, **kwargs):
            pass

        def verify_frozen(self):
            pass

        def score(self, prompt, path):
            calls.append(prompt)
            if len(calls) == 2:
                raise RuntimeError("interruption")
            return 1.0, 0.5

    monkeypatch.setattr(rewards, "FrozenDualScorer", Scorer)
    rows = []
    for i in range(2):
        image = tmp_path / f"{i}.png"
        image.write_bytes(bytes([i]))
        rows.append(
            {"request_id": str(i), "prompt": str(i), "image_path": str(image), "image_sha256": sha256_file(image)}
        )
    source, output = tmp_path / "images.jsonl", tmp_path / "scores.jsonl"
    write_new_jsonl(source, rows)
    with pytest.raises(RuntimeError, match="interruption"):
        rewards.score_image_manifest(source, output)
    rewards.score_image_manifest(source, output)
    assert calls == ["0", "1", "1"]
    assert len(read_jsonl(output)) == 2
    assert len(read_jsonl(tmp_path / "scores.jsonl.records/failed-attempts.jsonl")) == 1


def test_geneval_uniform_plan_materializes_all_four_images(tmp_path):
    from genode.latent_clock.collection import prepare_geneval

    source, freeze, output = tmp_path / "geneval.jsonl", tmp_path / "freeze.json", tmp_path / "plan.json"
    source.write_text("\n".join(json.dumps({"prompt": f"p{i}", "tag": "counting"}) for i in range(553)))
    freeze.write_text(json.dumps({"method_identities": ["uniform"]}))
    prepare_geneval(
        source=str(source), output=str(output), method="uniform", nfe=6, checkpoint=None, freeze_path=str(freeze)
    )
    plan = json.loads(output.read_text())
    assert len(plan["requests"]) == 2212
    assert all(len(row["clock"]["nodes"]) == 7 for row in plan["requests"])
