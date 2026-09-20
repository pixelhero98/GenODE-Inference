"""Collection budgets, allocation and source-group isolation without generation."""

from collections import Counter
from copy import deepcopy

import pytest

from genode.gico.collection import CollectionConfig, plan_collection, validate_collection


def plan(task, inventory, **options):
    return plan_collection(
        task,
        "frozen-backbone",
        inventory,
        [("euler", 4), ("euler", 6)],
        source_revision="a" * 40,
        config=CollectionConfig(**options),
    )


def test_sparse_budget_balances_densities_and_preserves_context_membership():
    inventory = [{"context_id": str(i), "stratum": str(i % 8)} for i in range(512)]
    result = plan("traffic_hourly", inventory)
    validate_collection(result)
    assert result == plan("traffic_hourly", list(reversed(inventory)))
    assert [len(result["split_contexts"][s]) for s in ("train", "validation")] == [204, 52]
    selected = [r for r in result["requests"] if r["nfe"] == 4 and r["collection_repeat"] == 0]
    counts = Counter(r["schedule_key"] for r in selected if r["schedule_key"] != "uniform")
    uniform_assigned = 256 - sum(counts.values())
    counts["uniform"] = uniform_assigned
    assert Counter(counts.values()) == {10: 19, 11: 6}
    assert len(result["density_holdout"]["euler:4"]) == 5
    assert {r["collection_seed"] for r in result["requests"]} == {0, 1}
    assert all(r["sample_count"] == 5 for r in result["requests"])


def test_coco_uses_one_caption_per_source_image_and_never_splits_related_sources():
    inventory = [{"context_id": f"{i}:{j}", "group_id": str(i)} for i in range(300) for j in range(5)]
    result = plan("sana", inventory)
    assert len(result["inventory"]) == 256
    assert len({r["group_id"] for r in result["inventory"]}) == 256
    assert result == plan("sana", list(reversed(inventory)))


@pytest.mark.parametrize("task,total", [("cifar10", 10000), ("imagenet64", 125440)])
def test_image_budgets_include_repeats_and_reuse_uniform(task, total):
    inventory = (
        [{"context_id": "unconditional"}]
        if task == "cifar10"
        else [{"context_id": f"class:{i}", "class_id": i} for i in range(1000)]
    )
    result = plan(task, inventory)
    for nfe in (4, 6):
        rows = [r for r in result["requests"] if r["nfe"] == nfe]
        assert sum(r["sample_count"] for r in rows) == total
        assert min(r["sample_count"] for r in rows) >= 2
        if task == "cifar10":
            assert sum(r["sample_count"] for r in rows if r["split"] == "train") == 8000
        else:
            assert len(result["split_contexts"]["train"]) == 800
            assert len(result["split_contexts"]["validation"]) == 200
            assert Counter(r["sample_count"] for r in rows) == {32: len(rows)}


def test_tampered_plan_and_insufficient_contexts_fail():
    with pytest.raises(ValueError, match="at least 50"):
        plan("sana", [{"context_id": str(i)} for i in range(49)])
    result = plan("sana", [{"context_id": str(i)} for i in range(256)])
    damaged = deepcopy(result)
    damaged["requests"][0]["nfe"] = 8
    with pytest.raises(ValueError, match="checksum"):
        validate_collection(damaged)


def test_explicit_lpips_collection_keeps_the_same_image_allowance():
    result = plan("cifar10", [{"context_id": "unconditional"}], image_objective="lpips")
    validate_collection(result)
    assert len(result["requests"]) == 20000  # Two NFEs, not twice the per-NFE budget.
    assert all(r["sample_count"] == 1 and r["sample_seeds"] == [r["seed"]] for r in result["requests"])


@pytest.mark.parametrize("task,size", [("traffic_hourly", 5), ("molecule_3d_set1", 16)])
def test_sequence_repeats_have_disjoint_complete_member_seeds_and_exact_solve_counts(task, size):
    result = plan(task, [{"context_id": str(i)} for i in range(256)])
    validate_collection(result)
    rows = [r for r in result["requests"] if r["nfe"] == 4 and r["schedule_key"] == "uniform"]
    assert sum(r["sample_count"] for r in rows) == 256 * 2 * size
    by_context = {}
    for row in rows:
        assert row["sample_seeds"] == list(range(row["seed"], row["seed"] + size))
        previous = by_context.setdefault(row["context_id"], set())
        assert not previous.intersection(row["sample_seeds"])
        previous.update(row["sample_seeds"])
