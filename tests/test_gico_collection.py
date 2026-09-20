"""Collection budgets, allocation and source-group isolation without generation."""

from collections import Counter
from copy import deepcopy

import pytest

from genode.gico.collection import CollectionConfig, digest, plan_collection, validate_collection


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


def rebind_plan(manifest):
    """Recompute identities so semantic checks, rather than checksums, reject changes."""
    fitting = set(manifest["split_contexts"]["train"])
    for request in manifest["requests"]:
        request["split"] = "train" if request["context_id"] in fitting else "validation"
        request["request_id"] = digest({k: v for k, v in request.items() if k != "request_id"})
    manifest["plan_sha256"] = digest({k: v for k, v in manifest.items() if k != "plan_sha256"})


@pytest.mark.parametrize("task", ["traffic_hourly", "sana", "imagenet64"])
@pytest.mark.parametrize("mutation", ["size", "density", "duplicate"])
def test_rehashed_context_splits_must_match_configured_allocation(task, mutation):
    count = 1000 if task == "imagenet64" else 256
    inventory = [{"context_id": str(i), **({"class_id": i} if task == "imagenet64" else {})} for i in range(count)]
    result = plan(task, inventory)
    validate_collection(result)
    splits = result["split_contexts"]
    if mutation == "size":
        splits["train"] += splits["validation"][:-1]
        splits["validation"] = splits["validation"][-1:]
        error = "split size"
    elif mutation == "density":
        fitting = splits["train"][0]
        held = next(c for c in splits["validation"] if result["assignments"][c] != result["assignments"][fitting])
        splits["train"].remove(fitting)
        splits["train"].append(held)
        splits["validation"].remove(held)
        splits["validation"].append(fitting)
        error = "density split"
    else:
        splits["train"].append(splits["train"][0])
        error = "duplicate"
    rebind_plan(result)
    with pytest.raises(ValueError, match=error):
        validate_collection(result)


@pytest.mark.parametrize(
    "sizes,fraction,held", [([12, 18, 30], 0.8, 12), ([20, 20, 20], 0.8, 20), ([10, 20, 30], 0.75, 20)]
)
def test_grouped_split_keeps_nearest_feasible_size_and_larger_holdout_ties(sizes, fraction, held):
    inventory = [{"context_id": f"{g}:{i}", "group_id": str(g)} for g, size in enumerate(sizes) for i in range(size)]
    result = plan("traffic_hourly", inventory, fitting_fraction=fraction)
    validate_collection(result)
    assert len(result["split_contexts"]["validation"]) == held
    splits = result["split_contexts"]
    group = splits["train"][0].split(":")[0]
    moved = [c for c in splits["train"] if c.split(":")[0] == group]
    splits["train"] = [c for c in splits["train"] if c not in moved]
    splits["validation"] += moved
    rebind_plan(result)
    with pytest.raises(ValueError, match="nearest feasible"):
        validate_collection(result)


@pytest.mark.parametrize("task", ["traffic_hourly", "sana", "cifar10", "imagenet64"])
def test_custom_budget_repeat_and_split_settings_remain_valid(task):
    if task == "imagenet64":
        inventory = [{"context_id": str(i), "class_id": i} for i in range(1000)]
        options = {"imagenet_images_per_class": 32, "generation_seeds": (2, 3, 4, 5), "fitting_fraction": 0.75}
        expected_contexts, expected_solves = (750, 250), 62720
    elif task == "cifar10":
        inventory = [{"context_id": "unconditional"}]
        options = {"cifar_images": 12500, "generation_seeds": (2, 3, 4, 5, 6), "fitting_fraction": 0.75}
        expected_contexts, expected_solves = (1, 1), 12500
    else:
        inventory = [{"context_id": str(i)} for i in range(123)]
        options = {"context_budget": 111, "generation_seeds": (2, 3, 4), "fitting_fraction": 0.65}
        expected_contexts, expected_solves = (72, 39), None
    result = plan(task, inventory, **options)
    validate_collection(result)
    assert tuple(len(result["split_contexts"][s]) for s in ("train", "validation")) == expected_contexts
    if expected_solves is not None:
        assert sum(r["sample_count"] for r in result["requests"] if r["nfe"] == 4) == expected_solves
    if task == "imagenet64":
        counts = Counter(result["assignments"][c][0] for c in result["split_contexts"]["train"])
        assert set(counts.values()) == {30}
    if task == "cifar10":
        assert sum(r["sample_count"] for r in result["requests"] if r["nfe"] == 4 and r["split"] == "train") == 9375


@pytest.mark.parametrize("task,field", [("cifar10", "cifar_images"), ("imagenet64", "imagenet_images_per_class")])
def test_rehashed_image_budgets_cannot_silently_drop_remainders(task, field):
    inventory = (
        [{"context_id": "unconditional"}]
        if task == "cifar10"
        else [{"context_id": str(i), "class_id": i} for i in range(1000)]
    )
    result = plan(task, inventory)
    result["config"][field] += 1
    rebind_plan(result)
    with pytest.raises(ValueError, match="divide exactly"):
        validate_collection(result)


def test_cifar_panel_split_identities_cannot_be_swapped_after_rehashing():
    result = plan("cifar10", [{"context_id": "unconditional"}])
    splits = result["split_contexts"]
    splits["train"], splits["validation"] = splits["validation"], splits["train"]
    rebind_plan(result)
    with pytest.raises(ValueError, match="panel identities"):
        validate_collection(result)
