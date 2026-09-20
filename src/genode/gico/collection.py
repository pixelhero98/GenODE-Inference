"""Budgeted complete-solve collection, independent of policy optimization."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from genode.gico.clocks import REFERENCE_KEYS, density_identity, materialize, reference_densities


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CollectionConfig:
    context_budget: int = 256
    generation_seeds: tuple[int, ...] = (0, 1)
    seed: int = 0
    fitting_fraction: float = 0.8
    density_holdout_fraction: float = 0.2
    cifar_images: int = 10000
    imagenet_images_per_class: int = 64
    image_objective: str = "kid"
    forecast_ensemble_size: int = 5
    molecule_ensemble_size: int = 16

    def __post_init__(self):
        if self.image_objective not in ("kid", "lpips"):
            raise ValueError("Image collection objective must be kid or explicit lpips.")
        for size in (self.forecast_ensemble_size, self.molecule_ensemble_size):
            if type(size) is not int or size < 2:
                raise ValueError("Sequence collection requires complete ensembles with at least two members.")
        for name in ("context_budget", "cifar_images", "imagenet_images_per_class"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Collection budgets must be positive integers.")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Collection seed must be nonnegative.")
        if not self.generation_seeds or any(type(s) is not int or s < 0 for s in self.generation_seeds):
            raise ValueError("Generation seeds must be nonnegative integers.")
        if len(set(self.generation_seeds)) != len(self.generation_seeds):
            raise ValueError("Generation repeats require distinct seeds.")
        if not 0 < self.fitting_fraction < 1 or not 0 < self.density_holdout_fraction < 1:
            raise ValueError("Collection split fractions must be strictly between zero and one.")


def reference_support(settings):
    return {
        f"{solver}:{nfe}": {
            name: {
                "density_mass": list(mass),
                "time_grid": list(materialize(mass, solver, nfe)),
                "density_identity": density_identity(mass),
            }
            for name, mass in reference_densities(solver, nfe).items()
        }
        for solver, nfe in sorted(set(settings))
    }


def density_holdouts(support, fraction=0.2, seed=0):
    """Group equivalent realizations across settings; uniform is always the anchor."""
    parents = dict.fromkeys(REFERENCE_KEYS)
    for key in parents:
        parents[key] = key

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    for pool in support.values():
        seen = {}
        for key, value in pool.items():
            identity = value["density_identity"]
            if identity in seen:
                parents[root(key)] = root(seen[identity])
            seen[identity] = key
    groups = defaultdict(list)
    for key in parents:
        groups[root(key)].append(key)
    candidates = sorted(sorted(g) for g in groups.values() if "uniform" not in g)
    if len(candidates) < 2:
        raise ValueError("Density holdout requires at least two distinct nonuniform reference densities.")
    count = min(len(candidates) - 1, math.ceil(len(candidates) * fraction))
    chosen = np.random.default_rng(seed).choice(len(candidates), count, replace=False)
    names = sorted(key for index in chosen for key in candidates[index])
    return {setting: sorted({pool[key]["density_identity"] for key in names}) for setting, pool in support.items()}


def _stratified(records, count, rng):
    strata = defaultdict(list)
    for row in sorted(records, key=lambda r: r["context_id"]):
        strata[str(row.get("stratum", "all"))].append(row)
    pools = [list(rng.permutation(rows)) for _, rows in sorted(strata.items())]
    selected = []
    while len(selected) < count:
        for index in rng.permutation(len(pools)):
            if pools[index] and len(selected) < count:
                selected.append(pools[index].pop())
    return selected


def _fitting_counts(counts, fraction):
    sizes = [math.floor(count * fraction) for count in counts]
    remaining = math.floor(sum(counts) * fraction) - sum(sizes)
    order = sorted(range(len(counts)), key=lambda i: -(counts[i] * fraction - sizes[i]))
    for index in order[:remaining]:
        sizes[index] += 1
    return sizes


def _grouped_holdout(counts, fraction):
    # Exact subset sum where possible; nearest feasible size otherwise. Ties
    # prefer the larger holdout, keeping each source group intact.
    reachable = {0: ()}
    for name, count in counts.items():
        for size, chosen in list(reachable.items()):
            reachable.setdefault(size + count, (*chosen, name))
    total = sum(counts.values())
    target = total - math.floor(total * fraction)
    feasible = [size for size in reachable if 0 < size < total]
    if not feasible:
        raise ValueError("A grouped holdout requires at least two independent source groups.")
    size = min(feasible, key=lambda value: (abs(value - target), value < target, value))
    return reachable[size]


def _assign(records, fraction, rng):
    keys = list(rng.permutation(REFERENCE_KEYS))
    assigned = {r["context_id"]: keys[i % len(keys)] for i, r in enumerate(records)}
    groups = [[r for r in records if assigned[r["context_id"]] == key] for key in keys]
    sizes = _fitting_counts([len(group) for group in groups], fraction)
    splits = {}
    for group, count in zip(groups, sizes, strict=True):
        for index, row in enumerate(group):
            splits[row["context_id"]] = "train" if index < count else "validation"
    if len({row.get("group_id", row["context_id"]) for row in records}) < len(records):
        if not all(isinstance(row.get("group_id"), str) and row["group_id"] for row in records):
            raise ValueError("Grouped contexts require a source group_id for every entry.")
        grouped = defaultdict(list)
        for row in records:
            grouped[row["group_id"]].append(row["context_id"])
        names = list(rng.permutation(sorted(grouped)))
        heldout = set(_grouped_holdout({name: len(grouped[name]) for name in names}, fraction))
        splits = {row["context_id"]: "validation" if row["group_id"] in heldout else "train" for row in records}
    return assigned, splits


def plan_collection(task, backbone, inventory, settings, *, source_revision, config=None):
    """Plan all requests before executing any generator or observing any reward.

    Inventory entries require context_id and may carry stratum/class_id/group_id. Native
    embeddings and runtime-specific cases stay in separate input files.
    """
    config = config or CollectionConfig()
    from genode.gico.rewards import TASK_METRICS

    if task not in TASK_METRICS:
        raise ValueError("Collection requires a registered task.")
    if not source_revision or not backbone or not settings:
        raise ValueError("Collection requires source, frozen backbone and solver/NFE settings.")
    records = copy.deepcopy(inventory)
    ids = [r["context_id"] for r in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Inventory must contain distinct context identities.")
    rng = np.random.default_rng(config.seed)
    if task in ("sana", "sd15") and any("group_id" in r for r in records):
        grouped = defaultdict(list)
        for row in sorted(records, key=lambda r: r["context_id"]):
            if not isinstance(row.get("group_id"), str) or not row["group_id"]:
                raise ValueError("Caption inventories require the source image group_id on every entry.")
            grouped[row["group_id"]].append(row)
        records = [grouped[key][int(rng.integers(len(grouped[key])))] for key in sorted(grouped)]
    support = reference_support(settings)
    holdout = density_holdouts(support, config.density_holdout_fraction, config.seed)
    repeats = len(config.generation_seeds)
    if task == "cifar10":
        if len(records) != 1 or config.cifar_images % (25 * repeats):
            raise ValueError("CIFAR requires one unconditional context and an exactly divisible image budget.")
        records = [
            {**records[0], "context_id": f"{phase}:unconditional", "source_context_id": ids[0]}
            for phase in ("train", "validation")
        ]
        splits = {r["context_id"]: r["context_id"].split(":")[0] for r in records}
        assigned = {r["context_id"]: list(REFERENCE_KEYS) for r in records}
    else:
        if task == "imagenet64":
            if len(records) != 1000 or {r.get("class_id") for r in records} != set(range(1000)):
                raise ValueError("ImageNet collection requires all 1000 distinct class contexts.")
            if config.imagenet_images_per_class % repeats:
                raise ValueError("ImageNet image budget must divide exactly across collection repeats.")
            count = 1000
        else:
            count = min(config.context_budget, len(records))
            if count < 2 * len(REFERENCE_KEYS):
                raise ValueError(
                    "Sparse collection needs at least 50 contexts to populate both splits and all densities."
                )
        records = _stratified(records, count, rng)
        assigned, splits = _assign(records, config.fitting_fraction, rng)
        assigned = {key: [value] for key, value in assigned.items()}
    requests = _requests(task, backbone, records, support, assigned, splits, config)
    manifest = {
        "protocol": "complete-solve-collection",
        "task": task,
        "backbone": backbone,
        "source_revision": source_revision,
        "config": {**asdict(config), "generation_seeds": list(config.generation_seeds)},
        "inventory": records,
        "assignments": assigned,
        "reference_support": support,
        "density_holdout": holdout,
        "split_contexts": {s: sorted(k for k, v in splits.items() if v == s) for s in ("train", "validation")},
        "requests": requests,
        "state": "planned",
    }
    manifest["plan_sha256"] = digest(manifest)
    return manifest


def _requests(task, backbone, records, support, assigned, splits, config):
    repeats = len(config.generation_seeds)
    requests = []
    for setting, pool in support.items():
        solver, nfe = setting.rsplit(":", 1)
        for row in records:
            context = row["context_id"]
            phase = splits[context]
            for repeat, seed in enumerate(config.generation_seeds):
                samples = 1
                if task == "cifar10":
                    total = config.cifar_images // (25 * repeats)
                    train = math.floor(total * config.fitting_fraction)
                    samples = train if phase == "train" else total - train
                elif task == "imagenet64":
                    samples = config.imagenet_images_per_class // repeats
                elif task.startswith("molecule_"):
                    samples = config.molecule_ensemble_size
                elif task not in ("sana", "sd15"):
                    samples = config.forecast_ensemble_size
                if task in ("cifar10", "imagenet64") and samples < 2:
                    raise ValueError("Each complete KID block needs at least two samples.")
                if task in ("cifar10", "imagenet64"):
                    seeds = [int(digest([config.seed, seed, context, i])[:15], 16) for i in range(samples)]
                elif task in ("sana", "sd15"):
                    seeds = [seed]
                else:
                    # Native sequence runtimes use consecutive physical member seeds.
                    # Hash the repeat/context first so adjacent repeat labels never share members.
                    first = int(digest([config.seed, seed, context])[:15], 16)
                    seeds = list(range(first, first + samples))
                for key in dict.fromkeys(["uniform", *assigned[context]]):
                    request = {
                        "task": task,
                        "backbone": backbone,
                        "solver": solver,
                        "nfe": int(nfe),
                        "context_id": context,
                        "split": phase,
                        "schedule_key": key,
                        "seed": seeds[0],
                        "collection_seed": seed,
                        "collection_repeat": repeat,
                        "sample_seeds": seeds,
                        "sample_count": samples,
                        **pool[key],
                    }
                    if task == "imagenet64":
                        request["class_id"] = row["class_id"]
                    request["request_id"] = digest(request)
                    requests.append(request)
    if task in ("cifar10", "imagenet64") and config.image_objective == "lpips":
        expanded = []
        for request in requests:
            for sample_seed in request["sample_seeds"]:
                single = {k: v for k, v in request.items() if k != "request_id"}
                single.update(seed=sample_seed, sample_seeds=[sample_seed], sample_count=1)
                single["request_id"] = digest(single)
                expanded.append(single)
        requests = expanded
    return requests


def validate_collection(manifest, rows=None):
    if not isinstance(manifest, dict) or manifest.get("protocol") != "complete-solve-collection":
        raise ValueError("A canonical complete-solve collection manifest is required.")
    planned = {k: v for k, v in manifest.items() if k not in {"plan_sha256", "measurements_sha256", "complete_solves"}}
    planned["state"] = "planned"
    if digest(planned) != manifest.get("plan_sha256"):
        raise ValueError("Collection plan checksum mismatch.")
    if manifest.get("state") not in ("planned", "complete"):
        raise ValueError("Unknown collection state.")
    if manifest["state"] == "complete":
        checksum = manifest.get("measurements_sha256", "")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Completed collection requires a measurement checksum.")
        if manifest.get("complete_solves") != sum(r["sample_count"] for r in manifest["requests"]):
            raise ValueError("Collection solve accounting differs from its requests.")
    splits = manifest["split_contexts"]
    if (
        set(splits) != {"train", "validation"}
        or any(not values or len(values) != len(set(values)) for values in splits.values())
        or set(splits["train"]) & set(splits["validation"])
    ):
        raise ValueError("Collection contexts must have nonempty disjoint splits without duplicate entries.")
    support = manifest["reference_support"]
    settings = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in support]
    if support != reference_support(settings):
        raise ValueError("Collection reference support differs from the canonical densities.")
    config = CollectionConfig(**manifest["config"])
    if manifest["density_holdout"] != density_holdouts(support, config.density_holdout_fraction, config.seed):
        raise ValueError("Collection density holdout differs from the seeded identity split.")
    if manifest.get("purpose") != "functional":
        _validate_requests(manifest, config)
    group_splits = {}
    for record in manifest["inventory"]:
        if "group_id" in record:
            phase = "train" if record["context_id"] in splits["train"] else "validation"
            if group_splits.setdefault(record["group_id"], phase) != phase:
                raise ValueError("Related source contexts cannot cross the fitting/held-out split.")
    if rows is not None:
        if manifest.get("state") != "complete" or digest(rows) != manifest.get("measurements_sha256"):
            raise ValueError("Collection is incomplete or measurements changed.")
        requests = {r["request_id"]: r for r in manifest["requests"]}
        if len(requests) != len(manifest["requests"]) or len(rows) != len(requests):
            raise ValueError("Complete collection requires exactly one observation per request.")
        observed = set()
        for row in rows:
            identity = row.get("collection_request_id")
            if identity in observed or identity not in requests:
                raise ValueError("Duplicate or unplanned collection observation.")
            observed.add(identity)
            request = requests[identity]
            if manifest.get("purpose") != "functional" and row.get("ensemble_size") != request["sample_count"]:
                raise ValueError("Collected ensemble size differs from its planned physical solve allowance.")
            if manifest.get("purpose") != "functional" and any(
                row.get(key) != request[key] for key in ("sample_seeds", "collection_seed", "collection_repeat")
            ):
                raise ValueError("Collected physical member seeds or repeat identity differs from the plan.")
            for key in (
                "task",
                "backbone",
                "solver",
                "nfe",
                "context_id",
                "split",
                "schedule_key",
                "seed",
                "density_mass",
                "time_grid",
                "class_id",
            ):
                if row.get(key) != request.get(key):
                    raise ValueError(f"Collected measurement differs from its planned {key}.")
            if (
                manifest.get("purpose") != "functional"
                and manifest["task"] in ("cifar10", "imagenet64")
                and config.image_objective == "kid"
                and "sample_block" not in row
            ):
                raise ValueError("Image KID collection requires complete sample blocks.")
            if "sample_block" in row and row["sample_block"]["seeds"] != request["sample_seeds"]:
                raise ValueError("Collection must retain the complete planned image sample block.")
        if manifest.get("complete_solves") != sum(r["sample_count"] for r in requests.values()):
            raise ValueError("Collection solve accounting differs from its requests.")
    return manifest


def _validate_requests(manifest, config):
    """Validate resolved budgets and identities independently of manifest checksums."""
    records = manifest["inventory"]
    ids = {r["context_id"] for r in records}
    splits = manifest["split_contexts"]
    if len(ids) != len(records) or ids != set(splits["train"]) | set(splits["validation"]):
        raise ValueError("Collection inventory and split identities differ.")
    assigned = manifest["assignments"]
    if set(assigned) != ids:
        raise ValueError("Collection assignments must cover every context.")
    task = manifest["task"]
    if task == "cifar10":
        if len(ids) != 2 or any(set(keys) != set(REFERENCE_KEYS) for keys in assigned.values()):
            raise ValueError("CIFAR needs two disjoint panels covering every reference density.")
        if config.cifar_images % (len(REFERENCE_KEYS) * len(config.generation_seeds)):
            raise ValueError("CIFAR image budget must divide exactly across densities and collection repeats.")
        if splits != {phase: [f"{phase}:unconditional"] for phase in ("train", "validation")}:
            raise ValueError("CIFAR fitting and held-out panel identities differ from their planned splits.")
    else:
        if any(len(keys) != 1 or keys[0] not in REFERENCE_KEYS for keys in assigned.values()):
            raise ValueError("Each context must have exactly one assigned candidate density.")
        counts = [sum(keys == [key] for keys in assigned.values()) for key in REFERENCE_KEYS]
        if min(counts) < 2 or max(counts) - min(counts) > 1:
            raise ValueError("Context assignments must be balanced across the reference pool.")
        if task == "imagenet64":
            if len(ids) != 1000 or {r.get("class_id") for r in records} != set(range(1000)):
                raise ValueError("ImageNet needs all 1000 distinct class contexts.")
            if config.imagenet_images_per_class % len(config.generation_seeds):
                raise ValueError("ImageNet image budget must divide exactly across collection repeats.")
        elif len(ids) > config.context_budget:
            raise ValueError("Collection exceeded its context budget.")
        _validate_split_allocation(records, assigned, splits, config.fitting_fraction)
    phases = {context: phase for phase, contexts in splits.items() for context in contexts}
    expected = _requests(task, manifest["backbone"], records, manifest["reference_support"], assigned, phases, config)
    if expected != manifest["requests"]:
        raise ValueError("Collection requests differ from resolved budgets, assignments or seeds.")


def _validate_split_allocation(records, assigned, splits, fraction):
    source_groups = defaultdict(list)
    for row in records:
        source_groups[row.get("group_id", row["context_id"])].append(row["context_id"])
    if len(source_groups) < len(records):
        if not all(isinstance(row.get("group_id"), str) and row["group_id"] for row in records):
            raise ValueError("Grouped contexts require a source group_id for every entry.")
        counts = {name: len(group) for name, group in source_groups.items()}
        expected = sum(counts[name] for name in _grouped_holdout(counts, fraction))
        if len(splits["validation"]) != expected:
            raise ValueError("Grouped collection split differs from its nearest feasible held-out allocation.")
    else:
        if len(splits["train"]) != math.floor(len(records) * fraction):
            raise ValueError("Collection fitting split size differs from its configured fraction.")
        # First appearances preserve the planner's seeded density order, which
        # breaks equal-remainder ties without introducing a second RNG stream.
        density_groups = defaultdict(list)
        for row in records:
            density_groups[assigned[row["context_id"]][0]].append(row["context_id"])
        expected = _fitting_counts([len(group) for group in density_groups.values()], fraction)
        fitting = set(splits["train"])
        if [sum(context in fitting for context in group) for group in density_groups.values()] != expected:
            raise ValueError("Collection density split differs from its stratified fitting allocation.")


def complete_collection(manifest, rows):
    validate_collection(manifest)
    result = {
        **copy.deepcopy(manifest),
        "state": "complete",
        "measurements_sha256": digest(rows),
        "complete_solves": sum(r["sample_count"] for r in manifest["requests"]),
    }
    validate_collection(result, rows)
    return result


def functional_manifest(rows):
    """Describe explicit small contract-test observations, never research evidence."""
    support = reference_support((r["solver"], r["nfe"]) for r in rows)
    requests = []
    for row in rows:
        request = {
            k: copy.deepcopy(row[k])
            for k in (
                "task",
                "backbone",
                "solver",
                "nfe",
                "context_id",
                "split",
                "schedule_key",
                "seed",
                "density_mass",
                "time_grid",
            )
        }
        if "class_id" in row:
            request["class_id"] = row["class_id"]
        request["sample_seeds"] = row.get("sample_block", {}).get("seeds", [row["seed"]])
        request["sample_count"] = len(request["sample_seeds"])
        request["request_id"] = digest(request)
        requests.append(request)
    manifest = {
        "protocol": "complete-solve-collection",
        "purpose": "functional",
        "task": rows[0]["task"],
        "backbone": rows[0]["backbone"],
        "source_revision": "functional-fixture",
        "config": {**asdict(CollectionConfig()), "generation_seeds": [0, 1]},
        "inventory": [],
        "reference_support": support,
        "density_holdout": density_holdouts(support),
        "split_contexts": {
            s: sorted({r["context_id"] for r in rows if r["split"] == s}) for s in ("train", "validation")
        },
        "requests": requests,
        "state": "planned",
    }
    manifest["plan_sha256"] = digest(manifest)
    bound = [
        {**row, "collection_request_id": request["request_id"]} for row, request in zip(rows, requests, strict=True)
    ]
    return bound, complete_collection(manifest, bound)


def collect(manifest, measure):
    """Execute each frozen request once; callbacks return complete raw observations."""
    validate_collection(manifest)
    if manifest["state"] != "planned":
        raise ValueError("Collection has already completed; use its saved evidence.")
    rows = []
    for request in manifest["requests"]:
        row = measure(copy.deepcopy(request))
        rows.append({**row, "collection_request_id": request["request_id"]})
    return rows, complete_collection(manifest, rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    destination = path.parent / config["output"]
    if destination.exists():
        raise FileExistsError(destination)
    manifest = plan_collection(
        config["task"],
        config["backbone"],
        config["inventory"],
        config["settings"],
        source_revision=config["source_revision"],
        config=CollectionConfig(**config.get("collection", {})),
    )
    if args.plan_only:
        result = {"collection_manifest": manifest}
    else:
        module, name = config["collector"]["factory"].split(":")
        measure = getattr(importlib.import_module(module), name)(config["collector"]["config"])
        rows, manifest = collect(manifest, measure)
        result = {"rows": rows, "collection_manifest": manifest}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
