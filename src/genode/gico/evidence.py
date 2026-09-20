"""Validated, paired and split-isolated observations for common GICO training."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from genode.gico.clocks import REFERENCE_KEYS, density_identity, verify_measurement_clock
from genode.gico.conditioning import Conditioning
from genode.gico.rewards import RewardCalibration, calibrate_rewards, construct_rewards


def content_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class Evidence:
    cells: list[dict]
    contexts: dict
    conditioning: Conditioning
    calibrations: dict[str, RewardCalibration]
    task: str
    backbone: str
    purpose: str
    evidence_sha256: str
    collection_manifest: dict

    @property
    def reference_support(self):
        return self.collection_manifest["reference_support"]

    @property
    def density_holdout(self):
        return self.collection_manifest["density_holdout"]

    def is_density_holdout(self, row):
        return density_identity(row["density_mass"]) in self.density_holdout[f"{row['solver']}:{row['nfe']}"]

    def groups(self, split: str) -> list[list[dict]]:
        result = defaultdict(list)
        for cell in self.cells:
            if cell["split"] == split:
                result[(cell["solver"], cell["nfe"], cell["context_id"])].append(cell)
        return list(result.values())


def prepare_evidence(
    rows: list[dict],
    contexts: dict,
    *,
    calibration_rows: list[dict] | None = None,
    purpose: str = "research",
    collection_manifest: dict | None = None,
) -> Evidence:
    if purpose not in ("research", "functional"):
        raise ValueError("Evidence purpose must be research or functional.")
    if not rows or any(r.get("split") not in ("train", "validation") for r in rows):
        raise ValueError("Fitting accepts explicit train and validation rows only; test data is forbidden.")
    scopes = {(r["task"], r["backbone"]) for r in rows}
    if len(scopes) != 1:
        raise ValueError("Each policy fit requires one task and frozen backbone.")
    task, backbone = scopes.pop()
    phase_contexts = {
        phase: {r["context_id"] for r in rows if r["split"] == phase} for phase in ("train", "validation")
    }
    if not all(phase_contexts.values()) or phase_contexts["train"] & phase_contexts["validation"]:
        raise ValueError("Training and validation contexts must be nonempty and disjoint.")
    from genode.gico.collection import functional_manifest, validate_collection

    if collection_manifest is None:
        if purpose != "functional":
            raise ValueError("Research fitting requires a completed collection manifest.")
        rows, collection_manifest = functional_manifest(rows)
    validate_collection(collection_manifest, rows)
    if purpose == "research" and collection_manifest.get("purpose") == "functional":
        raise ValueError("Functional observations cannot be relabelled as research evidence.")
    if collection_manifest["split_contexts"] != {s: sorted(v) for s, v in phase_contexts.items()}:
        raise ValueError("Collection context membership differs from the fitting observations.")

    def eligible(row):
        setting = f"{row['solver']}:{row['nfe']}"
        return (
            row["split"] == "train"
            and density_identity(row["density_mass"]) not in collection_manifest["density_holdout"][setting]
        )

    for row in rows:
        verify_measurement_clock(row)
        if row["context_id"] not in contexts:
            raise ValueError(f"Missing native context {row['context_id']!r}.")
        if row["schedule_key"] not in REFERENCE_KEYS:
            raise ValueError("Teacher evidence must use the declared reference-clock pool.")
    if calibration_rows is not None:
        if any(r["context_id"] in phase_contexts["validation"] for r in calibration_rows):
            raise ValueError("Validation contexts cannot be used to calibrate rewards.")
        if any(not eligible(r) for r in calibration_rows):
            raise ValueError("Calibration must contain eligible fitting observations only, excluding both holdouts.")
        if any(content_hash(r) not in {content_hash(x) for x in rows} for r in calibration_rows):
            raise ValueError("Calibration observations must belong to the completed collection.")
    calibration_rows = calibration_rows if calibration_rows is not None else [r for r in rows if eligible(r)]
    all_rows = rows + calibration_rows
    from genode.gico.image_objective import IMAGE_TASKS, validate_image_rows

    validate_image_rows(all_rows)
    if task == "imagenet64" and purpose == "research":
        classes = {phase: {r["class_id"] for r in rows if r["split"] == phase} for phase in phase_contexts}
        if classes["train"] & classes["validation"] or classes["train"] | classes["validation"] != set(range(1000)):
            raise ValueError("Research ImageNet requires all 1000 classes with disjoint fitting/held-out classes.")
    if task.startswith("molecule_"):
        from genode.evaluation.molecule_energy import MoleculeFeatureMap

        by_context = {}
        frozen_maps = {
            content_hash(r["molecule_feature_map"])
            for r in all_rows
            if r["split"] in ("train", "calibration") and "molecule_feature_map" in r
        }
        for row in all_rows:
            if "molecule_feature_map" not in row:
                if purpose == "research" or any("molecule_feature_map" in value for value in all_rows):
                    raise ValueError("Research molecular evidence requires its frozen molecule_feature_map.")
                continue
            MoleculeFeatureMap.from_dict(row["molecule_feature_map"])
            identity = content_hash(row["molecule_feature_map"])
            if row["split"] == "validation" and identity not in frozen_maps:
                raise ValueError("Validation molecular feature map was not frozen from training/calibration evidence.")
            if by_context.setdefault(row["context_id"], identity) != identity:
                raise ValueError("Paired molecular context changed its frozen feature map.")
    bindings = [r.get("backbone_binding") for r in all_rows]
    if any(binding != bindings[0] for binding in bindings):
        raise ValueError("Native backbone bindings must match across calibration, training, and validation.")
    for solver in {r["solver"] for r in all_rows}:
        protocols = {r["measurement_protocol"] for r in all_rows if r["solver"] == solver}
        if len(protocols) != 1:
            raise ValueError(
                "Measurement protocols must match across calibration, training, and validation per solver."
            )
    if any(r["context_id"] in phase_contexts["validation"] for r in calibration_rows):
        raise ValueError("Validation contexts cannot be used to calibrate rewards.")
    for row in calibration_rows:
        verify_measurement_clock(row)
        if (row["task"], row["backbone"]) != (task, backbone):
            raise ValueError("Calibration and fitting scopes differ.")
    calibrations = {}
    cells = []
    for solver in sorted({r["solver"] for r in rows}):
        calibration = [r for r in calibration_rows if r["solver"] == solver]
        training = [r for r in rows if r["solver"] == solver and eligible(r)]
        if task in ("sana", "sd15"):
            calibrated = calibrate_rewards(training, component_calibration_rows=calibration)
        else:
            calibration_nfes = {r["nfe"] for r in calibration}
            training_nfes = {r["nfe"] for r in training}
            if not training_nfes <= calibration_nfes or (task not in IMAGE_TASKS and training_nfes != calibration_nfes):
                raise ValueError(
                    "Reward calibration must cover training NFEs; only image fitting permits a shared calibration NFE superset."
                )
            calibrated = calibrate_rewards(calibration)
        calibrations[solver] = calibrated
        cells.extend(construct_rewards([r for r in rows if r["solver"] == solver], calibrated))
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["split"], cell["solver"], cell["nfe"], cell["context_id"])].append(cell)
    deduplicated = []
    for group in groups.values():
        unique = {}
        for cell in group:
            identity = density_identity(cell["density_mass"])
            if identity in unique:
                previous = unique[identity]
                if not np.allclose(previous["reward_vector"], cell["reward_vector"], atol=1e-10, rtol=1e-6):
                    raise ValueError("Identical reference densities have inconsistent paired measurements.")
                previous["aliases"].append(cell["schedule_key"])
            else:
                unique[identity] = {**cell, "density_sha256": identity, "aliases": [cell["schedule_key"]]}
        deduplicated.extend(unique.values())
    conditioner = Conditioning.fit([r for r in cells if eligible(r)], contexts, unconditional=task == "cifar10")
    return Evidence(
        deduplicated,
        contexts,
        conditioner,
        calibrations,
        task,
        backbone,
        purpose,
        content_hash(
            {
                "measurements": rows,
                "calibration": calibration_rows,
                "contexts": {k: np.asarray(v).tolist() for k, v in sorted(contexts.items())},
                "collection_manifest": collection_manifest,
            }
        ),
        collection_manifest,
    )
