"""Validated, paired and split-isolated observations for common GICO training."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from genode.gico.clocks import REFERENCE_KEYS, density_identity, reference_densities, verify_measurement_clock
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

    def groups(self, split: str) -> list[list[dict]]:
        result = defaultdict(list)
        for cell in self.cells:
            if cell["split"] == split:
                result[(cell["solver"], cell["nfe"], cell["context_id"])].append(cell)
        return list(result.values())


def prepare_evidence(
    rows: list[dict], contexts: dict, *, calibration_rows: list[dict] | None = None, purpose: str = "research"
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
    for row in rows:
        verify_measurement_clock(row)
        if row["context_id"] not in contexts:
            raise ValueError(f"Missing native context {row['context_id']!r}.")
        if row["schedule_key"] not in REFERENCE_KEYS:
            raise ValueError("Teacher evidence must use the declared reference-clock pool.")
        expected = reference_densities(row["solver"], row["nfe"])[row["schedule_key"]]
        if not np.allclose(row["density_mass"], expected, atol=1e-12, rtol=0):
            raise ValueError("Reference name does not match its realized density.")
    calibration_rows = calibration_rows if calibration_rows is not None else [r for r in rows if r["split"] == "train"]
    all_rows = rows + calibration_rows
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
                if purpose == "research":
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
        training = [r for r in rows if r["solver"] == solver and r["split"] == "train"]
        if task in ("sana", "sd15"):
            calibrated = calibrate_rewards(training, component_calibration_rows=calibration)
        else:
            if {r["nfe"] for r in calibration} != {r["nfe"] for r in training}:
                raise ValueError("Reward calibration must cover exactly the training NFEs for each solver.")
            calibrated = calibrate_rewards(calibration)
        calibrations[solver] = calibrated
        cells.extend(construct_rewards([r for r in rows if r["solver"] == solver], calibrated))
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["split"], cell["solver"], cell["nfe"], cell["context_id"])].append(cell)
    deduplicated = []
    for group in groups.values():
        if purpose == "research" and {r["schedule_key"] for r in group} != set(REFERENCE_KEYS):
            raise ValueError("Research fitting requires the complete 25-reference-clock pool in every cell.")
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
    conditioner = Conditioning.fit(
        [r for r in cells if r["split"] == "train"], contexts, unconditional=task == "cifar10"
    )
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
            }
        ),
    )
