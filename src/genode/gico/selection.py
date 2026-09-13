"""Held-out terminal-utility selection, isolated from student optimization."""

from __future__ import annotations

import copy
import hashlib
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from genode.gico.clocks import materialize, verify_measurement_clock
from genode.gico.codec import wire_role
from genode.gico.conditioning import Conditioning
from genode.gico.evidence import Evidence, content_hash
from genode.gico.image_objective import validate_image_rows
from genode.gico.rewards import LOG_METRICS, construct_rewards

SELECTION_PROTOCOL = "heldout_paired_terminal_utility_v1"


def state_fingerprint(model) -> str:
    return content_hash(
        {
            key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in model.state_dict().items()
        }
    )


def teacher_fingerprint(model, conditioning, step, temperature):
    return content_hash(
        {
            "state": state_fingerprint(model),
            "conditioning": conditioning.to_payload(),
            "step": step,
            "temperature": temperature,
        }
    )


def validate_teacher_selection(model, conditioning, metadata):
    """Require measured selection and a binding to the actual selected teacher."""
    from genode.gico.training import selection_key

    history, profile = metadata["history"], metadata["fitting_profile"]
    records = history.get("teacher", [])
    selected = history.get("teacher_selection", {})
    if not records or any(
        type(r.get("step")) is not int
        or not 1 <= r["step"] <= profile["teacher_steps"]
        or r.get("temperature") not in profile["temperatures"]
        or not np.isfinite(r.get("regret", np.nan))
        for r in records
    ):
        raise ValueError("Teacher requires finite measured checkpoint/temperature selection history.")
    best = min(
        records, key=lambda r: selection_key(r["regret"], r["temperature"], r["step"], profile["preferred_temperature"])
    )
    if selected != best or selected["temperature"] != metadata["selected_temperature"]:
        raise ValueError("Teacher was not selected by minimum held-out reference regret.")
    expected = teacher_fingerprint(model, conditioning, selected["step"], selected["temperature"])
    if history.get("teacher_selection_fingerprint") != expected:
        raise ValueError(
            "Teacher reuse requires a selected-weight fingerprint; use the original runtime for older teachers."
        )


@dataclass(frozen=True)
class StudentCandidate:
    """A detached inference snapshot. Evaluators must return raw paired measurements."""

    model: torch.nn.Module
    conditioning: Conditioning
    student_kind: str
    step: int
    coefficient: float
    checkpoint_id: str

    def density(self, context, solver, nfe, *, seed=0, request_id=""):
        from genode.gico.policy import sample_density

        return sample_density(self.model, self.conditioning, self.student_kind, context, solver, nfe, seed, request_id)

    def materialize(self, context, solver, nfe, *, seed=0, request_id=""):
        return materialize(self.density(context, solver, nfe, seed=seed, request_id=request_id), solver, nfe)


def candidate_fingerprint(model, conditioning, kind, step):
    return content_hash(
        {
            "state": state_fingerprint(model),
            "conditioning": conditioning.to_payload(),
            "kind": wire_role(kind),
            "step": step,
        }
    )


def _check_clock(row, candidate, context, identities):
    if (
        row["schedule_key"] == "student"
        and candidate.student_kind == "GICO-sto-policy"
        and row["ensemble_size"] > 1
        and "sample_clocks" not in row
    ):
        raise ValueError("Stochastic ensemble selection requires one clock per member.")
    clocks = row.get("sample_clocks", [row])
    if "sample_clocks" in row and (len(clocks) != row["ensemble_size"] or "density_mass" in row or "time_grid" in row):
        raise ValueError("Selection requires one complete clock per ensemble member.")
    for clock in clocks:
        verify_measurement_clock({**row, "density_mass": clock["density_mass"], "time_grid": clock["time_grid"]})
        if row["schedule_key"] == "uniform":
            expected = np.full(64, 1 / 64)
        else:
            if candidate.student_kind == "GICO-sto-policy" and (
                type(clock.get("clock_seed")) is not int or not clock.get("clock_request_id")
            ):
                raise ValueError("Stochastic selection requires replayable, independent clock RNG identities.")
            if candidate.student_kind == "GICO-sto-policy":
                identity = (
                    clock["clock_seed"],
                    clock["clock_request_id"],
                )
                if identity in identities:
                    raise ValueError(
                        "Stochastic selection must use independent clocks for each generated member/replicate."
                    )
                identities.add(identity)
            expected = candidate.density(
                context,
                row["solver"],
                row["nfe"],
                seed=clock.get("clock_seed", 0),
                request_id=clock.get("clock_request_id", ""),
            )
        if not np.allclose(expected, clock["density_mass"], atol=1e-12, rtol=0):
            raise ValueError("Selection clock differs from the measured candidate or uniform anchor.")


def measured_utility(
    rows: list[dict], evidence: Evidence, candidate: StudentCandidate, *, clock_replicates: int
) -> dict:
    """Pair first, average repeated metrics, then apply the frozen task calibration."""
    if not rows or any(row.get("split") != "validation" for row in rows):
        raise ValueError("Student selection accepts held-out validation measurements only.")
    validate_image_rows(rows)
    expected = {
        (group[0]["context_id"], group[0]["solver"], group[0]["nfe"]): next(
            row for row in group if row["schedule_key"] == "uniform"
        )
        for group in evidence.groups("validation")
    }
    count = clock_replicates if candidate.student_kind == "GICO-sto-policy" else 1
    measurements = defaultdict(dict)
    identities = set()
    for row in rows:
        key = row["context_id"], row["solver"], row["nfe"]
        if key not in expected:
            raise ValueError("Selection context/settings are outside the held-out panel.")
        anchor = expected[key]
        metrics = evidence.calibrations[key[1]].metric_keys
        if any(not np.isfinite(row.get("metrics", {}).get(k, np.nan)) for k in metrics):
            raise ValueError("Selection metrics must be finite and complete before repeat averaging.")
        if any(row["metrics"][k] < 0 for k in metrics if k in LOG_METRICS or k == "lpips"):
            raise ValueError("Selection error metrics must be nonnegative before repeat averaging.")
        if type(row.get("seed")) is not int:
            raise ValueError("Selection generation seed must be an integer.")
        if row["schedule_key"] not in ("uniform", "student"):
            raise ValueError("Selection requires exactly student and uniform measurements.")
        if row["schedule_key"] == "student" and row.get("selection_checkpoint_id") != candidate.checkpoint_id:
            raise ValueError("Selection measurement refers to a different student checkpoint.")
        for field in (
            "task",
            "backbone",
            "ensemble_size",
            "measurement_protocol",
            "backbone_binding",
            "image_objective",
        ):
            if row.get(field) != anchor.get(field):
                raise ValueError(f"Selection {field} differs from frozen reference evidence.")
        if row["seed"] not in anchor["seeds"] or row["reference_id"] != anchor["reference_ids"][str(row["seed"])]:
            raise ValueError("Selection generation-noise/reference identities differ from the registered panel.")
        if "sample_blocks" in anchor and row.get("sample_block") != anchor["sample_blocks"][str(row["seed"])]:
            raise ValueError("Selection KID sample block differs from frozen evidence.")
        if row.get("molecule_feature_map") != anchor.get("molecule_feature_map"):
            raise ValueError("Selection molecular geometry differs from the frozen feature map.")
        replicate = row.get("clock_replicate", 0)
        if type(replicate) is not int or not 0 <= replicate < count:
            raise ValueError("Selection clock replicate is outside the declared allowance.")
        slot = row["schedule_key"], row["seed"], replicate
        if slot in measurements[key]:
            raise ValueError("Duplicate selection measurement.")
        _check_clock(row, candidate, evidence.contexts[row["context_id"]], identities)
        measurements[key][slot] = row
    if set(measurements) != set(expected):
        raise ValueError("Selection requires the complete held-out context/settings panel.")
    collapsed = []
    for key, panel in measurements.items():
        anchor = expected[key]
        support = {
            (kind, seed, rep) for kind in ("uniform", "student") for seed in anchor["seeds"] for rep in range(count)
        }
        if set(panel) != support:
            raise ValueError("Student and uniform seed/clock replicate panels are incomplete.")
        metric_keys = evidence.calibrations[key[1]].metric_keys
        for seed in anchor["seeds"]:
            uniform = [panel["uniform", seed, rep] for rep in range(count)]
            if any(row["metrics"] != uniform[0]["metrics"] for row in uniform[1:]):
                raise ValueError("Uniform measurements changed between clock-only replicates.")
            for kind in ("uniform", "student"):
                repeated = [panel[kind, seed, rep] for rep in range(count)]
                collapsed.append(
                    {
                        **repeated[0],
                        "metrics": {
                            metric: float(np.mean([r["metrics"][metric] for r in repeated])) for metric in metric_keys
                        },
                    }
                )
        actual_anchor = {
            metric: float(np.mean([panel["uniform", seed, 0]["metrics"][metric] for seed in anchor["seeds"]]))
            for metric in metric_keys
        }
        if any(not np.isclose(actual_anchor[k], anchor["metrics"][k], atol=1e-12, rtol=0) for k in metric_keys):
            raise ValueError("Selection uniform-anchor measurements differ from the frozen panel.")
    cells = []
    for solver, calibration in evidence.calibrations.items():
        cells.extend(
            row
            for row in construct_rewards(
                [r for r in collapsed if r["solver"] == solver], calibration, varying_clocks=True
            )
            if row["schedule_key"] == "student"
        )
    settings = defaultdict(list)
    for cell in cells:
        settings[(cell["solver"], cell["nfe"])].append(cell)
    utilities = []
    for group in settings.values():
        if evidence.task == "imagenet64":
            classes = defaultdict(list)
            for cell in group:
                classes[cell["class_id"]].append(cell["reward"])
            utilities.append(float(np.mean([np.mean(values) for values in classes.values()])))
        else:
            utilities.append(float(np.mean([cell["reward"] for cell in group])))
    utility = float(np.mean(utilities))
    if not np.isfinite(utility):
        raise ValueError("Nonfinite held-out student utility.")
    return {
        "utility": utility,
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_checkpoint_id": candidate.checkpoint_id,
        "measurements_sha256": content_hash(rows),
        "selection_contexts": sorted({key[0] for key in expected}),
        "selection_groups": len(expected),
        "clock_replicates": count,
        "candidate_trajectories": sum(r["ensemble_size"] for r in rows if r["schedule_key"] == "student"),
    }


def evaluate_candidate(model, conditioning, kind, step, coefficient, evaluator: Callable, evidence, clock_replicates):
    """Metrics run on a detached snapshot; restore all ambient training RNG streams."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            snapshot = copy.deepcopy(model).eval().requires_grad_(False).cpu()
            identity = candidate_fingerprint(snapshot, conditioning, kind, step)
            candidate = StudentCandidate(snapshot, copy.deepcopy(conditioning), kind, step, coefficient, identity)
            measurements = evaluator(candidate)
            if candidate_fingerprint(snapshot, candidate.conditioning, kind, step) != identity or any(
                m.training for m in snapshot.modules()
            ):
                raise ValueError("Selection evaluator mutated its candidate snapshot.")
            return measured_utility(measurements, evidence, candidate, clock_replicates=clock_replicates)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
