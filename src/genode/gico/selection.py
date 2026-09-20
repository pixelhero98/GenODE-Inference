"""Selected-weight identities and measured teacher-selection proof."""

from __future__ import annotations

import hashlib

import numpy as np

from genode.gico.evidence import content_hash


def state_fingerprint(model) -> str:
    state = model if isinstance(model, dict) else model.state_dict()
    return content_hash(
        {
            key: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in state.items()
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
    steps = list(
        range(profile["teacher_checkpoint_every"], profile["teacher_steps"] + 1, profile["teacher_checkpoint_every"])
    )
    if not steps or steps[-1] != profile["teacher_steps"]:
        steps.append(profile["teacher_steps"])
    if [(r.get("step"), r.get("temperature")) for r in records] != [
        (step, temperature) for step in steps for temperature in profile["temperatures"]
    ]:
        raise ValueError("Teacher checkpoint/temperature history is incomplete.")
    if not records or any(
        type(r.get("step")) is not int
        or not 1 <= r["step"] <= profile["teacher_steps"]
        or r.get("temperature") not in profile["temperatures"]
        or not np.isfinite(r.get("regret", np.nan))
        or r["regret"] < 0
        or not np.isfinite(r.get("context_regret", np.nan))
        or (r.get("density_regret") is not None and not np.isfinite(r["density_regret"]))
        or not np.isclose(
            r["regret"],
            r["context_regret"] if r.get("density_regret") is None else (r["context_regret"] + r["density_regret"]) / 2,
            rtol=0,
            atol=1e-12,
        )
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


def candidate_fingerprint(model, conditioning, kind, step):
    return content_hash(
        {
            "state": state_fingerprint(model),
            "conditioning": conditioning.to_payload(),
            "kind": kind,
            "step": step,
        }
    )
