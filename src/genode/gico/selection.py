"""Selected-weight identities and measured utility_surrogate-selection proof."""

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


def utility_surrogate_fingerprint(model, conditioning, step, temperature, stage="ranked_regression"):
    return content_hash(
        {
            "state": state_fingerprint(model),
            "conditioning": conditioning.to_payload(),
            "step": step,
            "temperature": temperature,
            "prediction_stage": stage,
        }
    )


def validate_utility_surrogate_selection(model, conditioning, metadata):
    """Require measured selection and a binding to the actual selected utility_surrogate."""
    from genode.gico.training import selection_key

    history, profile = metadata["history"], metadata["fitting_profile"]
    records = history.get("utility_surrogate", [])
    selected = history.get("utility_surrogate_selection", {})
    steps = list(
        range(
            profile["utility_surrogate_checkpoint_every"],
            profile["utility_surrogate_steps"] + 1,
            profile["utility_surrogate_checkpoint_every"],
        )
    )
    if not steps or steps[-1] != profile["utility_surrogate_steps"]:
        steps.append(profile["utility_surrogate_steps"])
    if [(r.get("step"), r.get("temperature")) for r in records] != [
        (step, temperature) for step in steps for temperature in profile["temperatures"]
    ]:
        raise ValueError("UtilitySurrogate checkpoint/temperature history is incomplete.")
    if metadata.get("purpose") == "research" and any(r.get("density_mse") is None for r in records):
        raise ValueError("Research utility-surrogate selection requires measured density-only holdouts.")
    if not records or any(
        type(r.get("step")) is not int
        or not 1 <= r["step"] <= profile["utility_surrogate_steps"]
        or r.get("temperature") not in profile["temperatures"]
        or not np.isfinite(r.get("regret", np.nan))
        or r["regret"] < 0
        or not np.isfinite(r.get("context_regret", np.nan))
        or (r.get("density_regret") is not None and not np.isfinite(r["density_regret"]))
        or not np.isfinite(r.get("component_mse", np.nan))
        or not np.isfinite(r.get("context_mse", np.nan))
        or r["component_mse"] < 0
        or r["context_mse"] < 0
        or (r.get("density_mse") is not None and (not np.isfinite(r["density_mse"]) or r["density_mse"] < 0))
        or not np.isclose(
            r["component_mse"],
            r["context_mse"] if r.get("density_mse") is None else (r["context_mse"] + r["density_mse"]) / 2,
            rtol=0,
            atol=1e-12,
        )
        or r.get("stage")
        != (
            "density"
            if profile["utility_surrogate_profile"] == "density_context_projection" and r["step"] <= 500
            else "context"
            if profile["utility_surrogate_profile"] == "density_context_projection"
            else "ranked_regression"
        )
        or not np.isclose(
            r["regret"],
            r["context_regret"] if r.get("density_regret") is None else (r["context_regret"] + r["density_regret"]) / 2,
            rtol=0,
            atol=1e-12,
        )
        for r in records
    ):
        raise ValueError("Utility surrogate requires finite measured MSE and temperature selection history.")
    by_step = {step: [r for r in records if r["step"] == step] for step in steps}
    if any(len({row["component_mse"] for row in rows}) != 1 for rows in by_step.values()):
        raise ValueError("Component MSE must be independent of temperature at each checkpoint.")
    selected_step = min(steps, key=lambda step: (by_step[step][0]["component_mse"], step))
    best = min(
        by_step[selected_step],
        key=lambda r: selection_key(r["regret"], r["temperature"], r["step"], profile["preferred_temperature"]),
    )
    if selected != best or selected["temperature"] != metadata["selected_temperature"]:
        raise ValueError("Utility surrogate checkpoint must minimize held-out MSE before temperature regret.")
    if metadata.get("utility_surrogate_prediction_semantics") != (
        "density_only" if selected["stage"] == "density" else "native_context"
    ):
        raise ValueError("Selected utility surrogate prediction semantics disagree with its checkpoint stage.")
    expected = utility_surrogate_fingerprint(
        model, conditioning, selected["step"], selected["temperature"], selected["stage"]
    )
    if history.get("utility_surrogate_selection_fingerprint") != expected:
        raise ValueError(
            "UtilitySurrogate reuse requires a selected-weight fingerprint; use the original runtime for older utility_surrogates."
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
