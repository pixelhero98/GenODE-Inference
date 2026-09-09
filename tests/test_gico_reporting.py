"""Held-out reporting validates the executed clocks and frozen measurement scope."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from genode.gico.clocks import materialize, reference_densities
from genode.gico.report_locked_test import summarize_measurements
from genode.gico.rewards import calibrate_rewards, construct_rewards
from tests.test_unified_gico_rewards import measurement, paired_rows


def report_fixture():
    calibration = calibrate_rewards(paired_rows())
    policy = SimpleNamespace(
        artifact_sha256="fixture-policy",
        student_kind="stochastic",
        metadata={
            "split_contexts": {"train": ["train-a", "train-b"], "validation": ["validation-a"]},
            "solvers": ["euler"],
            "reward_calibrations": {"euler": calibration.to_payload()},
            "measurement_protocols": ["paired-terminal-v3"],
        },
    )
    rows = []
    for seed in (1, 2):
        anchor = measurement(context="test-a", split="test", seed=seed)
        candidate = measurement(
            context="test-a",
            split="test",
            seed=seed,
            schedule="late_p_3" if seed == 1 else "late_p_3_reversed",
            metrics={"crps": 1.0 if seed == 1 else 3.0, "mase": 1.0},
        )
        candidate.update(schedule_key="policy", policy_sha256=policy.artifact_sha256, student_kind="stochastic")
        rows.extend([anchor, candidate])
    return policy, calibration, rows


def test_stochastic_report_averages_repeated_metrics_before_log_with_each_clock_validated():
    policy, calibration, rows = report_fixture()
    with pytest.raises(ValueError, match="density_mass"):
        construct_rewards(rows, calibration)
    report = summarize_measurements(rows, policy)
    uniform = next(r for r in report["results"] if r["schedule"] == "uniform")
    candidate = next(r for r in report["results"] if r["schedule"] == "policy")
    expected = (
        np.mean(
            [
                np.log((4 + calibration.floors[0]) / (2 + calibration.floors[0])),
                np.log((2 + calibration.floors[1]) / (1 + calibration.floors[1])),
            ]
        )
        / calibration.reward_scale
    )
    assert uniform["reward_mean"] == 0
    assert candidate["reward_mean"] == pytest.approx(expected)
    assert candidate["paired_contexts"] == 1
    assert not report["selection_performed"]


def test_ensemble_report_validates_every_member_clock_and_count():
    policy, _, rows = report_fixture()
    for row in rows:
        clocks = [{"density_mass": row.pop("density_mass"), "time_grid": row.pop("time_grid")}]
        for key in ("late_p_2", "late_p_2_reversed"):
            mass = reference_densities("euler", 4)[key if row["schedule_key"] != "uniform" else "uniform"]
            clocks.append({"density_mass": list(mass), "time_grid": list(materialize(mass, "euler", 4))})
        row["sample_clocks"] = clocks
    assert summarize_measurements(rows, policy)["results"]
    broken = deepcopy(rows)
    broken[1]["sample_clocks"].pop()
    with pytest.raises(ValueError, match="one clock per ensemble"):
        summarize_measurements(broken, policy)
    rows[1]["sample_clocks"][2]["time_grid"][1] += 0.001
    with pytest.raises(ValueError, match="Measured clock differs"):
        summarize_measurements(rows, policy)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("measurement_protocol", "changed-scorer", "measurement protocol"),
        ("backbone_binding", {"weights": "changed"}, "backbone binding"),
        ("policy_sha256", "other-policy", "policy identity"),
        ("student_kind", "deterministic", "student kind"),
        ("context_id", "train-a", "overlap"),
    ],
)
def test_report_rejects_incompatible_or_leaked_evidence(field, value, match):
    policy, _, rows = report_fixture()
    rows[1][field] = value
    with pytest.raises(ValueError, match=match):
        summarize_measurements(rows, policy)


def test_imagenet_report_uses_complete_equally_weighted_class_panels_as_uncertainty_units():
    policy, _, _ = report_fixture()
    calibration = calibrate_rewards(paired_rows(task="imagenet64"))
    policy.metadata["reward_calibrations"] = {"euler": calibration.to_payload()}
    templates = [
        measurement(task="imagenet64", split="test", metrics={"kid": 2.0}),
        measurement(task="imagenet64", split="test", schedule="late_p_3", metrics={"kid": 1.0}),
    ]
    rows = [
        {**row, "class_id": label, "context_id": f"test-class:{label}"} for row in templates for label in range(1000)
    ]
    report = summarize_measurements(rows, policy)
    result = next(r for r in report["results"] if r["schedule"] == "late_p_3")
    assert result["paired_contexts"] == 1000
    assert result["independent_units"] == 1
    assert result["reward_standard_error"] is None
    assert result["reward_mean"] == pytest.approx(1 / calibration.reward_scale)
    assert result["raw_metrics"] == {"kid": 1.0}
    assert report["uncertainty_unit"] == "paired_panel_mean_over_classes_and_replicates"
    rows = [r for r in rows if r["class_id"] != 999]
    with pytest.raises(ValueError, match="all 1000 classes"):
        summarize_measurements(rows, policy)
