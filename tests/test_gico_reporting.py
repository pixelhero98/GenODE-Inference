"""Held-out reporting validates the executed clocks and frozen measurement scope."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from genode.gico.clocks import materialize, reference_densities
from genode.gico.report_locked_test import collapse_clock_replicates, summarize_measurements
from genode.gico.rewards import calibrate_rewards, construct_rewards
from tests.test_unified_gico_rewards import measurement, paired_rows


def report_fixture():
    calibration = calibrate_rewards(paired_rows())
    policy = SimpleNamespace(
        artifact_sha256="fixture-policy",
        student_kind="GICO-sto-policy",
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
        candidate.update(
            clock_seed=12,
            clock_request_id=f"seed:{seed}:member:0",
            schedule_key="policy",
            policy_sha256=policy.artifact_sha256,
            student_kind="GICO-sto-policy",
        )
        anchor["ensemble_size"] = candidate["ensemble_size"] = 1
        rows.extend([anchor, candidate])
    policy.density = lambda context, solver, nfe, seed=0, request_id="": reference_densities(solver, nfe)[
        "late_p_3" if request_id.startswith("seed:1") else "late_p_3_reversed"
    ]
    return policy, calibration, rows


def test_stochastic_report_averages_repeated_metrics_before_log_with_each_clock_validated():
    policy, calibration, rows = report_fixture()
    with pytest.raises(ValueError, match="density_mass"):
        construct_rewards(rows, calibration)
    report = summarize_measurements(rows, policy, contexts={"test-a": [0.0]})
    uniform = next(r for r in report["results"] if r["schedule"] == "uniform")
    candidate = next(r for r in report["results"] if r["schedule"] == "GICO-sto-policy")
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


@pytest.mark.parametrize("kind", ["GICO-det-policy", "GICO-sto-policy"])
@pytest.mark.parametrize("label", ["policy", "student", "gico_GICO-sto-policy"])
def test_report_renders_verified_student_names_and_preserves_reference_names(kind, label):
    policy, _, rows = report_fixture()
    policy.student_kind = kind
    for row in rows:
        if "policy_sha256" in row:
            row.update(student_kind=kind, schedule_key=label)
    report = summarize_measurements(rows, policy, contexts={"test-a": [0.0]})
    assert report["student_kind"] == kind
    assert {r["schedule"] for r in report["results"]} == {"uniform", kind}


def test_report_preserves_explicit_baseline_names_with_policy_provenance():
    policy, _, rows = report_fixture()
    for row in rows:
        row.update(measurement_role="baseline", policy_sha256=policy.artifact_sha256, student_kind=policy.student_kind)
    report = summarize_measurements(rows, policy)
    assert {r["schedule"] for r in report["results"]} == {"uniform", "policy"}


@pytest.mark.parametrize("member_clocks", [False, True])
def test_report_rejects_reference_label_with_another_executed_clock(member_clocks):
    policy, _, _ = report_fixture()
    rows = [measurement(context="test-a", split="test", schedule=name) for name in ("uniform", "late_p_3")]
    # Density and grid agree with each other but contradict the named reference.
    rows[1]["density_mass"] = list(reference_densities("euler", 4)["late_p_3_reversed"])
    rows[1]["time_grid"] = list(materialize(rows[1]["density_mass"], "euler", 4))
    if member_clocks:
        for row in rows:
            clock = {key: row.pop(key) for key in ("density_mass", "time_grid")}
            row["sample_clocks"] = [deepcopy(clock) for _ in range(row["ensemble_size"])]
    with pytest.raises(ValueError, match="Reference name"):
        summarize_measurements(rows, policy)


@pytest.mark.parametrize("change", ["candidate", "seed", "replicate"])
def test_molecular_report_rejects_geometry_changes_inside_a_comparison(change):
    from genode.gico.rewards import MOLECULE_METRICS

    geometry = {"atom_count": 3, "anchor_triangle": [0, 1, 2], "length_scale": 1.0, "reference_sha256": "a" * 64}
    other_geometry = {**geometry, "length_scale": 2.0}
    training = paired_rows()
    for row in training:
        row.update(
            task="molecule_3d_set1",
            molecule_feature_map=geometry,
            metrics=dict.fromkeys(MOLECULE_METRICS, row["metrics"]["crps"]),
        )
    calibration = calibrate_rewards(training)
    policy, _, _ = report_fixture()
    policy.metadata.update(
        reward_calibrations={"euler": calibration.to_payload()},
        molecular_feature_maps={"first": geometry, "second": other_geometry},
    )
    rows = deepcopy(training[:2])
    for row in rows:
        row.update(context_id="test-molecule", split="test")
    if change == "candidate":
        rows[1]["molecule_feature_map"] = other_geometry
    else:
        repeated = deepcopy(rows)
        for row in repeated:
            row["molecule_feature_map"] = other_geometry
            if change == "seed":
                row["seed"] = 1
            else:
                row["clock_replicate"] = 1
        if change == "replicate":
            for row in rows:
                row["clock_replicate"] = 0
        rows.extend(repeated)
    with pytest.raises(ValueError, match="molecule_feature_map|paired measurement assets"):
        summarize_measurements(rows, policy)


def test_report_collapses_paired_clock_replicates_before_log():
    policy, _, rows = report_fixture()
    repeated = []
    for row in rows:
        for replicate in range(2):
            value = deepcopy(row)
            value["clock_replicate"] = replicate
            if row["schedule_key"] == "policy":
                value["clock_request_id"] += f":replicate:{replicate}"
                value["metrics"]["crps"] += 0.5 * (-1 if replicate == 0 else 1)
            repeated.append(value)
    expected = summarize_measurements(rows, policy, contexts={"test-a": [0.0]})
    actual = summarize_measurements(repeated, policy, contexts={"test-a": [0.0]})
    assert actual["results"] == expected["results"]
    with pytest.raises(ValueError, match="incomplete or unpaired"):
        summarize_measurements(repeated[:-1], policy, contexts={"test-a": [0.0]})
    repeated[1]["metrics"]["crps"] += 0.1
    with pytest.raises(ValueError, match="Repeated uniform"):
        summarize_measurements(repeated, policy, contexts={"test-a": [0.0]})


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_report_rejects_invalid_raw_replicate_before_averaging(invalid):
    policy, _, rows = report_fixture()
    repeated = []
    for row in rows:
        for replicate in range(2):
            value = deepcopy(row)
            value["clock_replicate"] = replicate
            if row["schedule_key"] == "policy":
                value["clock_request_id"] += f":replicate:{replicate}"
                value["metrics"]["crps"] = invalid if replicate == 0 else 3.0
            repeated.append(value)
    with pytest.raises(ValueError, match="finite|nonnegative"):
        summarize_measurements(repeated, policy, contexts={"test-a": [0.0]})


@pytest.mark.parametrize("field,value", [("class_id", 1), ("panel_id", "changed-panel")])
def test_report_validates_every_raw_image_replicate_identity(field, value):
    from genode.gico.image_supervision import prepare_image_rows
    from tests.test_image_gico_supervision import image_manifest

    rows, _, _ = prepare_image_rows(image_manifest())
    repeated = []
    for row in rows:
        for replicate in range(2):
            value_row = deepcopy(row)
            value_row["clock_replicate"] = replicate
            repeated.append(value_row)
    repeated[1][field] = value
    with pytest.raises(ValueError, match="class identity|panels or classes"):
        collapse_clock_replicates(repeated)


def test_ensemble_report_validates_every_member_clock_and_count():
    policy, _, rows = report_fixture()
    for row in rows:
        row["ensemble_size"] = 3
        clocks = [
            {
                "density_mass": row.pop("density_mass"),
                "time_grid": row.pop("time_grid"),
                "clock_seed": 12,
                "clock_request_id": f"seed:{row['seed']}:member:0",
            }
        ]
        for key in ("late_p_2", "late_p_2_reversed"):
            mass = reference_densities("euler", 4)[key if row["schedule_key"] != "uniform" else "uniform"]
            clocks.append(
                {
                    "density_mass": list(mass),
                    "time_grid": list(materialize(mass, "euler", 4)),
                    "clock_seed": 12,
                    "clock_request_id": f"seed:{row['seed']}:member:{len(clocks)}",
                }
            )
        row["sample_clocks"] = clocks
    lookup = {
        c["clock_request_id"]: c["density_mass"]
        for r in rows
        if r["schedule_key"] == "policy"
        for c in r["sample_clocks"]
    }
    policy.density = lambda context, solver, nfe, seed=0, request_id="": lookup[request_id]
    assert summarize_measurements(rows, policy, contexts={"test-a": [0.0]})["results"]
    broken = deepcopy(rows)
    broken[1]["sample_clocks"].pop()
    with pytest.raises(ValueError, match="one clock per ensemble"):
        summarize_measurements(broken, policy, contexts={"test-a": [0.0]})
    rows[1]["sample_clocks"][2]["time_grid"][1] += 0.001
    with pytest.raises(ValueError, match="Measured clock differs"):
        summarize_measurements(rows, policy, contexts={"test-a": [0.0]})


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("measurement_protocol", "changed-scorer", "measurement protocol"),
        ("backbone_binding", {"weights": "changed"}, "backbone binding"),
        ("policy_sha256", "other-policy", "policy identity"),
        ("student_kind", "GICO-det-policy", "student kind"),
        ("context_id", "train-a", "overlap"),
    ],
)
def test_report_rejects_incompatible_or_leaked_evidence(field, value, match):
    policy, _, rows = report_fixture()
    rows[1][field] = value
    with pytest.raises(ValueError, match=match):
        summarize_measurements(rows, policy, contexts={"test-a": [0.0]})


def test_imagenet_report_uses_complete_equally_weighted_class_panels_as_uncertainty_units():
    from genode.gico.image_supervision import prepare_image_rows
    from tests.test_image_gico_supervision import image_manifest

    policy, _, _ = report_fixture()
    prepared, _, _ = prepare_image_rows(image_manifest("imagenet64"))
    calibration = calibrate_rewards([r for r in prepared if r["split"] == "train"])
    policy.metadata["reward_calibrations"] = {"euler": calibration.to_payload()}
    rows = [r for r in prepared if r["split"] == "validation"]
    for row in rows:
        row["split"] = "test"
    policy.metadata["image_objective"] = rows[0]["image_objective"]
    policy.metadata["image_split_identities"] = {
        phase: {"panels": [], "targets": [], "noises": []} for phase in ("train", "validation", "calibration")
    }
    policy.metadata["backbone_binding"] = rows[0]["backbone_binding"]
    policy.metadata["measurement_protocols"] = [rows[0]["measurement_protocol"]]
    report = summarize_measurements(rows, policy, contexts={"test-a": [0.0]})
    result = next(r for r in report["results"] if r["schedule"] == "late_p_3")
    assert result["paired_contexts"] == 1000
    assert result["independent_units"] == 1
    assert result["reward_standard_error"] is None
    expected = np.mean([r["metrics"]["lpips"] for r in rows if r["schedule_key"] == "late_p_3"])
    assert result["reward_mean"] == pytest.approx((0.1 - expected) / calibration.reward_scale)
    assert result["raw_metrics"] == {"lpips": pytest.approx(expected)}
    assert report["uncertainty_unit"] == "paired_panel_mean_over_classes_and_replicates"
    rows = [r for r in rows if r["class_id"] != 999]
    with pytest.raises(ValueError, match="all 1000 classes"):
        summarize_measurements(rows, policy, contexts={"test-a": [0.0]})
