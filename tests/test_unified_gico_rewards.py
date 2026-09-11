"""Calibration and pairing contracts, without fitting any model."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from genode.gico.clocks import REFERENCE_KEYS, materialize, reference_densities, verify_measurement_clock
from genode.gico.conditioning import Conditioning
from genode.gico.evidence import prepare_evidence
from genode.gico.rewards import MOLECULE_METRICS, RewardCalibration, calibrate_rewards, construct_rewards
from genode.schedule_transfer.reference_clocks import build_reference_clock_grid


def measurement(
    *, task="traffic_hourly", context="train-a", split="train", nfe=4, seed=0, schedule="uniform", metrics=None
):
    mass = reference_densities("euler", nfe)[schedule]
    return {
        "task": task,
        "backbone": "frozen-checkpoint-sha",
        "solver": "euler",
        "nfe": nfe,
        "context_id": context,
        "split": split,
        "seed": seed,
        "ensemble_size": 3,
        "reference_id": "fixed-reference-panel",
        "measurement_protocol": "paired-terminal-v3",
        "schedule_key": schedule,
        "metrics": metrics or {"crps": 4.0, "mase": 2.0},
        "density_mass": list(mass),
        "time_grid": list(materialize(mass, "euler", nfe)),
    }


def paired_rows(*, task="traffic_hourly"):
    rows = []
    for context, factor in (("train-a", 0.5), ("train-b", 0.8)):
        anchor = {"crps": 4.0, "mase": 2.0} if task == "traffic_hourly" else {"kid": 0.2}
        candidate = {key: value * factor for key, value in anchor.items()}
        rows.extend(
            (
                measurement(task=task, context=context, metrics=anchor),
                measurement(task=task, context=context, schedule="late_p_3", metrics=candidate),
            )
        )
    return rows


def test_solver_alias_rejected_before_fitting():
    rows, contexts = reference_evidence()
    for row in rows:
        row["solver"] = "Euler"
    with pytest.raises(ValueError, match="canonical name"):
        prepare_evidence(rows, contexts)


def test_log_improvement_does_not_overflow_for_finite_extreme_metrics():
    calibration = calibrate_rewards(paired_rows())
    calibration = RewardCalibration.from_payload({**calibration.to_payload(), "floors": (1e-300, 1e-300)})
    rows = paired_rows()
    for row in rows:
        row["metrics"] = dict.fromkeys(("crps", "mase"), 1e300 if row["schedule_key"] == "uniform" else 1e-300)
    assert all(np.isfinite(row["reward"]) for row in construct_rewards(rows, calibration))


def reference_evidence(*, task="traffic_hourly", complete=True):
    rows, contexts = [], {}
    keys = REFERENCE_KEYS if complete else ("uniform", "late_p_3", "late_p_3_reversed")
    for split, context in (("train", "train-a"), ("validation", "validation-a")):
        contexts[context] = [0.0, 0.0] if task == "cifar10" else ([1.0, 2.0] if split == "train" else [1e6, -1e6])
        for key in keys:
            mass = np.asarray(reference_densities("euler", 4)[key])
            moment = float(mass @ np.linspace(1, 3, 64))
            metrics = {"kid": moment} if task == "cifar10" else {"crps": 3 + moment, "mase": 2 + moment**2}
            rows.append(measurement(task=task, context=context, split=split, schedule=key, metrics=metrics))
    return rows, contexts


def test_repeats_are_averaged_before_log_and_uniform_is_exactly_zero():
    rows = []
    for seed, anchor, candidate in ((1, 2.0, 1.0), (8, 8.0, 2.0)):
        rows += [
            measurement(seed=seed, metrics={"crps": anchor, "mase": anchor}),
            measurement(seed=seed, schedule="late_p_3", metrics={"crps": candidate, "mase": candidate}),
        ]
    rows += [
        measurement(context="train-b"),
        measurement(context="train-b", schedule="late_p_3", metrics={"crps": 3, "mase": 1.5}),
    ]
    calibration = calibrate_rewards(rows)
    cells = construct_rewards(rows, calibration)
    candidate = next(r for r in cells if r["context_id"] == "train-a" and r["schedule_key"] == "late_p_3")
    expected = np.log((5 + np.asarray(calibration.floors)) / (1.5 + np.asarray(calibration.floors)))
    np.testing.assert_allclose(np.asarray(candidate["reward_vector"]) * calibration.reward_scale, expected)
    assert not np.allclose(expected, (np.log(2) + np.log(4)) / 2)
    assert candidate["seeds"] == [1, 8]
    assert candidate["metrics"] == {"crps": 1.5, "mase": 1.5}
    for cell in cells:
        if cell["schedule_key"] == "uniform":
            assert cell["reward"] == 0.0
            assert cell["reward_vector"] == [0.0, 0.0]
    assert calibration.floors == pytest.approx((4.5e-6, 3.5e-6))


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 21),
        ("reference_id", "another-reference"),
        ("ensemble_size", 9),
        ("measurement_protocol", "changed"),
        ("nfe", 8),
        ("solver", "heun"),
        ("context_id", "another-context"),
        ("backbone", "different-checkpoint"),
    ],
)
def test_candidate_pairing_identity_mismatch_is_rejected(field, value):
    rows = paired_rows()
    rows[1][field] = value
    with pytest.raises(ValueError):
        calibrate_rewards(rows)


def test_partial_seed_panels_cannot_be_silently_intersected():
    rows = paired_rows()
    rows.append(measurement(seed=1))
    with pytest.raises(ValueError, match="seed panels differ"):
        calibrate_rewards(rows)


def test_repeated_cells_preserve_seed_to_reference_provenance():
    rows = paired_rows()
    for row in deepcopy(rows[:2]):
        row.update(seed=1, reference_id="different-reference-panel")
        rows.append(row)
    cells = construct_rewards(rows, calibrate_rewards(rows))
    repeated = [cell for cell in cells if cell["context_id"] == "train-a"]
    assert len(repeated) == 2
    for cell in repeated:
        assert cell["reference_ids"] == {"0": "fixed-reference-panel", "1": "different-reference-panel"}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_reward_calibration_refuses_held_out_measurements(split):
    rows = paired_rows()
    rows[-1]["split"] = split
    with pytest.raises(ValueError, match="train/calibration"):
        calibrate_rewards(rows)


def test_calibration_scope_is_frozen_and_errors_have_equal_relative_weight():
    calibration = calibrate_rewards(paired_rows())
    assert calibration.component_scales == (1.0, 1.0)
    cell = {
        "task": "traffic_hourly",
        "backbone": "frozen-checkpoint-sha",
        "solver": "euler",
        "metrics": {"crps": 2.0, "mase": 1.6},
        "anchor_metrics": {"crps": 4.0, "mase": 2.0},
    }
    raw = calibration.vector(cell, normalize=False)
    expected = np.log((np.array([4, 2]) + calibration.floors) / (np.array([2, 1.6]) + calibration.floors))
    np.testing.assert_allclose(raw, expected)
    assert calibration.vector(cell).mean() == pytest.approx(expected.mean() / calibration.reward_scale)
    with pytest.raises(ValueError, match="scope|mismatch"):
        calibration.vector({**cell, "backbone": "changed"})
    assert RewardCalibration.from_payload(calibration.to_payload()) == calibration


def kid_panel(differences, *, nfe=4, context_prefix="cell"):
    rows = []
    for i, delta in enumerate(differences):
        context = f"{context_prefix}-{i}"
        rows += [
            measurement(task="cifar10", context=context, nfe=nfe, metrics={"kid": 20}),
            measurement(task="cifar10", context=context, nfe=nfe, schedule="late_p_3", metrics={"kid": 20 - delta}),
        ]
    return rows


def test_scalar_scale_balances_nfes_and_does_not_center_final_reward():
    rows = kid_panel([1, 3], nfe=4, context_prefix="four") + kid_panel([10, 14], nfe=8, context_prefix="eight")
    reference_scale = np.std([1, 3, 10, 14])
    calibration = calibrate_rewards(rows)
    assert calibration.reward_scale == pytest.approx(reference_scale)
    replicated = rows + sum((kid_panel([1, 3], nfe=4, context_prefix=f"replica-{i}") for i in range(4)), [])
    assert calibrate_rewards(replicated).reward_scale == pytest.approx(reference_scale)
    candidates = [r for r in construct_rewards(rows, calibration) if r["schedule_key"] != "uniform"]
    assert [r["reward"] for r in candidates] == pytest.approx(np.array([1, 3, 10, 14]) / reference_scale)
    assert all(r["reward"] > 0 for r in candidates)


@pytest.mark.parametrize("task", ["sana", "sd15"])
def test_text_to_image_components_are_scaled_before_equal_scalarization(task):
    rows = []
    deltas = np.array([[1, 10], [2, -10], [4, 20]], dtype=float)
    for i, (preference, alignment) in enumerate(deltas):
        rows += [
            measurement(task=task, context=f"prompt-{i}", metrics={"preference": 1, "alignment": 5}),
            measurement(
                task=task,
                context=f"prompt-{i}",
                schedule="late_p_3",
                metrics={"preference": 1 + preference, "alignment": 5 + alignment},
            ),
        ]
    calibration = calibrate_rewards(rows)
    scales = deltas.std(0)
    assert calibration.component_scales == pytest.approx(scales)
    assert calibration.reward_scale == pytest.approx((deltas / scales).mean(1).std())
    original = construct_rewards(rows, calibration)
    changed_units = deepcopy(rows)
    for row in changed_units:
        row["metrics"]["preference"] *= 100
    transformed = construct_rewards(changed_units, calibrate_rewards(changed_units))
    assert [r["reward"] for r in original] == pytest.approx([r["reward"] for r in transformed])


@pytest.mark.parametrize("case", ["zero-anchor", "constant-improvement", "single-molecule-member"])
def test_degenerate_reward_calibrations_are_rejected(case):
    rows = paired_rows()
    if case == "zero-anchor":
        for row in rows:
            if row["schedule_key"] == "uniform":
                row["metrics"] = {"crps": 0, "mase": 0}
    elif case == "constant-improvement":
        rows = kid_panel([1, 1])
    else:
        for row in rows:
            row.update(task="molecule_3d_set1", ensemble_size=1, metrics=dict.fromkeys(MOLECULE_METRICS, 1))
    with pytest.raises(ValueError, match="Degenerate|at least two"):
        calibrate_rewards(rows)


def test_full_reference_evidence_is_split_isolated_and_deduplicated():
    rows, contexts = reference_evidence()
    evidence = prepare_evidence(rows, contexts)
    assert len(REFERENCE_KEYS) == 25
    assert {alias for cell in evidence.groups("train")[0] for alias in cell["aliases"]} == set(REFERENCE_KEYS)
    assert len({r["density_sha256"] for r in evidence.groups("train")[0]}) == len(evidence.groups("train")[0])
    np.testing.assert_array_equal(evidence.conditioning.context.mean, [1, 2])
    assert evidence.calibrations["euler"].calibration_contexts == ("train-a",)
    assert evidence.evidence_sha256 == prepare_evidence(rows, contexts).evidence_sha256


def test_evidence_refuses_calibration_alias_of_validation_context():
    rows, contexts = reference_evidence()
    calibration = [deepcopy(r) for r in rows if r["split"] == "validation"]
    for row in calibration:
        row["split"] = "calibration"
    with pytest.raises(ValueError, match="Validation contexts"):
        prepare_evidence(rows, contexts, calibration_rows=calibration)


def test_evidence_requires_disjoint_fit_splits_and_complete_research_pool():
    rows, contexts = reference_evidence(complete=False)
    with pytest.raises(ValueError, match="complete 25"):
        prepare_evidence(rows, contexts)
    assert prepare_evidence(rows, contexts, purpose="functional").purpose == "functional"
    for row in rows:
        row["context_id"] = "train-a"
    with pytest.raises(ValueError, match="disjoint"):
        prepare_evidence(rows, contexts, purpose="functional")


def test_cifar_contexts_are_explicit_zero_and_normalizers_are_frozen():
    rows, contexts = reference_evidence(task="cifar10")
    evidence = prepare_evidence(rows, contexts)
    encoded = evidence.conditioning.transform([0, 0], "euler", 6)
    assert np.isfinite(encoded).all()
    np.testing.assert_array_equal(encoded[:2], [0, 0])
    with pytest.raises(ValueError, match="nonzero"):
        evidence.conditioning.transform([0, 1], "euler", 4)
    changed = deepcopy(contexts)
    changed["train-a"] = [1, 0]
    with pytest.raises(ValueError, match="zero contexts"):
        prepare_evidence(rows, changed)
    with pytest.raises(ValueError, match="training rows only"):
        Conditioning.fit(rows, contexts)


@pytest.mark.parametrize("solver,nfe", [("euler", 4), ("heun", 8), ("midpoint_rk2", 12)])
def test_all_reference_clocks_use_the_shared_64_bin_realization(solver, nfe):
    densities = reference_densities(solver, nfe)
    assert len(densities) == 25
    assert {"late_p_3", "late_p_3_reversed"} <= densities.keys()
    for schedule, mass in densities.items():
        grid = materialize(mass, solver, nfe)
        steps = nfe if solver == "euler" else nfe // 2
        assert len(grid) == steps + 1 and grid[0] == 0 and grid[-1] == 1
        assert np.all(np.diff(np.asarray(grid, dtype=np.float32)) > 0)
        verify_measurement_clock(
            {"density_mass": mass, "time_grid": grid, "solver": solver, "nfe": nfe, "schedule_key": schedule}
        )


def test_historical_exact_grid_must_be_recollected_when_64_bin_realization_changes():
    densities = reference_densities("euler", 8)
    for schedule, mass in densities.items():
        old = build_reference_clock_grid(schedule, 8)
        if not np.array_equal(old, materialize(mass, "euler", 8)):
            with pytest.raises(ValueError, match="recollect"):
                verify_measurement_clock(
                    {"density_mass": mass, "time_grid": old, "solver": "euler", "nfe": 8, "schedule_key": schedule}
                )
            break
    else:
        pytest.fail("Fixture must exercise a historically different exact reference grid")


def test_conditioning_roundtrip_extrapolates_budget_without_refitting_statistics():
    rows, contexts = reference_evidence()
    conditioning = prepare_evidence(rows, contexts).conditioning
    before = deepcopy(conditioning.to_payload())
    encoded = conditioning.transform(contexts["validation-a"], "euler", 20)
    restored = Conditioning.from_payload(before)
    np.testing.assert_array_equal(restored.transform(contexts["validation-a"], "euler", 20), encoded)
    assert conditioning.to_payload() == before
    with pytest.raises(ValueError, match="not trained"):
        conditioning.transform([1, 2], "heun", 4)
    with pytest.raises(ValueError, match="invalid width"):
        conditioning.transform([1, 2, 3], "euler", 4)


@pytest.mark.parametrize(
    "field,error",
    [
        ("backbone_binding", "Native backbone bindings must match"),
        ("measurement_protocol", "Measurement protocols must match"),
    ],
)
@pytest.mark.parametrize("changed_split", ["calibration", "train", "validation"])
def test_evidence_provenance_matches_across_pilot_training_and_validation(field, error, changed_split):
    rows, contexts = reference_evidence()
    pilot = deepcopy([row for row in rows if row["split"] == "train"])
    for row in pilot:
        row.update(split="calibration", context_id="pilot-only")
    for row in rows + pilot:
        row["backbone_binding"] = {"context_source": "native", "weights_sha256": "a" * 64}
    assert prepare_evidence(rows, contexts, calibration_rows=pilot).calibrations["euler"]
    for row in rows + pilot:
        if row["split"] == changed_split:
            row[field] = (
                {"context_source": "changed", "weights_sha256": "b" * 64}
                if field == "backbone_binding"
                else "different-measurement-protocol"
            )
    with pytest.raises(ValueError, match=error):
        prepare_evidence(rows, contexts, calibration_rows=pilot)


def text_image_panel(deltas, *, nfe, prefix, task="sana", split="train"):
    rows = []
    for index, (preference, alignment) in enumerate(deltas):
        for schedule, metrics in (
            ("uniform", {"preference": 0.0, "alignment": 0.0}),
            ("late_p_3", {"preference": preference, "alignment": alignment}),
        ):
            rows.append(
                measurement(
                    task=task, context=f"{prefix}-{index}", split=split, nfe=nfe, schedule=schedule, metrics=metrics
                )
            )
    return rows


@pytest.mark.parametrize("task", ["sana", "sd15"])
def test_pilot_component_scales_are_frozen_and_training_nfes_set_scalar_scale(task):
    pilot_deltas = np.array([[1, 10], [3, 14], [5, 20], [7, 24]], dtype=float)
    pilot = text_image_panel(pilot_deltas[:2], nfe=4, prefix="pilot4", task=task, split="calibration")
    pilot += text_image_panel(pilot_deltas[2:], nfe=8, prefix="pilot8", task=task, split="calibration")
    train_deltas = np.array([[10, 1], [30, 2], [50, 3], [70, 4]], dtype=float)
    train = text_image_panel(train_deltas[:2], nfe=6, prefix="train6", task=task)
    train += text_image_panel(train_deltas[2:], nfe=12, prefix="train12", task=task)
    calibration = calibrate_rewards(train, component_calibration_rows=pilot)
    expected_components = pilot_deltas.std(axis=0)
    expected_scalar = (train_deltas / expected_components).mean(axis=1).std()
    assert calibration.component_scales == pytest.approx(expected_components)
    assert calibration.reward_scale == pytest.approx(expected_scalar)
    assert calibration.calibration_nfes == (6, 12)
    assert not np.allclose(expected_components, train_deltas.std(axis=0))
    # More observations at NFE6 must not increase its share of either calibration.
    repeated = train + text_image_panel(train_deltas[:2], nfe=6, prefix="extra6", task=task)
    rebalanced = calibrate_rewards(repeated, component_calibration_rows=pilot)
    assert rebalanced.component_scales == calibration.component_scales
    assert rebalanced.reward_scale == pytest.approx(calibration.reward_scale)
    validation = text_image_panel(train_deltas[:2], nfe=6, prefix="validation", task=task, split="validation")
    contexts = {row["context_id"]: [1.0, 2.0] for row in train + validation}
    evidence = prepare_evidence(train + validation, contexts, calibration_rows=pilot, purpose="functional")
    assert evidence.calibrations["euler"] == calibration
    heldout = construct_rewards(validation, calibration)
    assert all(row["reward"] == 0 for row in heldout if row["schedule_key"] == "uniform")
    assert calibration.component_scales == pytest.approx(expected_components)


@pytest.mark.parametrize("invalid", ["missing", "degenerate-scale", "invalid-triangle", "old-version"])
def test_research_molecular_evidence_requires_valid_frozen_feature_map(invalid):
    from genode.evaluation.molecule_energy import MoleculeFeatureMap

    rows, contexts = reference_evidence()
    feature_map = MoleculeFeatureMap.fit([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).to_dict()
    for row in rows:
        row.update(
            task="molecule_3d_set1",
            metrics=dict.fromkeys(MOLECULE_METRICS, row["metrics"]["crps"]),
            molecule_feature_map=deepcopy(feature_map),
        )
    assert prepare_evidence(rows, contexts).task == "molecule_3d_set1"
    altered = next(row for row in rows if row["split"] == "validation")
    if invalid == "missing":
        del altered["molecule_feature_map"]
    elif invalid == "degenerate-scale":
        altered["molecule_feature_map"]["length_scale"] = 0.0
    elif invalid == "invalid-triangle":
        altered["molecule_feature_map"]["anchor_triangle"] = [0, 0, 1]
    else:
        altered["molecule_feature_map"]["version"] = "obsolete_geometry"
    with pytest.raises(ValueError, match="feature_map|length scale|triangle|feature map version"):
        prepare_evidence(rows, contexts)


@pytest.mark.parametrize("mismatch", ["within-context", "unseen-validation-map"])
def test_molecular_feature_maps_are_context_stable_and_validation_maps_come_from_fit(mismatch):
    from genode.evaluation.molecule_energy import MoleculeFeatureMap

    rows, contexts = reference_evidence()
    first_map = MoleculeFeatureMap.fit([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).to_dict()
    second_map = MoleculeFeatureMap.fit([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]).to_dict()
    for row in rows:
        row.update(
            task="molecule_3d_set1",
            metrics=dict.fromkeys(MOLECULE_METRICS, row["metrics"]["crps"]),
            molecule_feature_map=deepcopy(first_map),
        )
    # Distinct members may use distinct geometries when both are present in fitting.
    other_member = deepcopy(rows)
    for row in other_member:
        row["context_id"] += "-second-member"
        contexts[row["context_id"]] = [3.0, 4.0]
        row["molecule_feature_map"] = deepcopy(second_map)
    assert prepare_evidence(rows + other_member, contexts).task == "molecule_3d_set1"
    if mismatch == "within-context":
        rows[0]["molecule_feature_map"] = second_map
    else:
        for row in rows:
            if row["split"] == "validation":
                row["molecule_feature_map"] = deepcopy(second_map)
    with pytest.raises(ValueError, match="[Ff]eature map|[Ff]eature-map|molecule_feature_map"):
        prepare_evidence(rows, contexts)
