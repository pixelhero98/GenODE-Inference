from copy import deepcopy

import numpy as np
import pytest

from genode.gico.clocks import materialize, reference_densities
from genode.gico.image_objective import validate_image_rows
from genode.gico.kid_objective import KID_OBJECTIVE, cubic_kid, reference_identity
from genode.gico.rewards import calibrate_rewards, construct_rewards
from genode.image_comparators.frozen_clock import warp_frozen_grids


def evidence_rows():
    generator = {
        "backbone": "frozen-bezier",
        "checkpoint_sha256": "a" * 64,
        "transform_sha256": "b" * 64,
        "clock_protocol": "frozen-two-grid-index-warp-v1",
    }
    objective = {
        "protocol": KID_OBJECTIVE,
        "generator": generator,
        "features": {
            "weights_sha256": "c" * 64,
            "implementation_sha256": "d" * 64,
            "input_protocol": "edm-uint8",
            "estimator": "unbiased-cubic",
        },
    }
    rows = []
    for phase, offset in [("train", 0), ("validation", 100)]:
        ref = {"dataset_sha256": "e" * 64, "indices": [offset, offset + 1]}
        for repeat in range(2):
            seed = offset + repeat * 2
            block = {"seeds": [seed, seed + 1], "noise_sha256": [f"{seed:064x}", f"{seed + 1:064x}"]}
            for name, metric in [("uniform", 0.02), ("late_p_3", 0.01), ("early_p_2", -0.01)]:
                # Use a real reference name; retain distinct utilities for calibration.
                if name == "early_p_2":
                    name = "late_p_3_reversed"
                mass = reference_densities("euler", 4)[name]
                rows.append(
                    {
                        "task": "cifar10",
                        "backbone": "frozen-bezier",
                        "solver": "euler",
                        "nfe": 4,
                        "context_id": phase,
                        "panel_id": phase,
                        "split": phase,
                        "seed": seed,
                        "ensemble_size": 2,
                        "reference_id": reference_identity(ref),
                        "reference_block": ref,
                        "sample_block": block,
                        "measurement_protocol": KID_OBJECTIVE,
                        "image_objective": objective,
                        "backbone_binding": generator,
                        "schedule_key": name,
                        "metrics": {"kid": metric + repeat * 0.001},
                        "density_mass": list(mass),
                        "time_grid": list(materialize(mass, "euler", 4)),
                    }
                )
    return rows


def test_paired_kid_direction_frozen_scale_and_repeat_averaging():
    rows = evidence_rows()
    calibration = calibrate_rewards([r for r in rows if r["split"] == "train"])
    assert calibration.metric_keys == ("kid",)
    cells = construct_rewards(rows, calibration)
    for cell in cells:
        if cell["schedule_key"] == "uniform":
            assert cell["reward"] == 0
        else:
            assert cell["reward"] > 0
    assert cells[0]["metrics"]["kid"] == pytest.approx(0.0205)
    with pytest.raises(ValueError, match="only consume"):
        calibrate_rewards(rows)


def test_kid_partial_noise_overlap_and_pairing_rejected():
    rows = evidence_rows()
    row = deepcopy(rows[-1])
    row["sample_block"]["seeds"][1] = 0
    with pytest.raises(ValueError, match="overlap"):
        validate_image_rows(rows[:-1] + [row])
    row = deepcopy(rows[1])
    row["sample_block"]["noise_sha256"][1] = "f" * 64
    with pytest.raises(ValueError, match="blocks differ"):
        validate_image_rows([rows[0], row])


def test_kid_fair_estimator_can_be_negative():
    x = np.array([[0.0], [1.0]])
    assert cubic_kid(x, x) == pytest.approx(-3.5)
    with pytest.raises(ValueError):
        cubic_kid(x[:1], x)


def test_frozen_grid_warp_preserves_uniform_and_changes_both_grids():
    first, second = [0.0, 0.1, 0.4, 0.8, 1.0], [0.0, 0.2, 0.2, 0.7, 1.0]
    a, b = warp_frozen_grids(np.linspace(0, 1, 5), first, second)
    np.testing.assert_array_equal(a, first)
    np.testing.assert_array_equal(b, second)
    a, b = warp_frozen_grids([0, 0.1, 0.2, 0.8, 1], first, second)
    assert np.all(np.diff(a) > 0) and np.all(np.diff(b) >= 0)
    assert a[1] != first[1] and b[1] != second[1]
    with pytest.raises(ValueError):
        warp_frozen_grids([0, 0.2, 0.2, 0.8, 1], first, second)


def test_selection_cannot_replace_members_of_frozen_noise_block():
    from genode.gico.evidence import prepare_evidence
    from genode.gico.networks import DeterministicStudent, ModelConfig
    from genode.gico.reporting import PolicyCandidate, candidate_fingerprint, measured_utility
    from tests.selection_fixtures import evaluator_for

    rows, contexts = evidence_rows(), {"train": [0.0], "validation": [0.0]}
    evidence = prepare_evidence(rows, contexts, purpose="functional")
    model = DeterministicStudent(ModelConfig(evidence.conditioning.width, 1)).eval()
    candidate = PolicyCandidate(
        model,
        evidence.conditioning,
        "GICO-det-policy",
        2000,
        0.1,
        candidate_fingerprint(model, evidence.conditioning, "GICO-det-policy", 2000),
    )
    measured = evaluator_for(rows, contexts)(candidate)
    assert measured_utility(measured, evidence, candidate, clock_replicates=4)["utility"] > 0
    for row in measured[:2]:
        row["sample_block"]["seeds"][1] = 99999
        row["sample_block"]["noise_sha256"][1] = "f" * 64
    with pytest.raises(ValueError, match="sample block differs"):
        measured_utility(measured, evidence, candidate, clock_replicates=4)


def test_kid_artifact_roundtrip_and_transform_identity(tmp_path, monkeypatch):
    import torch

    from genode.gico.policy import GICOPolicy, load_policy
    from genode.gico.training import fit

    rows, contexts = evidence_rows(), {"train": [0.0], "validation": [0.0]}
    monkeypatch.setattr("genode.gico.training.accumulated_step", lambda *a, **kw: 0.0)
    destination = tmp_path / "policy"
    fit(
        rows,
        contexts,
        destination,
        purpose="functional",
        student_kind="GICO-det-policy",
        device="cpu",
        teacher_steps=2,
        student_steps=2,
        teacher_checkpoint_every=1,
        student_checkpoint_every=1,
    )
    policy = load_policy(destination)
    assert len(policy.materialize([0.0], "euler", 4)) == 5
    from genode.gico.report_locked_test import summarize_measurements

    test_rows = deepcopy([r for r in rows if r["split"] == "validation"])
    for row in test_rows:
        row.update(split="test", context_id="new-test", panel_id="new-test")
        row["reference_block"] = dict(row["reference_block"], indices=[9000, 9001])
        row["reference_id"] = reference_identity(row["reference_block"])
    with pytest.raises(ValueError, match="overlap"):
        summarize_measurements(test_rows, policy)
    payload = torch.load(destination / "policy.pt", weights_only=True)
    payload["metadata"]["backbone_binding"] = dict(payload["metadata"]["backbone_binding"], transform_sha256="f" * 64)
    with pytest.raises(ValueError, match="binding differs"):
        GICOPolicy(payload, "fixture", "GICO-det-policy")
