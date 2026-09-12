from __future__ import annotations

from copy import deepcopy

import pytest

from genode.gico.clocks import materialize, reference_densities
from genode.gico.image_objective import target_identity, validate_image_rows
from genode.gico.image_supervision import prepare_image_rows
from genode.gico.rewards import calibrate_rewards, construct_rewards
from tests.image_fixtures import image_fields
from tests.test_image_primary_runtime import _frozen_cifar_backbone, _frozen_imagenet_backbone


def image_manifest(task="cifar10"):
    backbone = (
        _frozen_cifar_backbone(digest="a" * 64)
        if task == "cifar10"
        else _frozen_imagenet_backbone(digest="b" * 64, offset=0)
    )
    result = {"backbone_manifest": backbone.manifest.to_manifest_dict(), "rows": []}
    if task == "imagenet64":
        result["native_context_table"] = backbone.canonical_conditioning_table().tolist()
    for split in ("train", "validation"):
        for schedule in ("uniform", "late_p_3"):
            mass = reference_densities("euler", 2)[schedule]
            for label in range(1000 if task == "imagenet64" else 1):
                row = {
                    "task": task,
                    "backbone": backbone.manifest.model_key,
                    "solver": "euler",
                    "nfe": 2,
                    "split": split,
                    "seed": 3,
                    "panel_id": split + "-panel",
                    "context_id": f"{split}:{split}-panel:"
                    + (f"class:{label}" if task == "imagenet64" else "unconditional"),
                    "schedule_key": schedule,
                    "density_mass": list(mass),
                    "time_grid": list(materialize(mass, "euler", 2)),
                    "metrics": {"lpips": 0.1 if schedule == "uniform" else 0.09 - (label % 7) * 0.001},
                }
                if task == "imagenet64":
                    row["class_id"] = label
                row = image_fields(row)
                row["image_objective"]["target_generator"]["checkpoint_sha256"] = backbone.manifest.checkpoint.sha256
                row["reference_id"] = target_identity(row["image_objective"], row["target"])
                result["rows"].append(row)
    return result


def test_cifar_uses_zero_context_and_preserves_per_sample_measurements():
    rows, contexts, metadata = prepare_image_rows(image_manifest())
    assert len(rows) == 4 and set(map(tuple, contexts.values())) == {(0.0,)}
    assert rows[0]["metrics"] == {"lpips": 0.1}
    assert metadata["backbone_binding"]["context_source"] == "zero"
    assert metadata["raw_metric_report"][0]["lpips"] == 0.1
    assert rows[0]["context_id"] != rows[2]["context_id"]
    assert rows[0]["context_id"] == rows[1]["context_id"]


def test_old_executed_clock_is_rejected():
    manifest = image_manifest()
    manifest["rows"][1]["time_grid"][1] += 0.01
    with pytest.raises(ValueError, match="recollect"):
        prepare_image_rows(manifest)


def test_imagenet_preserves_native_classes_and_equal_class_reports():
    manifest = image_manifest("imagenet64")
    rows, contexts, metadata = prepare_image_rows(manifest)
    assert len(contexts) == 2000
    assert metadata["backbone_binding"]["context_source"] == "native_class_embedding"
    assert metadata["raw_metric_report"][0]["lpips"] == pytest.approx(0.1)
    assert all("reward_metrics" not in row for row in rows)
    assert rows[0]["reference_id"] != rows[1]["reference_id"]
    manifest["rows"].pop()
    with pytest.raises(ValueError, match="complete equally weighted class"):
        prepare_image_rows(manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("metrics", {"kid": 0.1}),
        ("ensemble_size", 2),
        ("reference_id", "changed"),
        ("image_objective", {"protocol": "kid"}),
        ("panel_id", ""),
    ],
)
def test_incompatible_image_evidence_is_rejected(field, value):
    rows = image_manifest()["rows"]
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_image_rows(rows)


@pytest.mark.parametrize("change", ["noise", "class", "target", "scorer"])
def test_target_pairing_and_scorer_changes_are_rejected(change):
    rows = image_manifest()["rows"]
    row = rows[1]
    if change == "scorer":
        row["image_objective"]["lpips"]["weights_sha256"] = "d" * 64
    else:
        field = {"noise": "noise_sha256", "class": "class_id", "target": "image_sha256"}[change]
        row["target"][field] = 1 if change == "class" else "d" * 64
        row["reference_id"] = target_identity(row["image_objective"], row["target"])
    with pytest.raises(ValueError):
        validate_image_rows(rows)


def test_renamed_panel_cannot_hide_cross_split_target_reuse():
    rows = image_manifest()["rows"]
    for row in rows[2:]:
        row.update(seed=rows[0]["seed"], target=deepcopy(rows[0]["target"]), reference_id=rows[0]["reference_id"])
    with pytest.raises(ValueError, match="disjoint"):
        validate_image_rows(rows)


def test_repeats_average_lpips_without_noise_conditioning_or_log_reweighting():
    rows, contexts, _ = prepare_image_rows(image_manifest())
    training = rows[:2]
    for index, value in enumerate((0.8, 0.4)):
        repeated = deepcopy(training[index])
        repeated["seed"] += 1
        repeated["target"]["seed"] = repeated["seed"]
        repeated["target"]["noise_sha256"] = "e" * 64
        repeated["target"]["image_sha256"] = "f" * 64
        repeated["reference_id"] = target_identity(repeated["image_objective"], repeated["target"])
        repeated["metrics"]["lpips"] = value
        training.append(repeated)
    # Add a distinct reference utility so scalar calibration is nondegenerate.
    for row in deepcopy(training):
        if row["schedule_key"] != "uniform":
            row["schedule_key"] = "late_p_3_reversed"
            row["metrics"]["lpips"] *= 1.5
            training.append(row)
    calibration = calibrate_rewards(training)
    cells = construct_rewards(training, calibration)
    assert len(cells) == 3 and len(contexts) == 2
    candidate = next(row for row in cells if row["schedule_key"] == "late_p_3")
    assert candidate["reward"] * calibration.reward_scale == pytest.approx((0.1 + 0.8 - 0.09 - 0.4) / 2)
    assert next(row for row in cells if row["schedule_key"] == "uniform")["reward"] == 0
    assert calibration.floors == (0.0,) and calibration.component_scales == (1.0,)
    assert len(candidate["reference_ids"]) == 2


def test_lpips_evaluation_preserves_unclamped_float_inputs_and_frozen_scorer():
    import torch

    from genode.gico.image_objective import lpips_values

    class Scorer(torch.nn.Module):
        def forward(self, target, candidate):
            assert candidate.dtype == target.dtype == torch.float32
            assert candidate.max() > 1
            return ((candidate - target) ** 2).mean((1, 2, 3))

    scorer = Scorer().eval()
    candidate = torch.full((2, 3, 32, 32), 2.0, requires_grad=True)
    values = lpips_values(scorer, candidate, torch.zeros_like(candidate))
    values.mean().backward()
    assert torch.isfinite(candidate.grad).all() and candidate.grad.abs().sum() > 0
    scorer.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        lpips_values(scorer, candidate, torch.zeros_like(candidate))


def test_shared_image_calibration_covers_all_nfes_without_validation_access():
    from genode.gico.evidence import prepare_evidence
    from tests.test_unified_gico_rewards import reference_evidence

    rows, contexts = reference_evidence(task="cifar10")
    calibration = deepcopy([row for row in rows if row["split"] == "train"])
    extra = deepcopy(calibration)
    for row in extra:
        row["nfe"] = 8
        row["density_mass"] = list(reference_densities("euler", 8)[row["schedule_key"]])
        row["time_grid"] = list(materialize(row["density_mass"], "euler", 8))
    evidence = prepare_evidence(rows, contexts, calibration_rows=calibration + extra)
    assert evidence.calibrations["euler"].calibration_nfes == (4, 8)
    for row in extra:
        row["split"] = "validation"
    with pytest.raises(ValueError):
        prepare_evidence(rows, contexts, calibration_rows=calibration + extra)


def test_imagenet_scale_balances_classes_with_unequal_panel_counts():
    from tests.test_unified_gico_rewards import measurement

    rows = []
    for label, delta in ((0, 1.0), (1, 5.0)):
        for key, value in (("uniform", 10.0), ("late_p_3", 10.0 - delta)):
            row = measurement(task="imagenet64", context=f"class{label}", schedule=key, metrics={"lpips": value})
            row["class_id"] = label
            row["target"]["class_id"] = label
            row["reference_id"] = target_identity(row["image_objective"], row["target"])
            rows.append(row)
    baseline = calibrate_rewards(rows)
    repeated = deepcopy(rows[:2])
    for row in repeated:
        row["context_id"] += "-extra"
        row["panel_id"] += "-extra"
    assert calibrate_rewards(rows + repeated).reward_scale == pytest.approx(baseline.reward_scale)
    assert baseline.reward_scale == pytest.approx(2.0)


@pytest.mark.parametrize("kind", ["deterministic", "stochastic"])
def test_image_fit_artifact_roundtrip_and_old_objective_rejection(tmp_path, kind, monkeypatch):
    import hashlib
    import json

    import torch

    from genode.gico.policy import load_policy
    from genode.gico.training import fit
    from tests.test_unified_gico_rewards import reference_evidence

    rows, contexts = reference_evidence(task="cifar10")
    from tests.selection_fixtures import evaluator_for

    # Exercise interfaces, not optimizer fitting on the local development host.
    monkeypatch.setattr("genode.gico.training.accumulated_step", lambda *args, **kwargs: 0.0)
    destination = tmp_path / kind
    metadata = fit(
        rows,
        contexts,
        destination,
        student_kind=kind,
        device="cpu",
        teacher_steps=2,
        student_steps=2,
        teacher_checkpoint_every=1,
        student_checkpoint_every=1,
        teacher_score_weight=0.05,
        stochastic_likelihood_samples=1,
        stochastic_score_samples=1,
        selection_evaluator=evaluator_for(rows, contexts),
    )
    assert metadata["image_objective"]["protocol"] == "paired-lpips-v1"
    policy = load_policy(destination, student_kind=kind)
    first = policy.materialize([0.0, 0.0], "euler", 4, seed=2, request_id="sample")
    assert first == policy.materialize([0.0, 0.0], "euler", 4, seed=2, request_id="sample")
    assert len(first) == 5 and all(a < b for a, b in zip(first[:-1], first[1:], strict=True))
    payload = torch.load(destination / "policy.pt", weights_only=True)
    del payload["metadata"]["image_objective"]
    torch.save(payload, destination / "policy.pt")
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["policy_sha256"] = hashlib.sha256((destination / "policy.pt").read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="historical KID"):
        load_policy(destination, student_kind=kind)


def test_target_checkpoint_must_match_bound_candidate_checkpoint():
    manifest = image_manifest()
    for row in manifest["rows"]:
        row["image_objective"]["target_generator"]["checkpoint_sha256"] = "f" * 64
        row["reference_id"] = target_identity(row["image_objective"], row["target"])
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        prepare_image_rows(manifest)


@pytest.mark.parametrize("mutation", ["missing-phase", "missing-list", "bad-hash", "overlap"])
def test_image_artifact_requires_complete_disjoint_split_provenance(mutation):
    from genode.gico.image_objective import validate_image_split_identities

    rows = image_manifest()["rows"]
    provenance = {
        phase: {
            "panels": sorted({r["panel_id"] for r in rows if r["split"] == phase}),
            "targets": sorted({r["reference_id"] for r in rows if r["split"] == phase}),
            "noises": sorted({r["target"]["noise_sha256"] for r in rows if r["split"] == phase}),
        }
        for phase in ("train", "calibration", "validation")
    }
    validate_image_split_identities(provenance)
    if mutation == "missing-phase":
        del provenance["validation"]
    elif mutation == "missing-list":
        del provenance["train"]["targets"]
    elif mutation == "bad-hash":
        provenance["train"]["noises"] = ["not-a-sha"]
    else:
        provenance["validation"]["noises"] = provenance["train"]["noises"]
    with pytest.raises(ValueError):
        validate_image_split_identities(provenance)
