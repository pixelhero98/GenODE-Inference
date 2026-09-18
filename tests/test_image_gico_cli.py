from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from genode.backbones.registry import IMAGE_BACKBONE_REGISTRY
from genode.gico.image_cli import build_argparser, main
from tests.test_image_gico_supervision import image_manifest


def test_prepare_and_train_delegate_to_common_fit(tmp_path):
    manifest = tmp_path / "measurements.json"
    manifest.write_text(json.dumps(image_manifest()), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evaluator = tmp_path / "evaluator.json"
    evaluator.write_text(json.dumps({"factory": "tests.selection_fixtures:cli_factory", "config": {}}))
    assert main(["prepare", "--manifest", str(manifest), "--output", str(evidence)]) == 0
    with patch("genode.gico.training.fit", return_value={"task": "cifar10"}) as fit:
        assert (
            main(
                [
                    "train",
                    "--evidence",
                    str(evidence),
                    "--output",
                    str(tmp_path / "artifact"),
                    "--student-kind",
                    "both",
                    "--selection-evaluator",
                    str(evaluator),
                    "--teacher-score-weight",
                    "0.05",
                ]
            )
            == 0
        )
    assert fit.call_args.kwargs["student_kind"] == "both"
    assert fit.call_args.kwargs["teacher_score_weight"] == 0.05
    from genode.gico.profiles import resolve_profile

    settings = {
        k: v
        for k, v in fit.call_args.kwargs.items()
        if k not in {"student_kind", "device", "purpose", "calibration_rows", "selection_evaluator"}
    }
    assert resolve_profile("cifar10", **settings).student_steps == 2000
    assert len(fit.call_args.args[0]) == 4
    assert all("lpips" in row["metrics"] for row in fit.call_args.args[0])


def test_shared_image_training_flags():
    args = build_argparser().parse_args(
        ["train", "--evidence", "input.json", "--output", "out", "--selection-evaluator", "evaluator.json"]
    )
    assert args.student_kind == "both" and args.teacher_score_weight is None


def test_deterministic_image_fit_needs_no_generator_factory(tmp_path):
    manifest, evidence = tmp_path / "measurements.json", tmp_path / "evidence.json"
    manifest.write_text(json.dumps(image_manifest()), encoding="utf-8")
    main(["prepare", "--manifest", str(manifest), "--output", str(evidence)])
    with patch("genode.gico.training.fit", return_value={}) as fit:
        main(
            [
                "train",
                "--evidence",
                str(evidence),
                "--output",
                str(tmp_path / "policy"),
                "--student-kind",
                "GICO-det-policy",
            ]
        )
    assert fit.call_args.kwargs["selection_evaluator"] is None


@pytest.mark.parametrize("model_key", list(IMAGE_BACKBONE_REGISTRY))
def test_registered_native_backbones_prepare_and_train_with_kid(tmp_path, model_key):
    from genode.gico.rewards import calibrate_rewards, construct_rewards
    from tests.test_native_image_kid import native_manifest

    manifest, evidence, evaluator = (tmp_path / name for name in ("manifest.json", "evidence.json", "evaluator.json"))
    manifest.write_text(json.dumps(native_manifest(model_key)), encoding="utf-8")
    evaluator.write_text(json.dumps({"factory": "tests.selection_fixtures:cli_factory", "config": {}}))
    main(["prepare", "--manifest", str(manifest), "--output", str(evidence)])
    prepared = json.loads(evidence.read_text(encoding="utf-8"))
    assert prepared["metadata"]["protocol"] == "paired-image-kid-v1"
    assert prepared["metadata"]["raw_metric"] == "kid"
    assert all(set(r["metrics"]) == {"kid"} for r in prepared["rows"])
    calibration = calibrate_rewards([r for r in prepared["rows"] if r["split"] == "train"])
    assert calibration.metric_keys == ("kid",)
    cells = construct_rewards(prepared["rows"], calibration)
    assert any(c["metrics"]["kid"] < 0 and c["reward"] > 0 for c in cells)
    with patch("genode.gico.training.fit", return_value={"task": prepared["metadata"]["task"]}) as fit:
        main(
            [
                "train",
                "--evidence",
                str(evidence),
                "--output",
                str(tmp_path / "policy"),
                "--selection-evaluator",
                str(evaluator),
            ]
        )
    assert fit.call_args.kwargs["student_kind"] == "both"
    assert fit.call_args.args[0] == prepared["rows"]
