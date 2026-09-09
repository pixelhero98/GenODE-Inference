from __future__ import annotations

import json
from unittest.mock import patch

from genode.gico.image_cli import build_argparser, main
from tests.test_image_gico_supervision import image_manifest


def test_prepare_and_train_delegate_to_common_fit(tmp_path):
    manifest = tmp_path / "measurements.json"
    manifest.write_text(json.dumps(image_manifest()), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
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
                    "--teacher-score-weight",
                    "0.05",
                ]
            )
            == 0
        )
    assert fit.call_args.kwargs["student_kind"] == "both"
    assert fit.call_args.kwargs["teacher_score_weight"] == 0.05
    assert len(fit.call_args.args[0]) == 4
    assert all("kid" in row["metrics"] for row in fit.call_args.args[0])


def test_shared_image_training_flags():
    args = build_argparser().parse_args(["train", "--evidence", "input.json", "--output", "out"])
    assert args.student_kind == "both" and args.teacher_score_weight == 0.01
