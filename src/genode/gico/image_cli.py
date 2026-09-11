"""Prepare small-image evidence and use the shared GICO training/artifact API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from genode.gico.image_supervision import prepare_image_rows
from genode.path_safety import is_link_or_reparse_point


def _read(path: str) -> dict:
    source = Path(path).expanduser()
    if is_link_or_reparse_point(source) or not source.is_file():
        raise ValueError("Image evidence must be a regular JSON file.")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Image evidence must contain a JSON object.")
    return payload


def _write(path: str, payload: dict) -> None:
    target = Path(path).expanduser()
    if target.exists() or is_link_or_reparse_point(target):
        raise FileExistsError("Output already exists; choose a new evidence/result path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate raw paired KID measurements and bind native contexts.")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Fit the common teacher and requested student architectures.")
    train.add_argument("--evidence", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--student-kind", choices=("deterministic", "stochastic", "both"), default="both")
    train.add_argument("--teacher-score-weight", type=float, choices=(0.01, 0.05, 0.1))
    train.add_argument("--teacher-steps", type=int)
    train.add_argument("--student-steps", type=int)
    train.add_argument("--fitting-profile", help="JSON object of common task-profile overrides.")
    train.add_argument("--seed", type=int)
    train.add_argument("--device", default="cuda")
    train.add_argument("--purpose", choices=("research", "functional"), default="research")
    validate = commands.add_parser("validate", help="Validate a common GICO artifact and its identities.")
    validate.add_argument("--policy", required=True)
    validate.add_argument("--student-kind", choices=("deterministic", "stochastic"), default="deterministic")
    materialize = commands.add_parser("materialize", help="Decode a complete clock using the common policy.")
    materialize.add_argument("--policy", required=True)
    materialize.add_argument("--evidence", required=True)
    materialize.add_argument("--context-ids", required=True, help="Comma-separated IDs from prepared native contexts.")
    materialize.add_argument(
        "--sample-keys", required=True, help="Comma-separated independent clock request identities."
    )
    materialize.add_argument("--student-kind", choices=("deterministic", "stochastic"), default="deterministic")
    materialize.add_argument("--nfe", type=int, required=True)
    materialize.add_argument("--clock-seed", type=int, default=0)
    materialize.add_argument("--output", required=True)
    return parser


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    if args.command == "prepare":
        rows, contexts, metadata = prepare_image_rows(_read(args.manifest))
        _write(args.output, {"rows": rows, "contexts": contexts, "metadata": metadata})
        result = {"row_count": len(rows), "context_count": len(contexts), "task": metadata["task"]}
    elif args.command == "train":
        from genode.gico.training import fit

        evidence = _read(args.evidence)
        rows = evidence["rows"]
        if any(row["split"] == "test" for row in rows):
            raise ValueError("Locked-test measurements must not be supplied to image training.")
        calibration_rows = [row for row in rows if row["split"] == "calibration"] or None
        rows = [row for row in rows if row["split"] in {"train", "validation"}]
        if any(row["task"] not in {"cifar10", "imagenet64"} for row in rows):
            raise ValueError("The image CLI trains only small-image tasks.")
        fitting = _read(args.fitting_profile) if args.fitting_profile else {}
        for key in ("teacher_steps", "student_steps", "seed", "teacher_score_weight"):
            if getattr(args, key) is not None:
                fitting[key] = getattr(args, key)
        result = fit(
            rows,
            evidence["contexts"],
            args.output,
            student_kind=args.student_kind,
            **fitting,
            device=args.device,
            purpose=args.purpose,
            calibration_rows=calibration_rows,
        )
    else:
        from genode.gico.policy import load_policy

        policy = load_policy(args.policy, student_kind=args.student_kind)
        if policy.metadata["task"] not in {"cifar10", "imagenet64"}:
            raise ValueError("The image CLI requires a small-image policy.")
        result = {"artifact_sha256": policy.artifact_sha256, "metadata": policy.metadata}
        if args.command == "materialize":
            evidence = _read(args.evidence)
            if evidence["metadata"]["backbone_binding"] != policy.metadata["backbone_binding"]:
                raise ValueError("Native evidence context binding does not match the policy backbone.")
            ids, keys = args.context_ids.split(","), args.sample_keys.split(",")
            if len(ids) != len(keys) or any(not key for key in keys) or len(set(keys)) != len(keys):
                raise ValueError("Provide one unique nonempty clock sample key per context ID.")
            result = {
                "artifact_sha256": policy.artifact_sha256,
                "student_kind": args.student_kind,
                "nfe": args.nfe,
                "clock_seed": args.clock_seed,
                "sample_keys": keys,
                "context_ids": ids,
                "time_grids": [
                    list(
                        policy.materialize(
                            evidence["contexts"][context_id], "euler", args.nfe, seed=args.clock_seed, request_id=key
                        )
                    )
                    for context_id, key in zip(ids, keys, strict=True)
                ],
            }
            _write(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
