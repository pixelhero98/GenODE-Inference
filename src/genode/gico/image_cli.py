"""Portable command line interface for image GICO publication artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from genode.artifacts.identity import semantic_sha256
from genode.gico.image_causal_artifacts import (
    load_image_gico_causal_artifact,
    save_image_gico_causal_artifact,
)
from genode.gico.image_causal_rng import derive_image_gico_causal_uniforms
from genode.gico.image_causal_training import (
    ImageGICOCausalTrainingConfig,
    train_image_gico_causal_student,
)
from genode.gico.image_conditional import ImageGICOConditionalTargets
from genode.gico.image_conditional_training import ImageGICOBackboneContextTrainingConfig
from genode.gico.image_students import (
    load_image_gico_deterministic_artifact,
    materialize_image_gico_schedule,
    save_image_gico_deterministic_artifact,
    train_image_gico_deterministic_student,
)
from genode.gico.image_supervision import (
    IMAGE_GICO_STUDENT_KINDS,
    build_image_gico_conditional_supervision,
    build_image_gico_unconditional_supervision,
    load_image_gico_supervision,
    save_image_gico_supervision,
)
from genode.path_safety import is_link_or_reparse_point
from genode.provenance import file_sha256

MATERIALIZATION_PROTOCOL = "image_gico_schedule_materialization_v1"
MATERIALIZATION_NAMESPACE = "image-gico-schedule-materialization-v1"


def _json_object(path: Path) -> dict[str, Any]:
    if is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError(f"Input must be a regular JSON file: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path.name}")
    return payload


def _relative_input(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must remain below the manifest directory.")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or is_link_or_reparse_point(path) or not path.is_file():
        raise ValueError(f"{field} must resolve to a regular manifest-local file.")
    return path


def _load_npy(root: Path, value: object, *, field: str) -> np.ndarray:
    path = _relative_input(root, value, field=field)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{field} must name a .npy file.")
    array = np.load(path, allow_pickle=False)
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise ValueError(f"{field} must contain a finite numeric NumPy array.")
    return np.ascontiguousarray(array)


def _config(path: str | None, config_type: type[Any]) -> Any:
    if path is None:
        return None
    payload = _json_object(Path(path).expanduser().resolve(strict=True))
    try:
        return config_type(**payload)
    except TypeError as exc:
        raise ValueError(f"Invalid {config_type.__name__} fields.") from exc


def _print_identity(kind: str, identity: str) -> None:
    print(json.dumps({"artifact": kind, "sha256": identity}, sort_keys=True))


def _build_targets(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    payload = _json_object(manifest_path)
    root = manifest_path.parent
    kind = payload.get("kind")
    if kind == "conditional_kid":
        targets_payload = _json_object(
            _relative_input(root, payload.get("conditional_targets"), field="conditional_targets")
        )
        supervision = build_image_gico_conditional_supervision(
            targets=ImageGICOConditionalTargets.from_payload(targets_payload),
            fixed_density_mass=_load_npy(root, payload.get("fixed_density_mass"), field="fixed_density_mass"),
            normalized_contexts=_load_npy(root, payload.get("normalized_contexts"), field="normalized_contexts"),
        )
    elif kind == "unconditional_mixture":
        source_identities = payload.get("source_identities")
        if not isinstance(source_identities, Mapping):
            raise ValueError("source_identities must be a JSON object.")
        supervision = build_image_gico_unconditional_supervision(
            target_nfes=payload.get("target_nfes", ()),
            schedule_keys=payload.get("schedule_keys", ()),
            fixed_density_mass=_load_npy(root, payload.get("fixed_density_mass"), field="fixed_density_mass"),
            mixture_weights=_load_npy(root, payload.get("mixture_weights"), field="mixture_weights"),
            source_identities=source_identities,
        )
    else:
        raise ValueError("kind must be conditional_kid or unconditional_mixture.")
    save_image_gico_supervision(supervision, args.output)
    _print_identity("image_gico_supervision", supervision.sha256)
    return 0


def _train_deterministic(args: argparse.Namespace) -> int:
    supervision = load_image_gico_supervision(args.supervision)
    training = train_image_gico_deterministic_student(
        supervision,
        device=args.device,
        config=_config(args.config, ImageGICOBackboneContextTrainingConfig),
    )
    manifest = save_image_gico_deterministic_artifact(training, supervision, args.output)
    _print_identity(str(manifest["artifact"]), str(manifest["artifact_sha256"]))
    return 0


def _train_stochastic(args: argparse.Namespace) -> int:
    supervision = load_image_gico_supervision(args.supervision)
    training = train_image_gico_causal_student(
        supervision,
        device=args.device,
        config=_config(args.config, ImageGICOCausalTrainingConfig),
    )
    manifest = save_image_gico_causal_artifact(training, supervision, args.output)
    _print_identity(str(manifest["artifact"]), str(manifest["artifact_sha256"]))
    return 0


def _validate(args: argparse.Namespace) -> int:
    supervision = load_image_gico_supervision(args.supervision)
    identities: dict[str, str] = {"supervision_sha256": supervision.sha256}
    if args.deterministic is not None:
        artifact = load_image_gico_deterministic_artifact(args.deterministic, supervision, device=args.device)
        identities["deterministic_artifact_sha256"] = artifact.artifact_sha256
    if args.stochastic is not None:
        artifact = load_image_gico_causal_artifact(args.stochastic, supervision, device=args.device)
        identities["stochastic_artifact_sha256"] = artifact.artifact_sha256
    print(json.dumps(identities, sort_keys=True))
    return 0


def _context_indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--context-indices must be comma-separated integers.") from exc
    if not result:
        raise ValueError("--context-indices must not be empty.")
    return result


def _sample_keys(value: str | None, *, count: int) -> tuple[str, ...]:
    if value is None:
        return tuple(str(index) for index in range(count))
    keys = tuple(value.split(","))
    if len(keys) != count or any(not key for key in keys):
        raise ValueError("--sample-keys must provide one nonempty key per context index.")
    return keys


def _publish_materialization(output: str | Path, payload: Any) -> dict[str, Any]:
    payload.verify()
    target = Path(output).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        density_path = stage / "density-mass.npy"
        time_grid_path = stage / "time-grids.npy"
        np.save(density_path, payload.density_mass, allow_pickle=False)
        np.save(time_grid_path, payload.time_grids, allow_pickle=False)
        arrays: dict[str, dict[str, Any]] = {
            "density_mass": {"file": density_path.name, "sha256": file_sha256(density_path)},
            "time_grids": {"file": time_grid_path.name, "sha256": file_sha256(time_grid_path)},
        }
        if payload.tokens is not None:
            token_path = stage / "tokens.npy"
            np.save(token_path, payload.tokens, allow_pickle=False)
            arrays["tokens"] = {"file": token_path.name, "sha256": file_sha256(token_path)}
        body = {
            "protocol": MATERIALIZATION_PROTOCOL,
            "student_kind": payload.student_kind,
            "target_nfe": payload.target_nfe,
            "context_indices": list(payload.context_indices),
            "student_artifact_sha256": payload.artifact_sha256,
            "supervision_sha256": payload.supervision_sha256,
            "uniforms_sha256": payload.uniforms_sha256,
            "arrays": arrays,
        }
        manifest = {
            "artifact": "image_gico_schedule_materialization",
            **body,
            "artifact_sha256": semantic_sha256(body, namespace=MATERIALIZATION_NAMESPACE),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(stage, target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _materialize(args: argparse.Namespace) -> int:
    indices = _context_indices(args.context_indices)
    deterministic = None
    causal = None
    uniforms = None
    if args.student == "deterministic_barycenter":
        if args.uniforms is not None or args.request_sha256 is not None or args.sample_keys is not None:
            raise ValueError("Deterministic materialization does not accept random inputs.")
        deterministic = load_image_gico_deterministic_artifact(args.artifact, device=args.device)
    else:
        causal = load_image_gico_causal_artifact(args.artifact, device=args.device)
        if args.uniforms is not None:
            array = np.load(Path(args.uniforms).expanduser().resolve(strict=True), allow_pickle=False)
            uniforms = torch.as_tensor(np.ascontiguousarray(array, dtype=np.float64), dtype=torch.float64)
        else:
            if args.request_sha256 is None:
                raise ValueError("Stochastic materialization needs --uniforms or --request-sha256.")
            uniforms = derive_image_gico_causal_uniforms(
                artifact_sha256=causal.artifact_sha256,
                request_sha256=args.request_sha256,
                target_nfe=args.target_nfe,
                sample_keys=_sample_keys(args.sample_keys, count=len(indices)),
            )
    result = materialize_image_gico_schedule(
        args.student,
        deterministic_artifact=deterministic,
        causal_artifact=causal,
        target_nfe=args.target_nfe,
        context_indices=indices,
        uniforms=uniforms,
    )
    manifest = _publish_materialization(args.output, result)
    _print_identity(str(manifest["artifact"]), str(manifest["artifact_sha256"]))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genode-image-gico", description="Build, train, validate, and run image GICO students."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-targets", help="Build shared supervision from a portable manifest.")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=_build_targets)

    deterministic = subparsers.add_parser("train-deterministic", help="Train or bind the barycenter student.")
    deterministic.add_argument("--supervision", required=True)
    deterministic.add_argument("--output", required=True)
    deterministic.add_argument("--device", default="cpu")
    deterministic.add_argument("--config")
    deterministic.set_defaults(handler=_train_deterministic)

    stochastic = subparsers.add_parser("train-stochastic", help="Train the causal-AR student.")
    stochastic.add_argument("--supervision", required=True)
    stochastic.add_argument("--output", required=True)
    stochastic.add_argument("--device", default="cpu")
    stochastic.add_argument("--config")
    stochastic.set_defaults(handler=_train_stochastic)

    validate = subparsers.add_parser("validate", help="Strictly load and re-hash artifacts.")
    validate.add_argument("--supervision", required=True)
    validate.add_argument("--deterministic")
    validate.add_argument("--stochastic")
    validate.add_argument("--device", default="cpu")
    validate.set_defaults(handler=_validate)

    materialize = subparsers.add_parser("materialize", help="Freeze one student schedule batch.")
    materialize.add_argument("--student", choices=IMAGE_GICO_STUDENT_KINDS, required=True)
    materialize.add_argument("--artifact", required=True)
    materialize.add_argument("--target-nfe", type=int, choices=(2, 4, 8), required=True)
    materialize.add_argument("--context-indices", required=True)
    materialize.add_argument("--uniforms")
    materialize.add_argument("--request-sha256")
    materialize.add_argument("--sample-keys")
    materialize.add_argument("--device", default="cpu")
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(handler=_materialize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
