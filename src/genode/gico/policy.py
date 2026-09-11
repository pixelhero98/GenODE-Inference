"""Versioned unified policies; inference never loads or invokes the teacher."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from genode.gico.clocks import clock_generator, materialize, validate_mass
from genode.gico.conditioning import Conditioning
from genode.gico.networks import DensityTeacher, DeterministicStudent, ModelConfig, StochasticStudent
from genode.gico.profiles import AUXILIARY_NORMALIZATION, TEMPERATURE_UNITS, resolve_profile
from genode.gico.rewards import TASK_METRICS, RewardCalibration, metric_weights
from genode.gico.schedule_hash import json_hash

GICO_PROTOCOL = "genode-gico-v5"
RNG_PROTOCOL = "sha256-request-seeded-torch-normal-v1"


def stable_context_id(
    *,
    dataset: str,
    split_phase: str,
    example_idx: int,
    series_id: str,
    series_idx: int,
    target_t: int,
    history_start: int | None = None,
    history_stop: int | None = None,
    context_schema: str = "forecast_window",
) -> str:
    return json_hash(
        {
            "context_schema": context_schema,
            "dataset": dataset,
            "split_phase": split_phase,
            "example_idx": int(example_idx),
            "series_id": str(series_id),
            "series_idx": int(series_idx),
            "target_t": int(target_t),
            "history_start": history_start,
            "history_stop": history_stop,
        },
        prefix="ctx",
    )


def save_context_embedding_table(path, embeddings, *, metadata=None) -> dict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(embeddings)
    matrix = np.asarray([embeddings[key] for key in ids], dtype=np.float32)
    if not ids or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Context table must be a nonempty finite matrix.")
    np.savez_compressed(destination, context_ids=np.asarray(ids), embeddings=matrix)
    manifest = {
        "protocol": GICO_PROTOCOL,
        "context_count": len(ids),
        "embedding_dim": matrix.shape[1],
        "metadata": metadata or {},
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def load_context_embedding_table(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["context_ids"].tolist()
        values = archive["embeddings"]
    if (
        not ids
        or len(ids) != len(set(ids))
        or values.ndim != 2
        or len(values) != len(ids)
        or not all(isinstance(k, str) and k for k in ids)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Context table has invalid identities, shape, or values.")
    return {key: np.array(value, dtype=np.float32, copy=True) for key, value in zip(ids, values, strict=True)}


def save_artifact(path, teacher, students: dict, conditioning: Conditioning, metadata: dict) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Artifact destination already exists: {destination}")
    if not students or set(students) - {"deterministic", "stochastic"}:
        raise ValueError("Artifact requires supported student kinds.")
    metadata.update(
        student_kinds=sorted(students),
        rng_protocol=RNG_PROTOCOL,
        clock_scope="one_complete_clock_per_generated_trajectory",
        density_bins=64,
        density_uniform_mixture=1e-8,
    )
    payload = {
        "protocol": GICO_PROTOCOL,
        "architecture": teacher.config.to_payload(),
        "conditioning": conditioning.to_payload(),
        "teacher_conditioning": replace(
            conditioning, context_mode=metadata["fitting_profile"]["teacher_context_mode"]
        ).to_payload(),
        "architecture_protocol": "density-rope64-silu-additive-once-v1",
        "metadata": metadata,
        "teacher": {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
        "students": {
            kind: {k: v.detach().cpu() for k, v in model.state_dict().items()} for kind, model in students.items()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gico-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        torch.save(payload, stage / "policy.pt")
        digest = hashlib.sha256((stage / "policy.pt").read_bytes()).hexdigest()
        (stage / "manifest.json").write_text(
            json.dumps({"protocol": GICO_PROTOCOL, "policy_sha256": digest}, indent=2), encoding="utf-8"
        )
        for kind in students:
            load_policy(stage, student_kind=kind)
        os.rename(stage, destination)


def _read_artifact(path) -> tuple[dict, str]:
    path = Path(path)
    directory = path.parent if path.name == "policy.pt" else path
    manifest_path, weight_path = directory / "manifest.json", directory / "policy.pt"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, weight_path)):
        raise ValueError(
            "Unified artifact requires regular manifest.json and policy.pt files; old formats are unsupported."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"protocol", "policy_sha256"} or manifest["protocol"] != GICO_PROTOCOL:
        raise ValueError("Unsupported artifact version; retrain with the unified GICO protocol.")
    data = weight_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["policy_sha256"]:
        raise ValueError("Policy artifact checksum mismatch.")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (
        set(payload)
        != {
            "protocol",
            "architecture",
            "architecture_protocol",
            "conditioning",
            "teacher_conditioning",
            "metadata",
            "teacher",
            "students",
        }
        or payload["protocol"] != GICO_PROTOCOL
    ):
        raise ValueError("Unsupported policy payload; old architecture loaders have been removed.")
    return payload, digest


class GICOPolicy:
    def __init__(self, payload: dict, digest: str, student_kind: str):
        self.metadata = payload["metadata"]
        self.artifact_sha256 = digest
        self.student_kind = student_kind
        if student_kind not in ("deterministic", "stochastic") or student_kind not in payload["students"]:
            raise ValueError("Requested student kind is absent from this artifact.")
        if sorted(payload["students"]) != self.metadata["student_kinds"]:
            raise ValueError("Student manifest and model states disagree.")
        self.conditioning = Conditioning.from_payload(payload["conditioning"])
        self.teacher_conditioning = Conditioning.from_payload(payload["teacher_conditioning"])
        if payload["architecture_protocol"] != "density-rope64-silu-additive-once-v1":
            raise ValueError("Unsupported Transformer positional/conditioning protocol.")
        config = ModelConfig(**payload["architecture"])
        if self.metadata["task"] not in TASK_METRICS or config.metric_count != len(TASK_METRICS[self.metadata["task"]]):
            raise ValueError("Artifact architecture and reward task disagree.")
        required = {
            "fitting_profile",
            "metric_weights",
            "temperature_units",
            "auxiliary_normalization",
            "teacher_selection_criterion",
            "student_selection_criterion",
            "selected_temperature",
            "history",
        }
        if not required <= self.metadata.keys():
            raise ValueError("Artifact lacks the resolved fitting and selection protocols; retrain.")
        profile = dict(self.metadata["fitting_profile"])
        if profile.pop("task", None) != self.metadata["task"]:
            raise ValueError("Artifact fitting profile task mismatch.")
        from dataclasses import fields

        from genode.gico.profiles import TrainingConfig

        if set(profile) != {field.name for field in fields(TrainingConfig)}:
            raise ValueError("Artifact fitting profile is incomplete.")
        training = resolve_profile(self.metadata["task"], **profile)
        if (
            training.student_context_mode != self.conditioning.context_mode
            or training.teacher_context_mode != self.teacher_conditioning.context_mode
            or training.backbone != self.metadata["backbone"]
            or replace(self.teacher_conditioning, context_mode=self.conditioning.context_mode).to_payload()
            != self.conditioning.to_payload()
            or training.dropout != config.dropout
            or tuple(self.metadata["metric_weights"]) != metric_weights(self.metadata["task"])
            or self.metadata["temperature_units"] != TEMPERATURE_UNITS
            or self.metadata["auxiliary_normalization"] != AUXILIARY_NORMALIZATION
            or self.metadata["teacher_selection_criterion"] != "heldout_reference_utility_regret"
            or self.metadata["student_selection_criterion"] != "post_ramp_validation_distillation"
            or self.metadata["selected_temperature"] not in training.temperatures
        ):
            raise ValueError("Artifact fitting, scalarization or normalization protocols disagree.")
        from genode.gico.training import score_coefficient

        selection = self.metadata["history"].get("student_selection", {}).get(student_kind, {})
        step = selection.get("step", 0)
        if (
            type(step) is not int
            or not 0.6 * training.student_steps < step <= training.student_steps
            or not np.isclose(
                selection.get("coefficient", -1),
                score_coefficient(step - 1, training.student_steps, training.teacher_score_weight),
            )
        ):
            raise ValueError("Artifact student checkpoint is not eligible after the score ramp.")
        if (
            config.condition_dim != self.conditioning.width
            or tuple(self.metadata["solvers"]) != self.conditioning.solvers
        ):
            raise ValueError("Artifact conditioning does not match its architecture/solver scope.")
        if self.conditioning.unconditional != (self.metadata["task"] == "cifar10"):
            raise ValueError("Artifact unconditional-context semantics disagree with its task.")
        if (
            self.metadata["rng_protocol"] != RNG_PROTOCOL
            or self.metadata["density_bins"] != 64
            or self.metadata["density_uniform_mixture"] != 1e-8
            or self.metadata["locked_test_used"] is not False
        ):
            raise ValueError("Artifact runtime or split protocol is incompatible.")
        splits = self.metadata["split_contexts"]
        if not splits["train"] or not splits["validation"] or set(splits["train"]) & set(splits["validation"]):
            raise ValueError("Artifact train and validation contexts are not disjoint.")
        if set(self.metadata["reward_calibrations"]) != set(self.conditioning.solvers):
            raise ValueError("Artifact must calibrate every supported solver.")
        for solver, calibration in self.metadata["reward_calibrations"].items():
            value = RewardCalibration.from_payload(calibration)
            if (value.task, value.backbone, value.solver) != (self.metadata["task"], self.metadata["backbone"], solver):
                raise ValueError("Artifact reward calibration scope mismatch.")
            if set(value.calibration_contexts) & set(splits["validation"]):
                raise ValueError("Artifact calibration includes validation contexts.")
        if (
            set(self.metadata["reference_densities"]) != set(self.metadata["reference_grids"])
            or not self.metadata["reference_grids"]
        ):
            raise ValueError("Artifact reference densities/grids are incomplete.")
        for key, mass in self.metadata["reference_densities"].items():
            solver, nfe, _ = key.split(":", 2)
            if solver not in self.conditioning.solvers:
                raise ValueError("Artifact reference solver is outside its conditioning scope.")
            expected = materialize(validate_mass(mass), solver, int(nfe))
            if not np.array_equal(expected, self.metadata["reference_grids"][key]):
                raise ValueError("Artifact reference clock does not match its density realization.")
        with torch.random.fork_rng(devices=[]):
            self.model = DeterministicStudent(config) if student_kind == "deterministic" else StochasticStudent(config)
        self.model.load_state_dict(payload["students"][student_kind], strict=True)
        if any(not bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()):
            raise ValueError("Artifact contains nonfinite model parameters.")
        if student_kind == "stochastic" and bool((self.model.ratio_scale <= 0).any()):
            raise ValueError("Artifact log-ratio scales must be positive.")
        self.model.eval().requires_grad_(False)

    def density(self, context, solver: str, nfe: int, *, seed: int = 0, request_id: str = "") -> np.ndarray:
        features = torch.tensor(self.conditioning.transform(context, solver, nfe)[None])
        with torch.inference_mode():
            if self.student_kind == "stochastic":
                mass = self.model.sample(features, generator=clock_generator(seed, request_id))
            else:
                mass = self.model(features)
        values = mass[0].numpy().astype(np.float64, copy=True)
        return values / values.sum()

    def materialize(self, context, solver: str, nfe: int, *, seed: int = 0, request_id: str = "") -> tuple[float, ...]:
        return materialize(self.density(context, solver, nfe, seed=seed, request_id=request_id), solver, nfe)


def load_policy(path, *, student_kind: str = "deterministic", expected_backbone: str | None = None) -> GICOPolicy:
    payload, digest = _read_artifact(path)
    if expected_backbone is not None and payload["metadata"]["backbone"] != expected_backbone:
        raise ValueError("Policy and frozen generator backbone identities differ.")
    return GICOPolicy(payload, digest, student_kind)


def load_teacher(path) -> tuple[DensityTeacher, Conditioning, dict]:
    """Explicit research control; ordinary policy inference never calls this."""
    payload, digest = _read_artifact(path)
    # Apply the same architecture, profile, split and normalization contract as inference.
    validated = GICOPolicy(payload, digest, payload["metadata"]["student_kinds"][0])
    with torch.random.fork_rng(devices=[]):
        teacher = DensityTeacher(ModelConfig(**payload["architecture"]))
    teacher.load_state_dict(payload["teacher"], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in teacher.state_dict().values()):
        raise ValueError("Artifact contains nonfinite teacher parameters.")
    teacher.eval().requires_grad_(False)
    return teacher, validated.teacher_conditioning, validated.metadata
