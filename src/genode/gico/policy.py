"""Canonical policies; inference never loads or invokes the utility_surrogate."""

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
from genode.gico.networks import DeterministicPolicy, ModelConfig, StochasticPolicy, UtilitySurrogate
from genode.gico.profiles import AUXILIARY_NORMALIZATION, TEMPERATURE_UNITS, resolve_profile
from genode.gico.rewards import TASK_METRICS, RewardCalibration, metric_weights
from genode.gico.schedule_hash import json_hash

GICO_PROTOCOL = "genode-gico"
GICO_SCHEMA_VERSION = 1
RNG_PROTOCOL = "sha256-request-seeded-torch-normal"


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


def save_artifact(path, utility_surrogate, policies: dict, conditioning: Conditioning, metadata: dict) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Artifact destination already exists: {destination}")
    if set(policies) - {"deterministic", "stochastic"}:
        raise ValueError("Artifact requires supported policy kinds.")
    artifact_kind = "policy" if policies else "utility_surrogate"
    weight_name = "policy.pt" if policies else "surrogate.pt"
    metadata.update(
        policy_kinds=sorted(policies),
        rng_protocol=RNG_PROTOCOL,
        clock_scope="one_complete_clock_per_generated_trajectory",
        density_bins=64,
        density_uniform_mixture=1e-8,
    )
    payload = {
        "protocol": GICO_PROTOCOL,
        "schema_version": GICO_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "architecture": utility_surrogate.config.to_payload(),
        "conditioning": conditioning.to_payload(),
        "utility_surrogate_conditioning": replace(
            conditioning, context_mode=metadata["fitting_profile"]["utility_surrogate_context_mode"]
        ).to_payload(),
        "architecture_protocol": "density-rope64-silu-additive-once",
        "metadata": metadata,
        "utility_surrogate": {k: v.detach().cpu() for k, v in utility_surrogate.state_dict().items()},
        "policies": {
            kind: {k: v.detach().cpu() for k, v in model.state_dict().items()} for kind, model in policies.items()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gico-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        torch.save(payload, stage / weight_name)
        digest = hashlib.sha256((stage / weight_name).read_bytes()).hexdigest()
        (stage / "manifest.json").write_text(
            json.dumps(
                {
                    "protocol": GICO_PROTOCOL,
                    "schema_version": GICO_SCHEMA_VERSION,
                    "artifact_kind": artifact_kind,
                    "artifact_sha256": digest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for kind in policies:
            load_policy(stage, policy_kind=kind)
        if not policies:
            load_utility_surrogate(stage)
        os.rename(stage, destination)


def _read_artifact(path) -> tuple[dict, str]:
    path = Path(path)
    directory = path.parent if path.name in {"policy.pt", "surrogate.pt"} else path
    manifest_path = directory / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Artifact requires a regular manifest.json file.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        set(manifest) != {"protocol", "schema_version", "artifact_kind", "artifact_sha256"}
        or manifest["protocol"] != GICO_PROTOCOL
        or manifest["schema_version"] != GICO_SCHEMA_VERSION
        or manifest["artifact_kind"] not in {"policy", "utility_surrogate"}
    ):
        raise ValueError("Unsupported artifact version; retrain with the unified GICO protocol.")
    weight_name = "policy.pt" if manifest["artifact_kind"] == "policy" else "surrogate.pt"
    weight_path = directory / weight_name
    if (path.name in {"policy.pt", "surrogate.pt"} and path.name != weight_name) or (
        weight_path.is_symlink() or not weight_path.is_file()
    ):
        raise ValueError("Artifact weight file is missing or disagrees with its manifest kind.")
    data = weight_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["artifact_sha256"]:
        raise ValueError("Artifact checksum mismatch.")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (
        set(payload)
        != {
            "protocol",
            "schema_version",
            "artifact_kind",
            "architecture",
            "architecture_protocol",
            "conditioning",
            "utility_surrogate_conditioning",
            "metadata",
            "utility_surrogate",
            "policies",
        }
        or payload["protocol"] != manifest["protocol"]
        or payload["schema_version"] != GICO_SCHEMA_VERSION
        or payload["artifact_kind"] != manifest["artifact_kind"]
        or (payload["artifact_kind"] == "utility_surrogate" and payload["policies"])
    ):
        raise ValueError("Unsupported policy payload; old architecture loaders have been removed.")
    return payload, digest


def sample_density(model, conditioning, kind, context, solver, nfe, seed=0, request_id=""):
    features = torch.tensor(conditioning.transform(context, solver, nfe)[None])
    with torch.inference_mode():
        mass = (
            model.sample(features, generator=clock_generator(seed, request_id))
            if kind == "stochastic"
            else model(features)
        )
    values = mass[0].numpy().astype(np.float64, copy=True)
    return values / values.sum()


class GICOPolicy:
    def __init__(self, payload: dict, digest: str, policy_kind: str):
        if payload["protocol"] != GICO_PROTOCOL:
            raise ValueError("Old policy selection artifacts require their originating runtime.")
        if payload["artifact_kind"] != "policy":
            raise ValueError("Artifact contains only a utility surrogate; no policy weights are available.")
        self.metadata = payload["metadata"]
        if self.metadata["task"] in ("cifar10", "imagenet64"):
            from genode.gico.image_objective import validate_image_objective, validate_image_split_identities

            validate_image_objective(self.metadata.get("image_objective"))
            objective = self.metadata["image_objective"]
            target = objective.get("target_generator", objective.get("generator"))
            if "generator" in objective and self.metadata.get("backbone_binding") != target:
                raise ValueError("KID artifact frozen generator binding differs from its objective.")
            if target["backbone"] != self.metadata["backbone"] or (
                "backbone_binding" in self.metadata
                and self.metadata["backbone_binding"].get("checkpoint_sha256") != target["checkpoint_sha256"]
            ):
                raise ValueError("Image artifact target and candidate backbone identities disagree.")
            validate_image_split_identities(self.metadata.get("image_split_identities"), protocol=objective["protocol"])
            expected_metric = "lpips" if "target_generator" in objective else "kid"
            if any(
                tuple(c["metric_keys"]) != (expected_metric,) for c in self.metadata["reward_calibrations"].values()
            ):
                raise ValueError("Image objective and reward calibration disagree.")
        self.artifact_sha256 = digest
        self.policy_kind = policy_kind
        if policy_kind not in ("deterministic", "stochastic") or policy_kind not in payload["policies"]:
            raise ValueError("Requested policy kind is absent from this artifact.")
        if sorted(payload["policies"]) != self.metadata["policy_kinds"]:
            raise ValueError("Policy manifest and model states disagree.")
        self.conditioning = Conditioning.from_payload(payload["conditioning"])
        self.utility_surrogate_conditioning = Conditioning.from_payload(payload["utility_surrogate_conditioning"])
        if payload["architecture_protocol"] != "density-rope64-silu-additive-once":
            raise ValueError("Unsupported Transformer positional/conditioning protocol.")
        config = ModelConfig(**payload["architecture"])
        if self.metadata["task"] not in TASK_METRICS or config.metric_count != len(TASK_METRICS[self.metadata["task"]]):
            raise ValueError("Artifact architecture and reward task disagree.")
        required = {
            "fitting_profile",
            "metric_weights",
            "temperature_units",
            "auxiliary_normalization",
            "utility_surrogate_selection_criterion",
            "utility_surrogate_prediction_semantics",
            "policy_selection_criterion",
            "selected_temperature",
            "history",
        }
        if not required <= self.metadata.keys():
            raise ValueError("Artifact lacks the resolved fitting and selection protocols; retrain.")
        profile = dict(self.metadata["fitting_profile"])
        if profile.pop("task", None) != self.metadata["task"]:
            raise ValueError("Artifact fitting profile task mismatch.")
        from dataclasses import fields

        from genode.gico.deterministic_selection import DETERMINISTIC_SELECTION_PROTOCOL, select_deterministic
        from genode.gico.policy_selection import POLICY_SELECTION_PROTOCOL
        from genode.gico.profiles import TrainingConfig
        from genode.gico.stochastic_selection import STOCHASTIC_SELECTION_PROTOCOL, select_stochastic

        profile_fields = {field.name for field in fields(TrainingConfig)}
        criterion = self.metadata["policy_selection_criterion"]
        if set(profile) != profile_fields:
            raise ValueError("Artifact fitting profile is incomplete.")
        training = resolve_profile(self.metadata["task"], **profile)
        if (
            training.policy_context_mode != self.conditioning.context_mode
            or training.utility_surrogate_context_mode != self.utility_surrogate_conditioning.context_mode
            or training.backbone != self.metadata["backbone"]
            or replace(self.utility_surrogate_conditioning, context_mode=self.conditioning.context_mode).to_payload()
            != self.conditioning.to_payload()
            or training.dropout != config.dropout
            or tuple(self.metadata["metric_weights"]) != metric_weights(self.metadata["task"])
            or self.metadata["temperature_units"] != TEMPERATURE_UNITS
            or self.metadata["auxiliary_normalization"] != AUXILIARY_NORMALIZATION
            or self.metadata["utility_surrogate_selection_criterion"] != "heldout_component_mse_then_reference_regret"
            or criterion != POLICY_SELECTION_PROTOCOL
            or self.metadata["selected_temperature"] not in training.temperatures
        ):
            raise ValueError("Artifact fitting, scalarization or normalization protocols disagree.")
        from genode.gico.training import score_coefficient

        selection = self.metadata["history"].get("policy_selection", {}).get(policy_kind, {})
        step = selection.get("step", 0)
        if (
            type(step) is not int
            or not 0.6 * training.policy_steps < step <= training.policy_steps
            or not np.isclose(
                selection.get("coefficient", -1),
                score_coefficient(step - 1, training.policy_steps, training.refinement_weight, training.score_schedule),
            )
        ):
            raise ValueError("Artifact policy checkpoint is not eligible after the score ramp.")
        records = self.metadata["history"].get("policies", {}).get(policy_kind, [])
        if (
            not records
            or any(
                type(row.get("step")) is not int
                or not 1 <= row["step"] <= training.policy_steps
                or not np.isclose(
                    row.get("coefficient", -1),
                    score_coefficient(
                        row["step"] - 1,
                        training.policy_steps,
                        training.refinement_weight,
                        training.score_schedule,
                    ),
                    rtol=0,
                    atol=1e-12,
                )
                for row in records
            )
            or [row["step"] for row in records] != sorted({row["step"] for row in records})
        ):
            raise ValueError("Artifact checkpoint history does not follow its configured score schedule.")
        eligible = [row for row in records if row.get("coefficient", 0) > 0]
        from genode.gico.collection import validate_collection
        from genode.gico.evidence import content_hash
        from genode.gico.selection import validate_utility_surrogate_selection

        validate_utility_surrogate_selection(
            payload["utility_surrogate"], self.utility_surrogate_conditioning, self.metadata
        )
        manifest = self.metadata.get("collection_manifest")
        validate_collection(manifest)
        expected_binding = {
            "evidence_fingerprint": self.metadata["evidence_sha256"],
            "calibration_fingerprint": content_hash(self.metadata["reward_calibrations"]),
            "collection_fingerprint": content_hash(manifest),
            "support_fingerprint": content_hash(manifest["reference_support"]),
            "source_fingerprint": self.metadata.get("source_code_sha256"),
        }
        if any(self.metadata["history"].get(key) != value for key, value in expected_binding.items()):
            raise ValueError("Artifact fitting evidence, calibration, support or source fingerprint differs.")
        if (
            manifest.get("state") != "complete"
            or self.metadata.get("collection_sha256") != content_hash(manifest)
            or manifest["split_contexts"] != self.metadata["split_contexts"]
            or manifest["density_holdout"] != self.metadata["history"]["density_holdout"]
            or manifest["task"] != self.metadata["task"]
            or manifest["backbone"] != self.metadata["backbone"]
            or not isinstance(self.metadata.get("source_code_sha256"), str)
            or len(self.metadata["source_code_sha256"]) != 64
        ):
            raise ValueError("Artifact collection, source or split binding differs.")
        deterministic = policy_kind == "deterministic"
        cadence = training.deterministic_checkpoint_every if deterministic else training.policy_checkpoint_every
        expected_steps = list(range(cadence, training.policy_steps + 1, cadence))
        if not expected_steps or expected_steps[-1] != training.policy_steps:
            expected_steps.append(training.policy_steps)
        protocol = DETERMINISTIC_SELECTION_PROTOCOL if deterministic else STOCHASTIC_SELECTION_PROTOCOL
        selector = select_deterministic if deterministic else select_stochastic
        allowance = training.deterministic_kl_allowance if deterministic else training.stochastic_kl_allowance
        if (
            [row["step"] for row in records] != expected_steps
            or not eligible
            or any(
                row.get("selection_protocol") != protocol
                or row.get("selection_utility_surrogate_fingerprint")
                != self.metadata["history"]["utility_surrogate_selection_fingerprint"]
                or row.get("selection_contexts") != sorted(self.metadata["split_contexts"]["validation"])
                or not isinstance(row.get("predictions_sha256"), str)
                or len(row["predictions_sha256"]) != 64
                or type(row.get("selection_groups")) is not int
                or row["selection_groups"] < 1
                for row in eligible
            )
            or selection != selector(records, allowance)
        ):
            raise ValueError("Artifact policy violates calibrated utility_surrogate-utility/KL selection.")
        if not deterministic and any(
            row.get("clock_replicates") != training.selection_clock_replicates
            or row.get("kl_samples_per_reference") != training.stochastic_likelihood_samples
            for row in eligible
        ):
            raise ValueError("Stochastic selection sampling disagrees with the fitting profile.")
        if (
            config.condition_dim != self.conditioning.width
            or tuple(self.metadata["solvers"]) != self.conditioning.solvers
        ):
            raise ValueError("Artifact conditioning does not match its architecture/solver scope.")
        if self.conditioning.unconditional != (self.metadata["task"] == "cifar10"):
            raise ValueError("Artifact unconditional-context semantics disagree with its task.")
        if (
            self.metadata["rng_protocol"] != RNG_PROTOCOL
            or self.metadata.get("clock_scope") != "one_complete_clock_per_generated_trajectory"
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
        if set(self.metadata["reference_densities"]) != set(self.metadata["reference_grids"]) or set(
            self.metadata["reference_grids"]
        ) != {f"{setting}:{name}" for setting, pool in manifest["reference_support"].items() for name in pool}:
            raise ValueError("Artifact reference densities/grids are incomplete.")
        for key, mass in self.metadata["reference_densities"].items():
            solver, nfe, name = key.split(":", 2)
            planned = manifest["reference_support"].get(f"{solver}:{nfe}", {}).get(name)
            if planned is None or planned["density_mass"] != mass:
                raise ValueError("Artifact reference density differs from the collected support.")
            if solver not in self.conditioning.solvers:
                raise ValueError("Artifact reference solver is outside its conditioning scope.")
            expected = materialize(validate_mass(mass), solver, int(nfe))
            if not np.array_equal(expected, self.metadata["reference_grids"][key]):
                raise ValueError("Artifact reference clock does not match its density realization.")
        with torch.random.fork_rng(devices=[]):
            self.model = DeterministicPolicy(config) if policy_kind == "deterministic" else StochasticPolicy(config)
        self.model.load_state_dict(payload["policies"][policy_kind], strict=True)
        # Validate utility_surrogate structure without constructing it during inference.
        shapes = {
            key: tuple(value.shape)
            for key, value in self.model.state_dict().items()
            if key not in ("ratio_mean", "ratio_scale")
        }
        shapes.update(
            {
                "input.0.weight": (config.width, 3),
                "output.weight": (config.metric_count, config.width),
                "output.bias": (config.metric_count,),
            }
        )
        state = payload["utility_surrogate"]
        if set(state) != set(shapes) or any(
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shapes[key]
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            for key, value in state.items()
        ):
            raise ValueError("Selected utility_surrogate state does not match its Transformer architecture.")
        if any(not bool(torch.isfinite(v).all()) for v in self.model.state_dict().values()):
            raise ValueError("Artifact contains nonfinite model parameters.")
        if policy_kind == "stochastic" and bool((self.model.ratio_scale <= 0).any()):
            raise ValueError("Artifact log-ratio scales must be positive.")
        self.model.eval().requires_grad_(False)
        from genode.gico.selection import candidate_fingerprint

        if selection.get("selection_checkpoint_id") != candidate_fingerprint(
            self.model, self.conditioning, policy_kind, step
        ):
            raise ValueError("Selected checkpoint evidence does not match the stored policy state/conditioning.")

    def density(self, context, solver: str, nfe: int, *, seed: int = 0, request_id: str = "") -> np.ndarray:
        return sample_density(self.model, self.conditioning, self.policy_kind, context, solver, nfe, seed, request_id)

    def materialize(self, context, solver: str, nfe: int, *, seed: int = 0, request_id: str = "") -> tuple[float, ...]:
        return materialize(self.density(context, solver, nfe, seed=seed, request_id=request_id), solver, nfe)


def load_policy(path, *, policy_kind: str = "deterministic", expected_backbone: str | None = None) -> GICOPolicy:
    payload, digest = _read_artifact(path)
    if expected_backbone is not None and payload["metadata"]["backbone"] != expected_backbone:
        raise ValueError("Policy and frozen generator backbone identities differ.")
    return GICOPolicy(payload, digest, policy_kind)


def load_utility_surrogate(path) -> tuple[UtilitySurrogate, Conditioning, dict]:
    """Validate a selected utility surrogate from either supported artifact kind."""
    payload, digest = _read_artifact(path)
    if payload["artifact_kind"] == "policy":
        validated = GICOPolicy(payload, digest, payload["metadata"]["policy_kinds"][0])
        conditioning, metadata = validated.utility_surrogate_conditioning, validated.metadata
    else:
        metadata = payload["metadata"]
        conditioning = Conditioning.from_payload(payload["utility_surrogate_conditioning"])
        from dataclasses import fields

        from genode.gico.collection import validate_collection
        from genode.gico.evidence import content_hash
        from genode.gico.profiles import TrainingConfig

        profile = dict(metadata["fitting_profile"])
        if profile.pop("task", None) != metadata["task"] or set(profile) != {
            field.name for field in fields(TrainingConfig)
        }:
            raise ValueError("Utility-surrogate artifact has an incomplete fitting profile.")
        training = resolve_profile(metadata["task"], **profile)
        manifest = metadata["collection_manifest"]
        validate_collection(manifest)
        config = ModelConfig(**payload["architecture"])
        if (
            metadata["policy_kinds"] != []
            or metadata["history"]["policies"] != {}
            or metadata["history"]["policy_selection"] != {}
            or payload["architecture_protocol"] != "density-rope64-silu-additive-once"
            or config.condition_dim != conditioning.width
            or config.metric_count != len(TASK_METRICS[metadata["task"]])
            or config.dropout != training.dropout
            or training.backbone != metadata["backbone"]
            or training.utility_surrogate_context_mode != conditioning.context_mode
            or payload["conditioning"] != conditioning.to_payload()
            or tuple(metadata["solvers"]) != conditioning.solvers
            or tuple(metadata["metric_weights"]) != metric_weights(metadata["task"])
            or metadata["utility_surrogate_selection_criterion"] != "heldout_component_mse_then_reference_regret"
            or metadata["temperature_units"] != TEMPERATURE_UNITS
            or metadata["auxiliary_normalization"] != AUXILIARY_NORMALIZATION
            or metadata["rng_protocol"] != RNG_PROTOCOL
            or metadata["clock_scope"] != "one_complete_clock_per_generated_trajectory"
            or metadata["density_bins"] != 64
            or metadata["density_uniform_mixture"] != 1e-8
            or metadata["locked_test_used"] is not False
            or metadata["collection_sha256"] != content_hash(manifest)
            or metadata["history"]["evidence_fingerprint"] != metadata["evidence_sha256"]
            or metadata["history"]["calibration_fingerprint"] != content_hash(metadata["reward_calibrations"])
            or metadata["history"]["collection_fingerprint"] != content_hash(manifest)
            or metadata["history"]["support_fingerprint"] != content_hash(manifest["reference_support"])
            or metadata["history"]["source_fingerprint"] != metadata["source_code_sha256"]
            or metadata["split_contexts"]["train"] == []
            or metadata["split_contexts"]["validation"] == []
            or set(metadata["split_contexts"]["train"]) & set(metadata["split_contexts"]["validation"])
            or metadata["selected_temperature"] not in training.temperatures
        ):
            raise ValueError("Utility-surrogate artifact evidence, profile or split binding disagrees.")
        if set(metadata["reference_densities"]) != set(metadata["reference_grids"]) or set(
            metadata["reference_grids"]
        ) != {f"{setting}:{name}" for setting, pool in manifest["reference_support"].items() for name in pool}:
            raise ValueError("Utility-surrogate artifact reference densities/grids are incomplete.")
        for key, mass in metadata["reference_densities"].items():
            solver, nfe, name = key.split(":", 2)
            planned = manifest["reference_support"].get(f"{solver}:{nfe}", {}).get(name)
            if planned is None or planned["density_mass"] != mass:
                raise ValueError("Utility-surrogate artifact reference density differs from collected support.")
            if not np.array_equal(materialize(validate_mass(mass), solver, int(nfe)), metadata["reference_grids"][key]):
                raise ValueError("Utility-surrogate artifact reference grid differs from the shared density decoder.")
        for solver, calibration in metadata["reward_calibrations"].items():
            value = RewardCalibration.from_payload(calibration)
            if (value.task, value.backbone, value.solver) != (metadata["task"], metadata["backbone"], solver):
                raise ValueError("Utility-surrogate calibration scope mismatch.")
            if set(value.calibration_contexts) & set(metadata["split_contexts"]["validation"]):
                raise ValueError("Utility-surrogate calibration includes validation contexts.")
    with torch.random.fork_rng(devices=[]):
        utility_surrogate = UtilitySurrogate(ModelConfig(**payload["architecture"]))
    utility_surrogate.load_state_dict(payload["utility_surrogate"], strict=True)
    utility_surrogate.native_context_width = len(conditioning.context.mean)
    utility_surrogate.density_only = metadata["utility_surrogate_prediction_semantics"] == "density_only"
    if any(not bool(torch.isfinite(v).all()) for v in utility_surrogate.state_dict().values()):
        raise ValueError("Artifact contains nonfinite utility_surrogate parameters.")
    utility_surrogate.eval().requires_grad_(False)
    from genode.gico.selection import validate_utility_surrogate_selection

    validate_utility_surrogate_selection(utility_surrogate, conditioning, metadata)
    return utility_surrogate, conditioning, metadata
