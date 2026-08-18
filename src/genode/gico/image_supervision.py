"""Shared supervision for deterministic and causal-AR image GICO students.

The supervision object is the scientific boundary between reward construction
and deployment.  It stores a finite schedule law once: the deterministic
student consumes its density barycenter and the stochastic student consumes
the same mixture weights over the same support.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np

from genode.artifacts.identity import semantic_sha256
from genode.gico.image_artifact_io import (
    image_gico_directory_root,
    load_float64_npy,
    staged_image_gico_directory,
    validate_image_gico_directory,
    write_float64_npy,
)
from genode.gico.image_conditional import (
    ImageGICOConditionalTargets,
    validate_image_gico_density_alias_bindings,
)
from genode.path_safety import is_link_or_reparse_point

ImageGICOStudentKind: TypeAlias = Literal["deterministic_barycenter", "stochastic_causal_ar"]
IMAGE_GICO_STUDENT_KINDS: tuple[ImageGICOStudentKind, ...] = (
    "deterministic_barycenter",
    "stochastic_causal_ar",
)
IMAGE_GICO_SUPERVISION_PROTOCOL = "image_gico_supervision_v1"
IMAGE_GICO_SUPERVISION_NAMESPACE = "image-gico-supervision-v1"
IMAGE_GICO_CONTEXT_DIM = 768
IMAGE_GICO_DENSITY_BIN_COUNT = 64
IMAGE_GICO_TARGET_NFES = (2, 4, 8)
_IDENTITY_PATTERN = re.compile(r"(?:[a-z][a-z0-9_.-]*:)?[0-9a-f]{64}\Z")


def normalize_image_gico_student_kind(value: object) -> ImageGICOStudentKind:
    if value not in IMAGE_GICO_STUDENT_KINDS:
        raise ValueError(f"student_kind must be one of {IMAGE_GICO_STUDENT_KINDS}.")
    return value  # type: ignore[return-value]


def _frozen_float64(value: object, *, field: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{field} must use a floating-point dtype.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{field} must have rank {ndim}.")
    if result.size == 0 or not bool(np.isfinite(result).all()):
        raise ValueError(f"{field} must be nonempty and finite.")
    result.setflags(write=False)
    return result


def image_gico_supervision_array_sha256(value: np.ndarray, *, field: str) -> str:
    little_endian = np.ascontiguousarray(value.astype("<f8", copy=False))
    return semantic_sha256(
        {
            "field": field,
            "dtype": "float64-le",
            "shape": list(little_endian.shape),
            "content_sha256": hashlib.sha256(little_endian.tobytes(order="C")).hexdigest(),
        },
        namespace="image-gico-supervision-array-v1",
    )


def _normalized_source_identities(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_identities must be a nonempty mapping.")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError(f"Invalid source identity key {key!r}.")
        if not isinstance(raw_value, str) or _IDENTITY_PATTERN.fullmatch(raw_value) is None:
            raise ValueError(f"source_identities[{key!r}] must be a lowercase SHA-256 identity.")
        normalized[key] = raw_value
    return MappingProxyType(dict(sorted(normalized.items())))


def _validate_probability_rows(value: np.ndarray, *, field: str) -> None:
    if bool(np.any(value < 0.0)):
        raise ValueError(f"{field} must be nonnegative.")
    totals = np.sum(value, axis=-1, dtype=np.float64)
    if not bool(np.allclose(totals, np.ones_like(totals), rtol=0.0, atol=1e-10)):
        raise ValueError(f"Every {field} row must sum to one.")


def _validate_conditional_law(
    targets: ImageGICOConditionalTargets,
    *,
    support: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
) -> None:
    if bool(np.any(np.abs(rewards) > 5.0)):
        raise ValueError("Conditional normalized rewards must be clipped to [-5, 5].")
    try:
        uniform_index = tuple(targets.schedule_keys).index("uniform")
    except ValueError as exc:
        raise ValueError("Conditional support must contain the uniform schedule.") from exc
    if not np.array_equal(rewards[:, :, uniform_index], np.zeros(rewards.shape[:2], dtype=np.float64)):
        raise ValueError("Conditional uniform rewards must be exactly zero.")

    expected_weights = np.zeros_like(weights)
    density_hashes = tuple(tuple(row) for row in targets.density_mass_sha256s)
    validate_image_gico_density_alias_bindings(density_hashes, support)
    for nfe_index, nfe_hashes in enumerate(density_hashes):
        groups_by_hash: dict[str, list[int]] = {}
        for schedule_index, density_hash in enumerate(nfe_hashes):
            groups_by_hash.setdefault(density_hash, []).append(schedule_index)
        groups = tuple(groups_by_hash.values())
        logits = np.stack(
            [np.mean(rewards[nfe_index][:, group], axis=-1, dtype=np.float64) for group in groups],
            axis=-1,
        )
        logits -= np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True, dtype=np.float64)
        for group_index, group in enumerate(groups):
            expected_weights[nfe_index][:, group] = probabilities[:, group_index, None] / float(len(group))
    if not bool(np.allclose(expected_weights, weights, rtol=0.0, atol=2e-15)):
        raise ValueError("Conditional mixture weights do not match alias-aggregated reward softmax.")


@dataclass(frozen=True, slots=True)
class ImageGICOSupervision:
    """Validated finite teacher law shared by both image GICO students."""

    supervision_kind: Literal["conditional_kid", "unconditional_mixture"]
    target_nfes: tuple[int, ...]
    schedule_keys: tuple[str, ...]
    fixed_density_mass: np.ndarray
    mixture_weights: np.ndarray
    barycenter_density_mass: np.ndarray
    normalized_contexts: np.ndarray
    normalized_rewards: np.ndarray | None
    reward_diagnostics: Mapping[str, Any]
    source_identities: Mapping[str, str]
    context_binding_sha256: str
    conditional_target_payload: Mapping[str, Any] | None = None
    _construction_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.supervision_kind not in {
            "conditional_kid",
            "unconditional_mixture",
        }:
            raise ValueError("Unsupported image GICO supervision kind.")
        nfes = tuple(self.target_nfes)
        if nfes != IMAGE_GICO_TARGET_NFES or any(
            isinstance(value, bool) or not isinstance(value, int) for value in nfes
        ):
            raise ValueError(f"target_nfes must be exactly {IMAGE_GICO_TARGET_NFES}.")
        keys = tuple(self.schedule_keys)
        if not keys or len(set(keys)) != len(keys) or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("schedule_keys must be nonempty and unique.")

        support = _frozen_float64(self.fixed_density_mass, field="fixed_density_mass", ndim=3)
        weights = _frozen_float64(self.mixture_weights, field="mixture_weights", ndim=3)
        barycenter = _frozen_float64(
            self.barycenter_density_mass,
            field="barycenter_density_mass",
            ndim=3,
        )
        contexts = _frozen_float64(self.normalized_contexts, field="normalized_contexts", ndim=2)
        if not np.array_equal(contexts, contexts.astype(np.float32).astype(np.float64)):
            raise ValueError("normalized_contexts must be exactly representable as float32 execution inputs.")
        rewards = (
            None
            if self.normalized_rewards is None
            else _frozen_float64(self.normalized_rewards, field="normalized_rewards", ndim=3)
        )
        nfe_count = len(nfes)
        schedule_count = len(keys)
        context_count = contexts.shape[0]
        if support.shape != (
            nfe_count,
            schedule_count,
            IMAGE_GICO_DENSITY_BIN_COUNT,
        ):
            raise ValueError(
                f"fixed_density_mass must have shape [{nfe_count}, {schedule_count}, {IMAGE_GICO_DENSITY_BIN_COUNT}]."
            )
        if weights.shape != (nfe_count, context_count, schedule_count):
            raise ValueError(f"mixture_weights must have shape [{nfe_count}, {context_count}, {schedule_count}].")
        if barycenter.shape != (
            nfe_count,
            context_count,
            IMAGE_GICO_DENSITY_BIN_COUNT,
        ):
            raise ValueError("barycenter_density_mass has an incompatible shape.")
        if contexts.shape[1] != IMAGE_GICO_CONTEXT_DIM:
            raise ValueError(f"normalized_contexts must have {IMAGE_GICO_CONTEXT_DIM} columns.")
        if rewards is not None and rewards.shape != weights.shape:
            raise ValueError("normalized_rewards must match mixture_weights.")
        _validate_probability_rows(support, field="fixed_density_mass")
        _validate_probability_rows(weights, field="mixture_weights")
        _validate_probability_rows(barycenter, field="barycenter_density_mass")
        expected_barycenter = np.einsum("ncs,nsb->ncb", weights, support, dtype=np.float64)
        if not bool(
            np.allclose(
                expected_barycenter,
                barycenter,
                rtol=0.0,
                atol=2e-12,
            )
        ):
            raise ValueError("barycenter_density_mass is not the mixture-weighted support barycenter.")
        if self.supervision_kind == "unconditional_mixture":
            if context_count != 1 or bool(np.any(contexts != 0.0)):
                raise ValueError("Unconditional supervision requires one explicit zero context.")
            if rewards is not None:
                raise ValueError("Unconditional precomputed mixture evidence must not fabricate rewards.")
        elif context_count != 1_000 or schedule_count != 23:
            raise ValueError("Conditional ImageNet supervision requires exactly 1,000 contexts and 23 schedules.")
        if not isinstance(self.reward_diagnostics, Mapping):
            raise TypeError("reward_diagnostics must be a mapping.")
        diagnostics = json.loads(json.dumps(dict(self.reward_diagnostics), sort_keys=True, allow_nan=False))
        sources = _normalized_source_identities(self.source_identities)
        target_payload = self.conditional_target_payload
        if self.supervision_kind == "conditional_kid":
            if not isinstance(target_payload, Mapping):
                raise ValueError("Conditional supervision requires its validated target payload.")
            reconstructed = ImageGICOConditionalTargets.from_payload(target_payload)
            if np.asarray(reconstructed.mixture_weights).shape != weights.shape or reconstructed.sha256 != sources.get(
                "conditional_target"
            ):
                raise ValueError("Conditional target payload differs from supervision.")
            if tuple(reconstructed.target_nfes) != nfes or tuple(reconstructed.schedule_keys) != keys:
                raise ValueError("Conditional target support labels differ from supervision.")
            target_arrays = (
                ("mixture_weights", reconstructed.mixture_weights, weights),
                ("density_mass", reconstructed.density_mass, barycenter),
                ("normalized_rewards", reconstructed.normalized_rewards, rewards),
            )
            for field_name, target_value, supervision_value in target_arrays:
                if supervision_value is None or not np.array_equal(
                    np.asarray(target_value, dtype=np.float64), supervision_value
                ):
                    raise ValueError(f"Conditional target {field_name} differs from supervision.")
            assert rewards is not None
            _validate_conditional_law(
                reconstructed,
                support=support,
                rewards=rewards,
                weights=weights,
            )
            target_payload = MappingProxyType(
                json.loads(json.dumps(dict(target_payload), sort_keys=True, allow_nan=False))
            )
        elif target_payload is not None:
            raise ValueError("Unconditional supervision must not carry conditional target data.")
        expected_context_binding = image_gico_supervision_array_sha256(contexts, field="normalized_contexts")
        if self.context_binding_sha256 != expected_context_binding:
            raise ValueError("context_binding_sha256 does not match normalized_contexts.")

        object.__setattr__(self, "target_nfes", nfes)
        object.__setattr__(self, "schedule_keys", keys)
        object.__setattr__(self, "fixed_density_mass", support)
        object.__setattr__(self, "mixture_weights", weights)
        object.__setattr__(self, "barycenter_density_mass", barycenter)
        object.__setattr__(self, "normalized_contexts", contexts)
        object.__setattr__(self, "normalized_rewards", rewards)
        object.__setattr__(self, "reward_diagnostics", MappingProxyType(diagnostics))
        object.__setattr__(self, "source_identities", sources)
        object.__setattr__(self, "conditional_target_payload", target_payload)
        object.__setattr__(
            self,
            "_construction_sha256",
            semantic_sha256(
                self.identity_payload(),
                namespace=IMAGE_GICO_SUPERVISION_NAMESPACE,
            ),
        )

    @property
    def context_count(self) -> int:
        return int(self.normalized_contexts.shape[0])

    @property
    def schedule_count(self) -> int:
        return len(self.schedule_keys)

    def identity_payload(self) -> dict[str, Any]:
        arrays = {
            "fixed_density_mass": image_gico_supervision_array_sha256(
                self.fixed_density_mass, field="fixed_density_mass"
            ),
            "mixture_weights": image_gico_supervision_array_sha256(self.mixture_weights, field="mixture_weights"),
            "barycenter_density_mass": image_gico_supervision_array_sha256(
                self.barycenter_density_mass, field="barycenter_density_mass"
            ),
            "normalized_contexts": self.context_binding_sha256,
            "normalized_rewards": (
                None
                if self.normalized_rewards is None
                else image_gico_supervision_array_sha256(self.normalized_rewards, field="normalized_rewards")
            ),
        }
        return {
            "protocol": IMAGE_GICO_SUPERVISION_PROTOCOL,
            "supervision_kind": self.supervision_kind,
            "target_nfes": list(self.target_nfes),
            "schedule_keys": list(self.schedule_keys),
            "context_count": self.context_count,
            "context_dim": IMAGE_GICO_CONTEXT_DIM,
            "density_bin_count": IMAGE_GICO_DENSITY_BIN_COUNT,
            "arrays": arrays,
            "reward_diagnostics": dict(self.reward_diagnostics),
            "source_identities": dict(self.source_identities),
            "conditional_target_sha256": self.source_identities.get("conditional_target"),
        }

    @property
    def sha256(self) -> str:
        return self._construction_sha256

    def verify(self) -> None:
        """Revalidate the frozen scientific law and its construction identity."""

        try:
            current = ImageGICOSupervision(
                supervision_kind=self.supervision_kind,
                target_nfes=self.target_nfes,
                schedule_keys=self.schedule_keys,
                fixed_density_mass=self.fixed_density_mass,
                mixture_weights=self.mixture_weights,
                barycenter_density_mass=self.barycenter_density_mass,
                normalized_contexts=self.normalized_contexts,
                normalized_rewards=self.normalized_rewards,
                reward_diagnostics=self.reward_diagnostics,
                source_identities=self.source_identities,
                context_binding_sha256=self.context_binding_sha256,
                conditional_target_payload=self.conditional_target_payload,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Image GICO supervision scientific law was mutated.") from exc
        if current.sha256 != self._construction_sha256:
            raise ValueError("Image GICO supervision construction identity was mutated.")


def build_image_gico_conditional_supervision(
    *,
    targets: ImageGICOConditionalTargets,
    fixed_density_mass: np.ndarray,
    normalized_contexts: np.ndarray,
) -> ImageGICOSupervision:
    """Bind the existing KID/shrinkage target construction to both students."""

    if not isinstance(targets, ImageGICOConditionalTargets):
        raise TypeError("targets must be ImageGICOConditionalTargets.")
    contexts = _frozen_float64(normalized_contexts, field="normalized_contexts", ndim=2)
    if contexts.shape[0] != len(targets.density_mass[0]):
        raise ValueError("normalized_contexts and conditional targets disagree.")
    rewards = np.asarray(targets.normalized_rewards, dtype=np.float64)
    diagnostics = {
        "construction": "kid_uniform_advantage_jackknife_hierarchical_shrinkage_v1",
        "kid_direction": "lower_is_better",
        "advantage": "uniform_kid_minus_schedule_kid",
        "reward_clip": [-5.0, 5.0],
        "conditional_target_sha256": targets.sha256,
        "normalized_reward_min": float(np.min(rewards)),
        "normalized_reward_max": float(np.max(rewards)),
    }
    sources = {
        "conditional_target": targets.sha256,
        "feature_group": targets.feature_group_sha256,
        "reward_evidence": targets.reward_evidence_sha256,
        "fixed_support": targets.fixed_support_sha256,
        "backbone_protocol": targets.backbone_protocol_sha256,
        "backbone_checkpoint": targets.backbone_checkpoint_sha256,
        "feature_protocol": targets.feature_protocol_sha256,
    }
    return ImageGICOSupervision(
        supervision_kind="conditional_kid",
        target_nfes=tuple(targets.target_nfes),
        schedule_keys=tuple(targets.schedule_keys),
        fixed_density_mass=fixed_density_mass,
        mixture_weights=np.asarray(targets.mixture_weights, dtype=np.float64),
        barycenter_density_mass=np.asarray(targets.density_mass, dtype=np.float64),
        normalized_contexts=contexts,
        normalized_rewards=rewards,
        reward_diagnostics=diagnostics,
        source_identities=sources,
        context_binding_sha256=image_gico_supervision_array_sha256(contexts, field="normalized_contexts"),
        conditional_target_payload=targets.as_payload(),
    )


def build_image_gico_unconditional_supervision(
    *,
    target_nfes: Sequence[int],
    schedule_keys: Sequence[str],
    fixed_density_mass: np.ndarray,
    mixture_weights: np.ndarray,
    source_identities: Mapping[str, str],
) -> ImageGICOSupervision:
    """Build one-context supervision from authenticated mixture evidence."""

    weights = np.asarray(mixture_weights)
    if weights.ndim == 2:
        weights = weights[:, None, :]
    if weights.ndim != 3 or weights.shape[1] != 1:
        raise ValueError("Unconditional mixture_weights must have shape [nfe, schedule] or [nfe, 1, schedule].")
    support = np.asarray(fixed_density_mass)
    barycenter = np.einsum(
        "ncs,nsb->ncb",
        weights.astype(np.float64),
        support.astype(np.float64),
        dtype=np.float64,
    )
    contexts = np.zeros((1, IMAGE_GICO_CONTEXT_DIM), dtype=np.float64)
    return ImageGICOSupervision(
        supervision_kind="unconditional_mixture",
        target_nfes=tuple(target_nfes),
        schedule_keys=tuple(schedule_keys),
        fixed_density_mass=support,
        mixture_weights=weights,
        barycenter_density_mass=barycenter,
        normalized_contexts=contexts,
        normalized_rewards=None,
        reward_diagnostics={
            "construction": "authenticated_precomputed_mixture_v1",
            "recomputed_barycenter": True,
            "synthetic_class_labels": False,
        },
        source_identities=source_identities,
        context_binding_sha256=image_gico_supervision_array_sha256(contexts, field="normalized_contexts"),
        conditional_target_payload=None,
    )


def save_image_gico_supervision(
    supervision: ImageGICOSupervision,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Publish a portable, no-overwrite supervision directory."""

    if not isinstance(supervision, ImageGICOSupervision):
        raise TypeError("supervision must be ImageGICOSupervision.")
    supervision.verify()
    expected_members = {
        "manifest.json",
        "fixed-density-mass.npy",
        "mixture-weights.npy",
        "barycenter-density-mass.npy",
        "normalized-contexts.npy",
    }
    if supervision.normalized_rewards is not None:
        expected_members.add("normalized-rewards.npy")
    with staged_image_gico_directory(
        output_dir,
        expected_members=expected_members,
        label="Image GICO supervision directory",
    ) as stage:
        arrays = {
            "fixed_density_mass": write_float64_npy(
                stage / "fixed-density-mass.npy",
                supervision.fixed_density_mass,
            ),
            "mixture_weights": write_float64_npy(
                stage / "mixture-weights.npy",
                supervision.mixture_weights,
            ),
            "barycenter_density_mass": write_float64_npy(
                stage / "barycenter-density-mass.npy",
                supervision.barycenter_density_mass,
            ),
            "normalized_contexts": write_float64_npy(
                stage / "normalized-contexts.npy",
                supervision.normalized_contexts,
            ),
        }
        if supervision.normalized_rewards is not None:
            arrays["normalized_rewards"] = write_float64_npy(
                stage / "normalized-rewards.npy",
                supervision.normalized_rewards,
            )
        manifest = {
            "artifact": "image_gico_supervision",
            **supervision.identity_payload(),
            "supervision_sha256": supervision.sha256,
            "array_files": arrays,
            "conditional_target_payload": (
                None if supervision.conditional_target_payload is None else dict(supervision.conditional_target_payload)
            ),
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return MappingProxyType(manifest)


def load_image_gico_supervision(output_dir: str | Path) -> ImageGICOSupervision:
    """Strictly load and re-hash a supervision directory."""

    root = image_gico_directory_root(output_dir, label="Image GICO supervision directory")
    manifest_path = root / "manifest.json"
    if is_link_or_reparse_point(manifest_path) or not manifest_path.is_file():
        raise ValueError("Supervision manifest must be a regular file.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "artifact",
        "protocol",
        "supervision_kind",
        "target_nfes",
        "schedule_keys",
        "context_count",
        "context_dim",
        "density_bin_count",
        "arrays",
        "reward_diagnostics",
        "source_identities",
        "conditional_target_sha256",
        "supervision_sha256",
        "array_files",
        "conditional_target_payload",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("Supervision manifest fields are incomplete or unexpected.")
    arrays = manifest.get("array_files")
    if not isinstance(arrays, Mapping):
        raise ValueError("Supervision array manifest is invalid.")
    expected_arrays = {
        "fixed_density_mass",
        "mixture_weights",
        "barycenter_density_mass",
        "normalized_contexts",
    }
    if manifest.get("arrays", {}).get("normalized_rewards") is not None:
        expected_arrays.add("normalized_rewards")
    if set(arrays) != expected_arrays:
        raise ValueError("Supervision array files are incomplete or unexpected.")
    expected_members = {"manifest.json"} | {
        str(descriptor.get("file")) for descriptor in arrays.values() if isinstance(descriptor, Mapping)
    }
    validate_image_gico_directory(
        root,
        expected_members=expected_members,
        label="Image GICO supervision directory",
    )
    loaded = {field: load_float64_npy(root, arrays[field], field=field) for field in expected_arrays}
    supervision = ImageGICOSupervision(
        supervision_kind=manifest["supervision_kind"],
        target_nfes=tuple(manifest["target_nfes"]),
        schedule_keys=tuple(manifest["schedule_keys"]),
        fixed_density_mass=loaded["fixed_density_mass"],
        mixture_weights=loaded["mixture_weights"],
        barycenter_density_mass=loaded["barycenter_density_mass"],
        normalized_contexts=loaded["normalized_contexts"],
        normalized_rewards=loaded.get("normalized_rewards"),
        reward_diagnostics=manifest["reward_diagnostics"],
        source_identities=manifest["source_identities"],
        context_binding_sha256=manifest["arrays"]["normalized_contexts"],
        conditional_target_payload=manifest.get("conditional_target_payload"),
    )
    if (
        manifest.get("artifact") != "image_gico_supervision"
        or manifest.get("protocol") != IMAGE_GICO_SUPERVISION_PROTOCOL
    ):
        raise ValueError("Unsupported image GICO supervision artifact.")
    identity_payload = supervision.identity_payload()
    if any(manifest.get(field) != value for field, value in identity_payload.items()):
        raise ValueError("Supervision manifest contradicts the loaded scientific contract.")
    if supervision.sha256 != manifest.get("supervision_sha256"):
        raise ValueError("Supervision identity changed.")
    return supervision


__all__ = [
    "IMAGE_GICO_CONTEXT_DIM",
    "IMAGE_GICO_DENSITY_BIN_COUNT",
    "IMAGE_GICO_STUDENT_KINDS",
    "IMAGE_GICO_SUPERVISION_PROTOCOL",
    "IMAGE_GICO_TARGET_NFES",
    "ImageGICOStudentKind",
    "ImageGICOSupervision",
    "build_image_gico_conditional_supervision",
    "build_image_gico_unconditional_supervision",
    "image_gico_supervision_array_sha256",
    "load_image_gico_supervision",
    "normalize_image_gico_student_kind",
    "save_image_gico_supervision",
]
