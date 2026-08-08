from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
import re
import numpy as np
import torch
from torch import Tensor, nn

from genode.artifacts.identity import semantic_sha256
from genode.benchmarks.image.protocol import IMAGE_TARGET_NFES, normalize_image_nfe
from genode.gico.image_conditional import (
    IMAGE_GICO_BACKBONE_CONTEXT_DIM,
    IMAGE_GICO_CLASS_COUNT,
    IMAGE_GICO_CONDITIONAL_HIDDEN_DIM,
    IMAGE_GICO_NFE_EMBEDDING_DIM,
    ImageGICOConditionalTargets,
    validate_image_gico_backbone_context_tensor,
)
from genode.schedule_transfer.reference_clocks import reference_clock_provenance
from genode.schedules.density import (
    density_mass_hash,
    density_mass_to_time_grid,
    reference_time_grid_hash,
    time_grid_hash,
    time_grid_to_density_mass,
    uniform_reference_time_grid,
    validate_density_mass,
    validate_reference_time_grid,
)
from genode.schedules.fixed import (
    build_fixed_schedule,
    validate_fixed_schedule_keys,
)
from genode.schedules.policy import ScheduleBatch
from genode.schedules.progress import validate_time_grid
from genode.schedules.specification import ScheduleSpecification


IMAGE_GICO_CLOCK_LIBRARY_PROTOCOL = "image_gico_complete_clock_library_v1"
IMAGE_GICO_CLOCK_MIXTURE_MODEL_PROTOCOL = "image_gico_backbone_context_clock_mixture_v1"
IMAGE_GICO_CLOCK_MIXTURE_POLICY_PROTOCOL = "image_gico_clock_mixture_policy_v1"
IMAGE_GICO_CLOCK_RNG_PROTOCOL = "image_gico_complete_clock_sha256_counter_rng_v2"
IMAGE_GICO_CLOCK_MIXTURE_POLICY_SPECIFICATION = ScheduleSpecification("image_gico_clock_mixture_v1")
_EXACT_CLOCK_ATOL = 2e-14
_SUPERVISION_RECONSTRUCTION_ATOL = 1e-8
_MAX_SEED = 2**63 - 1
_UNIFORM_MANTISSA_BITS = 52
_UNIFORM_DENOMINATOR = 2**_UNIFORM_MANTISSA_BITS + 1
_SHA256_IDENTITY = re.compile(r"(?:(?:[a-z][a-z0-9_.-]*):)?[0-9a-f]{64}\Z")


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive.")
    return parsed


def _identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 identity.")
    return value


def _schedule_key(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a fixed-schedule key.")
    try:
        reference_clock_provenance(value)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{field} must be a supported fixed-schedule key.") from exc
    return value


def _nonnegative_integer(value: object, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a nonnegative integer.")
    parsed = int(value)
    if not 0 <= parsed <= maximum:
        raise ValueError(f"{field} must be in [0, {maximum}].")
    return parsed


def image_gico_clock_sample_key(
    *,
    sampling_plan_sha256: str,
    policy_artifact_sha256: str,
    clock_library_sha256: str,
    context_binding_sha256: str,
    root_seed: int,
    latent_seed: int,
    class_label: int,
    target_nfe: int,
) -> str:
    """Bind one ImageNet clock draw; alpha is intentionally not a key input."""

    label = _nonnegative_integer(
        class_label,
        field="class_label",
        maximum=IMAGE_GICO_CLASS_COUNT - 1,
    )
    payload = {
        "protocol": IMAGE_GICO_CLOCK_RNG_PROTOCOL,
        "sampling_plan_sha256": _identity(
            sampling_plan_sha256,
            field="sampling_plan_sha256",
        ),
        "policy_artifact_sha256": _identity(
            policy_artifact_sha256,
            field="policy_artifact_sha256",
        ),
        "clock_library_sha256": _identity(
            clock_library_sha256,
            field="clock_library_sha256",
        ),
        "context_binding_sha256": _identity(
            context_binding_sha256,
            field="context_binding_sha256",
        ),
        "root_seed": _nonnegative_integer(
            root_seed,
            field="root_seed",
            maximum=_MAX_SEED,
        ),
        "latent_seed": _nonnegative_integer(
            latent_seed,
            field="latent_seed",
            maximum=_MAX_SEED,
        ),
        "class_label": label,
        "solver_key": "euler",
        "target_nfe": normalize_image_nfe(target_nfe),
    }
    return semantic_sha256(
        payload,
        namespace="image-gico-complete-clock-sample-key-v2",
    )


def _open_unit_interval_float64(mantissa: int) -> float:
    maximum = 2**_UNIFORM_MANTISSA_BITS - 1
    parsed = _nonnegative_integer(
        mantissa,
        field="mantissa",
        maximum=maximum,
    )
    value = float(parsed + 1) / float(_UNIFORM_DENOMINATOR)
    if not 0.0 < value < 1.0:
        raise RuntimeError("Integer conversion left the open unit interval.")
    return value


def image_gico_clock_uniform(
    sample_key: str,
    *,
    draw_index: int = 0,
) -> float:
    """Map a content-bound sample key to one open-interval float64 draw."""

    key = _identity(sample_key, field="sample_key")
    counter = _nonnegative_integer(
        draw_index,
        field="draw_index",
        maximum=2**64 - 1,
    )
    digest = hashlib.sha256()
    digest.update(IMAGE_GICO_CLOCK_RNG_PROTOCOL.encode("ascii"))
    digest.update(b"\0")
    digest.update(key.encode("ascii"))
    digest.update(b"\0")
    digest.update(counter.to_bytes(8, byteorder="big", signed=False))
    mantissa = (
        int.from_bytes(digest.digest()[:8], byteorder="big")
        >> (64 - _UNIFORM_MANTISSA_BITS)
    )
    return _open_unit_interval_float64(mantissa)


def derive_image_gico_clock_uniforms(
    sample_keys: Sequence[str],
    *,
    draw_index: int = 0,
) -> Tensor:
    """Return contiguous CPU float64 draws without touching global RNG state."""

    keys = tuple(sample_keys)
    if not keys:
        raise ValueError("sample_keys must be nonempty.")
    values = tuple(
        image_gico_clock_uniform(key, draw_index=draw_index) for key in keys
    )
    return torch.tensor(values, dtype=torch.float64, device="cpu").contiguous()


def _tensor_payload(tensor: Tensor) -> dict[str, object]:
    if not isinstance(tensor, Tensor) or tensor.layout != torch.strided:
        raise TypeError("Identity tensors must use a strided torch layout.")
    values = tensor.detach().to(device="cpu").contiguous().numpy()
    return {
        "dtype": str(values.dtype),
        "shape": [int(value) for value in values.shape],
        "content_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def image_clock_mixture_module_state_sha256(module: nn.Module) -> str:
    """Return a semantic identity for the trainable clock-mixture state."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module.")
    payload = {
        "parameters": {
            name: _tensor_payload(tensor)
            for name, tensor in sorted(module.named_parameters())
        },
        "execution_buffers": {
            name: _tensor_payload(tensor)
            for name, tensor in sorted(module.named_buffers())
        },
    }
    return semantic_sha256(
        payload,
        namespace="image-gico-clock-mixture-model-state-v1",
    )


def image_clock_mixture_serialized_state_sha256(
    state: Mapping[str, Tensor],
) -> str:
    """Return a context-independent identity for the serialized state dict."""

    if not isinstance(state, Mapping) or not state:
        raise TypeError("state must be a nonempty tensor mapping.")
    payload: dict[str, object] = {}
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise TypeError("Serialized state names must be nonempty strings.")
        if not isinstance(tensor, Tensor):
            raise TypeError("Serialized state values must be torch tensors.")
        payload[name] = _tensor_payload(tensor)
    return semantic_sha256(
        payload,
        namespace="image-gico-clock-mixture-serialized-state-v1",
    )


@dataclass(frozen=True, slots=True)
class ImageGICOClockGroup:
    """One supervision-equivalent group of complete schedule realizations."""

    target_nfe: int
    group_index: int
    density_mass_sha256: str
    member_indices: tuple[int, ...]
    member_schedule_keys: tuple[str, ...]
    representative_index: int
    representative_schedule_key: str
    exact_density_mass_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        nfe = normalize_image_nfe(self.target_nfe)
        if isinstance(self.group_index, bool) or not isinstance(
            self.group_index,
            Integral,
        ):
            raise TypeError("group_index must be a nonnegative integer.")
        group_index = int(self.group_index)
        if group_index < 0:
            raise ValueError("group_index must be nonnegative.")
        density_identity = _identity(
            self.density_mass_sha256,
            field="density_mass_sha256",
        )
        indices = tuple(self.member_indices)
        if not indices:
            raise ValueError("A clock group must contain at least one member.")
        normalized_indices: list[int] = []
        for value in indices:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("Clock-group member indices must be integers.")
            index = int(value)
            if index < 0:
                raise ValueError("Clock-group member indices must be nonnegative.")
            normalized_indices.append(index)
        if tuple(normalized_indices) != tuple(sorted(set(normalized_indices))):
            raise ValueError("Clock-group member indices must be unique and increasing.")
        keys = tuple(
            _schedule_key(value, field="member_schedule_keys")
            for value in self.member_schedule_keys
        )
        exact_hashes = tuple(
            _identity(value, field="exact_density_mass_sha256s") for value in self.exact_density_mass_sha256s
        )
        if len(keys) != len(normalized_indices) or len(exact_hashes) != len(keys):
            raise ValueError("Clock-group indices, keys, and exact density identities must align.")
        if self.representative_index != normalized_indices[0]:
            raise ValueError("The first group member must be the representative.")
        if self.representative_schedule_key != keys[0]:
            raise ValueError("The representative key must be the first member key.")
        object.__setattr__(self, "target_nfe", nfe)
        object.__setattr__(self, "group_index", group_index)
        object.__setattr__(self, "density_mass_sha256", density_identity)
        object.__setattr__(self, "member_indices", tuple(normalized_indices))
        object.__setattr__(self, "member_schedule_keys", keys)
        object.__setattr__(self, "exact_density_mass_sha256s", exact_hashes)

    @property
    def member_count(self) -> int:
        return len(self.member_indices)

    def as_payload(self) -> dict[str, object]:
        return {
            "target_nfe": self.target_nfe,
            "group_index": self.group_index,
            "density_mass_sha256": self.density_mass_sha256,
            "member_indices": list(self.member_indices),
            "member_schedule_keys": list(self.member_schedule_keys),
            "representative_index": self.representative_index,
            "representative_schedule_key": self.representative_schedule_key,
            "exact_density_mass_sha256s": list(self.exact_density_mass_sha256s),
        }


@dataclass(frozen=True, slots=True)
class ImageGICOClockLibrary:
    """Canonical complete clocks on one lossless global reference grid.

    ``density_mass`` is the exact, refined representation used for inference.
    ``supervision_density_mass`` preserves the finite-bin representation that
    produced the conditional targets and determines duplicate groups.
    """

    target_nfes: tuple[int, ...]
    schedule_keys: tuple[str, ...]
    schedule_sha256s: tuple[str, ...]
    reference_time_grid: Tensor
    density_mass: Tensor
    time_grids: tuple[Tensor, ...]
    exact_density_mass_sha256s: tuple[tuple[str, ...], ...]
    supervision_reference_time_grid: Tensor
    supervision_density_mass: Tensor
    supervision_density_mass_sha256s: tuple[tuple[str, ...], ...]
    groups: tuple[tuple[ImageGICOClockGroup, ...], ...]
    target_sha256: str
    fixed_support_sha256: str

    def __post_init__(self) -> None:
        target_nfes = tuple(self.target_nfes)
        if target_nfes != tuple(IMAGE_TARGET_NFES):
            raise ValueError(f"target_nfes must be exactly {IMAGE_TARGET_NFES}.")
        keys = validate_fixed_schedule_keys(self.schedule_keys)
        schedule_count = len(keys)
        schedule_hashes = tuple(_identity(value, field="schedule_sha256s") for value in self.schedule_sha256s)
        if len(schedule_hashes) != schedule_count:
            raise ValueError("schedule_sha256s must align with schedule_keys.")

        reference = validate_reference_time_grid(self.reference_time_grid)
        supervision_reference = validate_reference_time_grid(self.supervision_reference_time_grid)
        for field, tensor in (
            ("reference_time_grid", reference),
            ("supervision_reference_time_grid", supervision_reference),
        ):
            if tensor.device.type != "cpu" or tensor.dtype != torch.float64:
                raise TypeError(f"{field} must be a CPU torch.float64 tensor.")

        if not isinstance(self.density_mass, Tensor):
            raise TypeError("density_mass must be a torch.Tensor.")
        if self.density_mass.ndim != 3 or int(self.density_mass.shape[-1]) <= 0:
            raise ValueError("density_mass must have shape [NFE, schedule, bin].")
        exact_mass = validate_density_mass(
            self.density_mass.reshape(-1, int(self.density_mass.shape[-1])),
            reference_time_grid=reference,
        ).reshape(self.density_mass.shape)
        if not isinstance(self.supervision_density_mass, Tensor):
            raise TypeError("supervision_density_mass must be a torch.Tensor.")
        if self.supervision_density_mass.ndim != 3 or int(self.supervision_density_mass.shape[-1]) <= 0:
            raise ValueError("supervision_density_mass must have shape [NFE, schedule, bin].")
        supervision_mass = validate_density_mass(
            self.supervision_density_mass.reshape(
                -1,
                int(self.supervision_density_mass.shape[-1]),
            ),
            reference_time_grid=supervision_reference,
        ).reshape(self.supervision_density_mass.shape)
        if exact_mass.device.type != "cpu" or exact_mass.dtype != torch.float64:
            raise TypeError("density_mass must be a CPU torch.float64 tensor.")
        if supervision_mass.device.type != "cpu" or supervision_mass.dtype != torch.float64:
            raise TypeError("supervision_density_mass must be a CPU torch.float64 tensor.")
        expected_exact_shape = (
            len(target_nfes),
            schedule_count,
            int(reference.numel()) - 1,
        )
        expected_supervision_shape = (
            len(target_nfes),
            schedule_count,
            int(supervision_reference.numel()) - 1,
        )
        if tuple(exact_mass.shape) != expected_exact_shape:
            raise ValueError(f"density_mass must have shape {expected_exact_shape}.")
        if tuple(supervision_mass.shape) != expected_supervision_shape:
            raise ValueError("supervision_density_mass has an incompatible shape.")

        expected_uniform = uniform_reference_time_grid(
            int(supervision_mass.shape[-1]),
            dtype=torch.float64,
            device="cpu",
        )
        if not torch.equal(supervision_reference, expected_uniform):
            raise ValueError("The supervision reference grid must be the declared uniform grid.")

        time_grids = tuple(self.time_grids)
        if len(time_grids) != len(target_nfes):
            raise ValueError("time_grids must contain one table per target NFE.")
        union_nodes: set[float] = set()
        normalized_time_grids: list[Tensor] = []
        for nfe_index, target_nfe in enumerate(target_nfes):
            grids = validate_time_grid(
                time_grids[nfe_index],
                target_nfe=target_nfe,
                batch_size=schedule_count,
            )
            if grids.device.type != "cpu" or grids.dtype != torch.float64:
                raise TypeError("time_grids must use CPU torch.float64 tensors.")
            normalized_time_grids.append(grids.contiguous())
            union_nodes.update(float(value) for value in grids.reshape(-1).tolist())
            reconstructed = density_mass_to_time_grid(
                exact_mass[nfe_index],
                target_nfe=target_nfe,
                reference_time_grid=reference,
            )
            if not torch.allclose(
                reconstructed,
                grids,
                rtol=0.0,
                atol=_EXACT_CLOCK_ATOL,
            ):
                maximum_error = float(torch.max(torch.abs(reconstructed - grids)))
                raise ValueError(
                    "Exact clock densities do not reconstruct their complete grids; "
                    f"NFE={target_nfe}, max_error={maximum_error:.3e}."
                )
        expected_union = torch.tensor(
            sorted(union_nodes),
            dtype=torch.float64,
            device="cpu",
        )
        if not torch.equal(reference, expected_union):
            raise ValueError("reference_time_grid must be the exact union of every clock node at NFE 2, 4, and 8.")

        exact_hashes = tuple(tuple(row) for row in self.exact_density_mass_sha256s)
        supervision_hashes = tuple(tuple(row) for row in self.supervision_density_mass_sha256s)
        expected_hash_shape = (len(target_nfes), schedule_count)
        if (
            len(exact_hashes) != expected_hash_shape[0]
            or any(len(row) != expected_hash_shape[1] for row in exact_hashes)
            or len(supervision_hashes) != expected_hash_shape[0]
            or any(len(row) != expected_hash_shape[1] for row in supervision_hashes)
        ):
            raise ValueError("Density identity tables have an incompatible shape.")
        for nfe_index in range(len(target_nfes)):
            for schedule_index in range(schedule_count):
                observed_exact = density_mass_hash(
                    exact_mass[nfe_index, schedule_index],
                    reference_time_grid=reference,
                )
                observed_supervision = density_mass_hash(
                    supervision_mass[nfe_index, schedule_index],
                    reference_time_grid=supervision_reference,
                )
                if exact_hashes[nfe_index][schedule_index] != observed_exact:
                    raise ValueError("An exact density identity is inconsistent.")
                if supervision_hashes[nfe_index][schedule_index] != observed_supervision:
                    raise ValueError("A supervision density identity is inconsistent.")

        groups = tuple(tuple(row) for row in self.groups)
        if len(groups) != len(target_nfes):
            raise ValueError("groups must contain one row per target NFE.")
        for nfe_index, target_nfe in enumerate(target_nfes):
            observed_indices: list[int] = []
            for group_index, group in enumerate(groups[nfe_index]):
                if not isinstance(group, ImageGICOClockGroup):
                    raise TypeError("groups must contain ImageGICOClockGroup values.")
                if group.target_nfe != target_nfe or group.group_index != group_index:
                    raise ValueError("Clock groups must be in canonical NFE/group order.")
                expected_keys = tuple(keys[index] for index in group.member_indices)
                expected_exact_hashes = tuple(exact_hashes[nfe_index][index] for index in group.member_indices)
                if (
                    group.member_schedule_keys != expected_keys
                    or group.exact_density_mass_sha256s != expected_exact_hashes
                ):
                    raise ValueError("Clock-group members do not match the library.")
                if any(
                    supervision_hashes[nfe_index][index] != group.density_mass_sha256 for index in group.member_indices
                ):
                    raise ValueError("Clock groups must contain one supervision density identity.")
                observed_indices.extend(group.member_indices)
            if tuple(sorted(observed_indices)) != tuple(range(schedule_count)):
                raise ValueError("Clock groups must partition every schedule exactly once.")

        object.__setattr__(self, "target_nfes", target_nfes)
        object.__setattr__(self, "schedule_keys", keys)
        object.__setattr__(self, "schedule_sha256s", schedule_hashes)
        object.__setattr__(self, "reference_time_grid", reference.contiguous())
        object.__setattr__(self, "density_mass", exact_mass.contiguous())
        object.__setattr__(self, "time_grids", tuple(normalized_time_grids))
        object.__setattr__(self, "exact_density_mass_sha256s", exact_hashes)
        object.__setattr__(
            self,
            "supervision_reference_time_grid",
            supervision_reference.contiguous(),
        )
        object.__setattr__(
            self,
            "supervision_density_mass",
            supervision_mass.contiguous(),
        )
        object.__setattr__(
            self,
            "supervision_density_mass_sha256s",
            supervision_hashes,
        )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(
            self,
            "target_sha256",
            _identity(self.target_sha256, field="target_sha256"),
        )
        object.__setattr__(
            self,
            "fixed_support_sha256",
            _identity(self.fixed_support_sha256, field="fixed_support_sha256"),
        )

    @property
    def schedule_count(self) -> int:
        return len(self.schedule_keys)

    @property
    def density_bin_count(self) -> int:
        return int(self.density_mass.shape[-1])

    @property
    def group_counts(self) -> tuple[int, ...]:
        return tuple(len(row) for row in self.groups)

    @property
    def maximum_group_count(self) -> int:
        return max(self.group_counts)

    def nfe_index(self, target_nfe: object) -> int:
        return self.target_nfes.index(normalize_image_nfe(target_nfe))

    def density_table(
        self,
        target_nfe: object,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype.")
        return self.density_mass[self.nfe_index(target_nfe)].to(
            dtype=dtype,
            device=device,
        )

    def supervision_density_table(
        self,
        target_nfe: object,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype.")
        return self.supervision_density_mass[self.nfe_index(target_nfe)].to(
            dtype=dtype,
            device=device,
        )

    def validate_targets(self, targets: ImageGICOConditionalTargets) -> None:
        """Verify the declared target barycenters and group equivalence."""

        if not isinstance(targets, ImageGICOConditionalTargets):
            raise TypeError("targets must be ImageGICOConditionalTargets.")
        if targets.sha256 != self.target_sha256:
            raise ValueError("Conditional targets do not match the clock library.")
        if (
            targets.target_nfes != self.target_nfes
            or targets.schedule_keys != self.schedule_keys
            or targets.schedule_sha256s != self.schedule_sha256s
            or targets.density_mass_sha256s != self.supervision_density_mass_sha256s
            or targets.fixed_support_sha256 != self.fixed_support_sha256
        ):
            raise ValueError("Conditional target support does not match the library.")
        if targets.density_bin_count != int(self.supervision_density_mass.shape[-1]):
            raise ValueError("Conditional target density dimension is inconsistent.")
        weights = torch.tensor(targets.mixture_weights, dtype=torch.float64)
        target_density = torch.tensor(targets.density_mass, dtype=torch.float64)
        reconstructed = torch.einsum(
            "ncs,nsb->ncb",
            weights,
            self.supervision_density_mass,
        )
        if not torch.allclose(
            reconstructed,
            target_density,
            rtol=_SUPERVISION_RECONSTRUCTION_ATOL,
            atol=_SUPERVISION_RECONSTRUCTION_ATOL,
        ):
            maximum_error = float(torch.max(torch.abs(reconstructed - target_density)))
            raise ValueError(
                f"Conditional density targets are not the declared clock barycenters; max_error={maximum_error:.3e}."
            )
        for nfe_index, groups in enumerate(self.groups):
            for group in groups:
                if group.member_count <= 1:
                    continue
                member_weights = weights[
                    nfe_index,
                    :,
                    list(group.member_indices),
                ]
                if not torch.allclose(
                    member_weights,
                    member_weights[:, :1].expand_as(member_weights),
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError("Supervision-equivalent clocks must split group mass uniformly.")

    def identity_payload(self) -> dict[str, object]:
        exact_hashes = tuple(
            tuple(
                density_mass_hash(
                    row,
                    reference_time_grid=self.reference_time_grid,
                )
                for row in nfe_rows
            )
            for nfe_rows in self.density_mass
        )
        supervision_hashes = tuple(
            tuple(
                density_mass_hash(
                    row,
                    reference_time_grid=self.supervision_reference_time_grid,
                )
                for row in nfe_rows
            )
            for nfe_rows in self.supervision_density_mass
        )
        return {
            "protocol": IMAGE_GICO_CLOCK_LIBRARY_PROTOCOL,
            "target_nfes": list(self.target_nfes),
            "schedule_keys": list(self.schedule_keys),
            "schedule_sha256s": list(self.schedule_sha256s),
            "reference_time_grid_sha256": reference_time_grid_hash(self.reference_time_grid),
            "time_grid_sha256s": [[time_grid_hash(row) for row in table] for table in self.time_grids],
            "exact_density_mass_sha256s": [list(row) for row in exact_hashes],
            "supervision_reference_time_grid_sha256": reference_time_grid_hash(self.supervision_reference_time_grid),
            "supervision_density_mass_sha256s": [
                list(row) for row in supervision_hashes
            ],
            "groups": [[group.as_payload() for group in row] for row in self.groups],
            "target_sha256": self.target_sha256,
            "fixed_support_sha256": self.fixed_support_sha256,
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.identity_payload(),
            namespace="image-gico-complete-clock-library-v1",
        )


def build_image_gico_clock_library(
    targets: ImageGICOConditionalTargets,
) -> ImageGICOClockLibrary:
    """Build and verify the target-bound canonical complete-clock library."""

    if not isinstance(targets, ImageGICOConditionalTargets):
        raise TypeError("targets must be ImageGICOConditionalTargets.")
    keys = validate_fixed_schedule_keys(targets.schedule_keys)
    specifications = tuple(ScheduleSpecification(key) for key in keys)
    time_grid_rows: list[Tensor] = []
    observed_schedule_sha256s: tuple[str, ...] | None = None
    for target_nfe in IMAGE_TARGET_NFES:
        schedules = tuple(
            build_fixed_schedule(
                specification,
                target_nfe,
                density_bin_count=targets.density_bin_count,
                dtype=torch.float64,
                device="cpu",
            )
            for specification in specifications
        )
        grids = torch.stack([schedule.time_grid for schedule in schedules])
        nfe_schedule_sha256s = tuple(
            schedule.specification.sha256 for schedule in schedules
        )
        if observed_schedule_sha256s is None:
            observed_schedule_sha256s = nfe_schedule_sha256s
        elif nfe_schedule_sha256s != observed_schedule_sha256s:
            raise RuntimeError(
                "Canonical schedule-specification identities changed across NFEs."
            )
        time_grid_rows.append(grids)
    if observed_schedule_sha256s != targets.schedule_sha256s:
        raise ValueError(
            "Conditional target schedule identities do not match the current "
            "canonical reference-clock specifications."
        )
    return build_image_gico_clock_library_from_time_grids(
        targets,
        time_grid_rows,
    )


def build_image_gico_clock_library_from_time_grids(
    targets: ImageGICOConditionalTargets,
    time_grids: Sequence[Tensor],
) -> ImageGICOClockLibrary:
    """Bind persisted complete grids without rebuilding them from current code."""

    if not isinstance(targets, ImageGICOConditionalTargets):
        raise TypeError("targets must be an ImageGICOConditionalTargets.")
    keys = validate_fixed_schedule_keys(targets.schedule_keys)
    rows = tuple(time_grids)
    if len(rows) != len(IMAGE_TARGET_NFES):
        raise ValueError(
            "time_grids must contain one complete table for each target NFE."
        )
    normalized_rows: list[Tensor] = []
    union_nodes: set[float] = set()
    for target_nfe, value in zip(IMAGE_TARGET_NFES, rows, strict=True):
        grids = validate_time_grid(
            value,
            target_nfe=target_nfe,
            batch_size=len(keys),
        )
        if grids.device.type != "cpu" or grids.dtype != torch.float64:
            raise TypeError("time_grids must use CPU torch.float64 tensors.")
        normalized = grids.contiguous()
        normalized_rows.append(normalized)
        union_nodes.update(float(node) for node in normalized.reshape(-1).tolist())

    supervision_reference = uniform_reference_time_grid(
        targets.density_bin_count,
        dtype=torch.float64,
        device="cpu",
    )
    supervision_mass = torch.stack(
        tuple(
            time_grid_to_density_mass(
                grids,
                reference_time_grid=supervision_reference,
            )
            for grids in normalized_rows
        )
    )
    supervision_hashes = tuple(
        tuple(
            density_mass_hash(
                row,
                reference_time_grid=supervision_reference,
            )
            for row in nfe_rows
        )
        for nfe_rows in supervision_mass
    )
    if supervision_hashes != targets.density_mass_sha256s:
        raise ValueError(
            "Persisted complete grids do not match the conditional target "
            "supervision-density identities."
        )

    reference = torch.tensor(
        sorted(union_nodes),
        dtype=torch.float64,
        device="cpu",
    )
    validate_reference_time_grid(reference)
    exact_mass = torch.stack(
        tuple(
            time_grid_to_density_mass(grids, reference_time_grid=reference)
            for grids in normalized_rows
        )
    )
    exact_hashes = tuple(
        tuple(
            density_mass_hash(row, reference_time_grid=reference)
            for row in nfe_rows
        )
        for nfe_rows in exact_mass
    )

    group_rows: list[tuple[ImageGICOClockGroup, ...]] = []
    for nfe_index, target_nfe in enumerate(IMAGE_TARGET_NFES):
        members_by_hash: dict[str, list[int]] = {}
        for schedule_index, density_identity in enumerate(targets.density_mass_sha256s[nfe_index]):
            members_by_hash.setdefault(density_identity, []).append(schedule_index)
        groups = tuple(
            ImageGICOClockGroup(
                target_nfe=target_nfe,
                group_index=group_index,
                density_mass_sha256=density_identity,
                member_indices=tuple(member_indices),
                member_schedule_keys=tuple(keys[index] for index in member_indices),
                representative_index=member_indices[0],
                representative_schedule_key=keys[member_indices[0]],
                exact_density_mass_sha256s=tuple(exact_hashes[nfe_index][index] for index in member_indices),
            )
            for group_index, (density_identity, member_indices) in enumerate(members_by_hash.items())
        )
        group_rows.append(groups)

    library = ImageGICOClockLibrary(
        target_nfes=tuple(IMAGE_TARGET_NFES),
        schedule_keys=keys,
        schedule_sha256s=targets.schedule_sha256s,
        reference_time_grid=reference,
        density_mass=exact_mass,
        time_grids=tuple(normalized_rows),
        exact_density_mass_sha256s=exact_hashes,
        supervision_reference_time_grid=supervision_reference,
        supervision_density_mass=supervision_mass,
        supervision_density_mass_sha256s=supervision_hashes,
        groups=tuple(group_rows),
        target_sha256=targets.sha256,
        fixed_support_sha256=targets.fixed_support_sha256,
    )
    library.validate_targets(targets)
    return library


@dataclass(frozen=True, slots=True)
class ImageGICOClockMixtureModelConfig:
    schedule_count: int
    clock_library_sha256: str
    group_counts: tuple[int, ...]
    context_dim: int = IMAGE_GICO_BACKBONE_CONTEXT_DIM
    class_count: int = IMAGE_GICO_CLASS_COUNT
    nfe_embedding_dim: int = IMAGE_GICO_NFE_EMBEDDING_DIM
    hidden_dim: int = IMAGE_GICO_CONDITIONAL_HIDDEN_DIM
    target_nfes: tuple[int, ...] = tuple(IMAGE_TARGET_NFES)

    def __post_init__(self) -> None:
        for field in (
            "schedule_count",
            "context_dim",
            "class_count",
            "nfe_embedding_dim",
            "hidden_dim",
        ):
            object.__setattr__(
                self,
                field,
                _positive_integer(getattr(self, field), field=field),
            )
        if self.context_dim != IMAGE_GICO_BACKBONE_CONTEXT_DIM:
            raise ValueError("ImageNet GICO requires 768-dimensional contexts.")
        if self.class_count != IMAGE_GICO_CLASS_COUNT:
            raise ValueError("Conditional ImageNet GICO requires 1,000 classes.")
        if tuple(self.target_nfes) != tuple(IMAGE_TARGET_NFES):
            raise ValueError(f"target_nfes must be exactly {IMAGE_TARGET_NFES}.")
        counts = tuple(_positive_integer(value, field="group_counts") for value in self.group_counts)
        if len(counts) != len(self.target_nfes) or any(value > self.schedule_count for value in counts):
            raise ValueError("group_counts are incompatible with the clock support.")
        object.__setattr__(self, "group_counts", counts)
        object.__setattr__(
            self,
            "clock_library_sha256",
            _identity(self.clock_library_sha256, field="clock_library_sha256"),
        )
        object.__setattr__(self, "target_nfes", tuple(self.target_nfes))

    @classmethod
    def for_library(
        cls,
        library: ImageGICOClockLibrary,
        *,
        nfe_embedding_dim: int = IMAGE_GICO_NFE_EMBEDDING_DIM,
        hidden_dim: int = IMAGE_GICO_CONDITIONAL_HIDDEN_DIM,
    ) -> ImageGICOClockMixtureModelConfig:
        if not isinstance(library, ImageGICOClockLibrary):
            raise TypeError("library must be an ImageGICOClockLibrary.")
        return cls(
            schedule_count=library.schedule_count,
            clock_library_sha256=library.sha256,
            group_counts=library.group_counts,
            nfe_embedding_dim=nfe_embedding_dim,
            hidden_dim=hidden_dim,
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "protocol": IMAGE_GICO_CLOCK_MIXTURE_MODEL_PROTOCOL,
            "conditioning": "normalized_frozen_backbone_map_label_plus_target_nfe",
            "schedule_count": self.schedule_count,
            "clock_library_sha256": self.clock_library_sha256,
            "group_counts": list(self.group_counts),
            "context_dim": self.context_dim,
            "class_count": self.class_count,
            "nfe_embedding_dim": self.nfe_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "target_nfes": list(self.target_nfes),
            "global_base": "learned_per_nfe_schedule_logits",
            "residual_centering": "canonical_1000_context_table_per_nfe",
            "duplicate_handling": "group_mean_logit_softmax_uniform_member_split",
            "residual_initialization": "zero_output_layer",
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            self.as_payload(),
            namespace="image-gico-clock-mixture-model-config-v1",
        )


class ImageGICOBackboneContextClockMixtureModel(nn.Module):
    """Conditional complete-clock logits with centered context residuals."""

    def __init__(
        self,
        config: ImageGICOClockMixtureModelConfig,
        canonical_context_table: np.ndarray | Tensor,
        library: ImageGICOClockLibrary,
    ) -> None:
        super().__init__()
        if not isinstance(config, ImageGICOClockMixtureModelConfig):
            raise TypeError("config must be ImageGICOClockMixtureModelConfig.")
        if not isinstance(library, ImageGICOClockLibrary):
            raise TypeError("library must be an ImageGICOClockLibrary.")
        if (
            config.schedule_count != library.schedule_count
            or config.clock_library_sha256 != library.sha256
            or config.group_counts != library.group_counts
            or config.target_nfes != library.target_nfes
        ):
            raise ValueError("Model config and clock library are inconsistent.")
        self.config = config
        self.library = library
        context_table = validate_image_gico_backbone_context_tensor(
            canonical_context_table,
            field="canonical_context_table",
            expected_rows=config.class_count,
        )
        self.register_buffer(
            "_canonical_context_table",
            context_table,
            persistent=False,
        )
        self.register_buffer(
            "_clock_density_mass",
            library.density_mass.to(dtype=torch.float32),
            persistent=False,
        )

        maximum_groups = library.maximum_group_count
        group_average = torch.zeros(
            (len(config.target_nfes), maximum_groups, config.schedule_count),
            dtype=torch.float32,
        )
        group_expand = torch.zeros(
            (len(config.target_nfes), config.schedule_count, maximum_groups),
            dtype=torch.float32,
        )
        group_active = torch.zeros(
            (len(config.target_nfes), maximum_groups),
            dtype=torch.bool,
        )
        for nfe_index, groups in enumerate(library.groups):
            for group in groups:
                member_count = float(group.member_count)
                group_average[
                    nfe_index,
                    group.group_index,
                    list(group.member_indices),
                ] = 1.0 / member_count
                group_expand[
                    nfe_index,
                    list(group.member_indices),
                    group.group_index,
                ] = 1.0 / member_count
                group_active[nfe_index, group.group_index] = True
        self.register_buffer("_group_average", group_average, persistent=False)
        self.register_buffer("_group_expand", group_expand, persistent=False)
        self.register_buffer("_group_active", group_active, persistent=False)

        self.nfe_embedding = nn.Embedding(
            len(config.target_nfes),
            config.nfe_embedding_dim,
        )
        self.context_network = nn.Sequential(
            nn.Linear(
                config.context_dim + config.nfe_embedding_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.schedule_count),
        )
        self.global_logits_by_nfe = nn.Parameter(torch.zeros(len(config.target_nfes), config.schedule_count))
        final = self.context_network[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("The context network must end in a linear layer.")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def canonical_context_table(self) -> Tensor:
        return self._canonical_context_table

    @property
    def state_sha256(self) -> str:
        return image_clock_mixture_module_state_sha256(self)

    def _validate_contexts(self, contexts: Tensor) -> Tensor:
        if not isinstance(contexts, Tensor):
            raise TypeError("contexts must be a torch.Tensor.")
        if contexts.dtype != torch.float32:
            raise TypeError("contexts must use torch.float32.")
        if contexts.device != self.global_logits_by_nfe.device:
            raise ValueError("contexts and model must share a device.")
        if contexts.ndim != 2 or contexts.shape[1] != self.config.context_dim:
            raise ValueError(f"contexts must have shape [batch, {self.config.context_dim}].")
        if contexts.shape[0] <= 0 or not bool(torch.isfinite(contexts).all()):
            raise ValueError("contexts must be nonempty and finite.")
        return contexts

    def _nfe_indices(self, target_nfes: Tensor) -> Tensor:
        if not isinstance(target_nfes, Tensor):
            raise TypeError("target_nfes must be a torch.Tensor.")
        if target_nfes.ndim != 1 or target_nfes.numel() <= 0:
            raise ValueError("target_nfes must have shape [batch].")
        if (
            target_nfes.dtype == torch.bool
            or target_nfes.is_floating_point()
            or target_nfes.is_complex()
        ):
            raise TypeError("target_nfes must use an integer dtype.")
        if target_nfes.device != self.global_logits_by_nfe.device:
            raise ValueError("target_nfes and model must share a device.")
        supported = torch.tensor(
            self.config.target_nfes,
            dtype=target_nfes.dtype,
            device=target_nfes.device,
        )
        matches = target_nfes[:, None] == supported[None, :]
        if not bool(torch.all(matches.sum(dim=1) == 1)):
            raise ValueError(f"target_nfes must contain only {self.config.target_nfes}.")
        return torch.argmax(matches.to(dtype=torch.int64), dim=1)

    def _raw_residual(self, contexts: Tensor, nfe_indices: Tensor) -> Tensor:
        conditioning = torch.cat(
            (contexts, self.nfe_embedding(nfe_indices)),
            dim=-1,
        )
        return self.context_network(conditioning)

    def _residual_centers(self) -> Tensor:
        centers = []
        for nfe_index in range(len(self.config.target_nfes)):
            indices = torch.full(
                (self.config.class_count,),
                nfe_index,
                dtype=torch.int64,
                device=self.global_logits_by_nfe.device,
            )
            centers.append(self._raw_residual(self.canonical_context_table, indices).mean(dim=0))
        return torch.stack(centers)

    def centered_residual_table(self) -> Tensor:
        rows = []
        for nfe_index in range(len(self.config.target_nfes)):
            indices = torch.full(
                (self.config.class_count,),
                nfe_index,
                dtype=torch.int64,
                device=self.global_logits_by_nfe.device,
            )
            residual = self._raw_residual(self.canonical_context_table, indices)
            rows.append(residual - residual.mean(dim=0, keepdim=True))
        return torch.stack(rows)

    def schedule_logits(self, contexts: Tensor, target_nfes: Tensor) -> Tensor:
        validated_contexts = self._validate_contexts(contexts)
        indices = self._nfe_indices(target_nfes)
        if validated_contexts.shape[0] != indices.shape[0]:
            raise ValueError("contexts and target_nfes must share a batch size.")
        residual = self._raw_residual(validated_contexts, indices) - self._residual_centers()[indices]
        return self.global_logits_by_nfe[indices] + residual

    def _group_logits_from_schedule_logits(
        self,
        schedule_logits: Tensor,
        nfe_indices: Tensor,
    ) -> Tensor:
        group_logits = torch.bmm(
            self._group_average[nfe_indices],
            schedule_logits.unsqueeze(-1),
        ).squeeze(-1)
        return group_logits.masked_fill(
            ~self._group_active[nfe_indices],
            -torch.inf,
        )

    def group_logits(self, contexts: Tensor, target_nfes: Tensor) -> Tensor:
        indices = self._nfe_indices(target_nfes)
        logits = self.schedule_logits(contexts, target_nfes)
        return self._group_logits_from_schedule_logits(logits, indices)

    def group_probabilities(self, contexts: Tensor, target_nfes: Tensor) -> Tensor:
        return torch.softmax(self.group_logits(contexts, target_nfes), dim=-1)

    def schedule_probabilities(
        self,
        contexts: Tensor,
        target_nfes: Tensor,
    ) -> Tensor:
        indices = self._nfe_indices(target_nfes)
        probabilities = self.group_probabilities(contexts, target_nfes)
        return torch.bmm(
            self._group_expand[indices],
            probabilities.unsqueeze(-1),
        ).squeeze(-1)

    def canonical_schedule_logit_table(self) -> Tensor:
        return self.global_logits_by_nfe[:, None, :] + self.centered_residual_table()

    def canonical_group_probability_table(self) -> Tensor:
        rows = []
        schedule_logits = self.canonical_schedule_logit_table()
        for nfe_index in range(len(self.config.target_nfes)):
            indices = torch.full(
                (self.config.class_count,),
                nfe_index,
                dtype=torch.int64,
                device=self.global_logits_by_nfe.device,
            )
            group_logits = self._group_logits_from_schedule_logits(
                schedule_logits[nfe_index],
                indices,
            )
            rows.append(torch.softmax(group_logits, dim=-1))
        return torch.stack(rows)

    def canonical_schedule_probability_table(self) -> Tensor:
        group_probabilities = self.canonical_group_probability_table()
        return torch.einsum(
            "nsg,ncg->ncs",
            self._group_expand,
            group_probabilities,
        )

    def canonical_density_table(self) -> Tensor:
        return torch.einsum(
            "ncs,nsb->ncb",
            self.canonical_schedule_probability_table(),
            self._clock_density_mass,
        )

    def forward(self, contexts: Tensor, target_nfes: Tensor) -> Tensor:
        return self.schedule_probabilities(contexts, target_nfes)


@dataclass(frozen=True, slots=True)
class ImageGICOClockRealization:
    """A replayable convex realization of one complete-clock mixture."""

    schedule: ScheduleBatch
    alpha: float
    selected_schedule_indices: tuple[int | None, ...]
    selected_schedule_keys: tuple[str | None, ...]
    probability_sha256: str
    uniforms_sha256: str | None
    clock_library_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ScheduleBatch):
            raise TypeError("schedule must be a ScheduleBatch.")
        alpha = _alpha(self.alpha)
        indices = tuple(self.selected_schedule_indices)
        keys = tuple(self.selected_schedule_keys)
        if len(indices) != self.schedule.batch_size or len(keys) != len(indices):
            raise ValueError("Selected clock metadata must match the schedule batch.")
        if alpha == 0.0:
            if (
                any(value is not None for value in indices)
                or any(value is not None for value in keys)
                or self.uniforms_sha256 is not None
            ):
                raise ValueError("alpha=0 realizations must not depend on uniforms.")
        else:
            for index, key in zip(indices, keys, strict=True):
                if isinstance(index, bool) or not isinstance(index, Integral) or int(index) < 0:
                    raise ValueError("Selected schedule indices must be nonnegative.")
                _schedule_key(key, field="selected_schedule_keys")
            _identity(self.uniforms_sha256, field="uniforms_sha256")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(
            self,
            "selected_schedule_indices",
            tuple(None if value is None else int(value) for value in indices),
        )
        object.__setattr__(self, "selected_schedule_keys", keys)
        object.__setattr__(
            self,
            "probability_sha256",
            _identity(self.probability_sha256, field="probability_sha256"),
        )
        object.__setattr__(
            self,
            "clock_library_sha256",
            _identity(self.clock_library_sha256, field="clock_library_sha256"),
        )
        object.__setattr__(
            self,
            "policy_sha256",
            _identity(self.policy_sha256, field="policy_sha256"),
        )

    @property
    def sha256(self) -> str:
        return semantic_sha256(
            {
                "protocol": IMAGE_GICO_CLOCK_MIXTURE_POLICY_PROTOCOL,
                "schedule_sha256": self.schedule.sha256,
                "alpha": self.alpha,
                "selected_schedule_indices": list(self.selected_schedule_indices),
                "selected_schedule_keys": list(self.selected_schedule_keys),
                "probability_sha256": self.probability_sha256,
                "uniforms_sha256": self.uniforms_sha256,
                "clock_library_sha256": self.clock_library_sha256,
                "policy_sha256": self.policy_sha256,
            },
            namespace="image-gico-clock-realization-v1",
        )


def _alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("alpha must be a finite real in [0, 1].")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("alpha must be a finite real in [0, 1].")
    return parsed


class ImageGICOClockMixturePolicy(nn.Module):
    """Mean prediction and explicit complete-clock stochastic realization."""

    def __init__(
        self,
        model: ImageGICOBackboneContextClockMixtureModel,
        library: ImageGICOClockLibrary,
        *,
        context_binding_sha256: str,
    ) -> None:
        super().__init__()
        if not isinstance(model, ImageGICOBackboneContextClockMixtureModel):
            raise TypeError("model must be ImageGICOBackboneContextClockMixtureModel.")
        if not isinstance(library, ImageGICOClockLibrary):
            raise TypeError("library must be an ImageGICOClockLibrary.")
        if model.library is not library or model.library.sha256 != library.sha256:
            raise ValueError("Model and policy clock libraries disagree.")
        self.model = model
        self.library = library
        self.context_binding_sha256 = _identity(
            context_binding_sha256,
            field="context_binding_sha256",
        )

    @property
    def policy_sha256(self) -> str:
        return semantic_sha256(
            {
                "protocol": IMAGE_GICO_CLOCK_MIXTURE_POLICY_PROTOCOL,
                "conditioning": ("normalized_frozen_backbone_map_label_plus_target_nfe"),
                "context_binding_sha256": self.context_binding_sha256,
                "target_sha256": self.library.target_sha256,
                "clock_library_sha256": self.library.sha256,
                "model_config_sha256": self.model.config.sha256,
                "model_state_sha256": self.model.state_sha256,
            },
            namespace="image-gico-clock-mixture-policy-v1",
        )

    def _schedule_probabilities(
        self,
        contexts: Tensor,
        *,
        target_nfe: object,
    ) -> Tensor:
        nfe = normalize_image_nfe(target_nfe)
        validated = self.model._validate_contexts(contexts)
        target_nfes = torch.full(
            (validated.shape[0],),
            nfe,
            dtype=torch.int64,
            device=validated.device,
        )
        with torch.no_grad():
            return self.model.schedule_probabilities(validated, target_nfes)

    def _portable_schedule_probabilities(
        self,
        contexts: Tensor,
        *,
        target_nfe: object,
    ) -> Tensor:
        current_library_sha256 = self.library.sha256
        if (
            self.model.library is not self.library
            or current_library_sha256 != self.model.config.clock_library_sha256
        ):
            raise ValueError("Clock-mixture executable library identity changed.")
        probabilities = self._schedule_probabilities(
            contexts,
            target_nfe=target_nfe,
        ).to(device="cpu", dtype=torch.float64)
        totals = probabilities.sum(dim=-1, keepdim=True)
        if (
            not bool(torch.isfinite(probabilities).all())
            or bool(torch.any(probabilities <= 0.0))
            or not bool(torch.isfinite(totals).all())
            or bool(torch.any(totals <= 0.0))
        ):
            raise FloatingPointError(
                "Clock-mixture probabilities must be finite and positive."
            )
        return probabilities / totals

    def predict(self, contexts: Tensor, *, target_nfe: int) -> ScheduleBatch:
        """Return the conditional barycenter with no stochastic draw."""

        nfe = normalize_image_nfe(target_nfe)
        probabilities = self._portable_schedule_probabilities(
            contexts,
            target_nfe=nfe,
        )
        density = probabilities @ self.library.density_table(nfe)
        return ScheduleBatch.from_density_mass(
            density,
            target_nfe=nfe,
            reference_time_grid=self.library.reference_time_grid,
            specification=IMAGE_GICO_CLOCK_MIXTURE_POLICY_SPECIFICATION,
        )

    def sample_realization(
        self,
        contexts: Tensor,
        *,
        target_nfe: int,
        uniforms: Tensor,
        alpha: float,
    ) -> ImageGICOClockRealization:
        """Draw one full clock per row using only caller-supplied uniforms."""

        nfe = normalize_image_nfe(target_nfe)
        mixing = _alpha(alpha)
        if not isinstance(uniforms, Tensor):
            raise TypeError("uniforms must be a torch.Tensor.")
        if uniforms.device.type != "cpu" or uniforms.dtype != torch.float64:
            raise TypeError("uniforms must be a CPU torch.float64 tensor.")
        if uniforms.ndim != 1:
            raise ValueError("uniforms must have shape [batch].")
        validated_contexts = self.model._validate_contexts(contexts)
        if uniforms.shape[0] != validated_contexts.shape[0]:
            raise ValueError("uniforms and contexts must share a batch size.")
        if (
            not bool(torch.isfinite(uniforms).all())
            or bool(torch.any(uniforms < 0.0))
            or bool(torch.any(uniforms >= 1.0))
        ):
            raise ValueError("uniforms must contain only values in [0, 1).")

        probabilities = self._portable_schedule_probabilities(
            validated_contexts,
            target_nfe=nfe,
        )
        probability_identity = semantic_sha256(
                _tensor_payload(probabilities),
            namespace="image-gico-clock-probability-table-v1",
        )
        exact_bank = self.library.density_table(nfe)
        mean_density = probabilities @ exact_bank
        if mixing == 0.0:
            density = mean_density
            selected_indices: tuple[int | None, ...] = (None,) * int(uniforms.numel())
            selected_keys: tuple[str | None, ...] = (None,) * int(uniforms.numel())
            uniforms_identity = None
        else:
            cumulative = torch.cumsum(probabilities, dim=-1)
            cumulative[:, -1] = 1.0
            selected = torch.searchsorted(
                cumulative.contiguous(),
                uniforms[:, None].contiguous(),
                right=False,
            ).squeeze(-1)
            if bool(torch.any(selected >= self.library.schedule_count)):
                raise RuntimeError("Clock sampling produced an invalid support index.")
            selected_density = exact_bank[selected]
            density = (
                selected_density.clone()
                if mixing == 1.0
                else mean_density + mixing * (selected_density - mean_density)
            )
            selected_indices = tuple(int(value) for value in selected.tolist())
            selected_keys = tuple(self.library.schedule_keys[index] for index in selected_indices)
            uniforms_identity = semantic_sha256(
                _tensor_payload(uniforms),
                namespace="image-gico-clock-uniforms-v1",
            )
        schedule = ScheduleBatch.from_density_mass(
            density,
            target_nfe=nfe,
            reference_time_grid=self.library.reference_time_grid,
            specification=IMAGE_GICO_CLOCK_MIXTURE_POLICY_SPECIFICATION,
        )
        return ImageGICOClockRealization(
            schedule=schedule,
            alpha=mixing,
            selected_schedule_indices=selected_indices,
            selected_schedule_keys=selected_keys,
            probability_sha256=probability_identity,
            uniforms_sha256=uniforms_identity,
            clock_library_sha256=self.library.sha256,
            policy_sha256=self.policy_sha256,
        )


__all__ = [
    "IMAGE_GICO_CLOCK_LIBRARY_PROTOCOL",
    "IMAGE_GICO_CLOCK_MIXTURE_MODEL_PROTOCOL",
    "IMAGE_GICO_CLOCK_MIXTURE_POLICY_PROTOCOL",
    "IMAGE_GICO_CLOCK_MIXTURE_POLICY_SPECIFICATION",
    "IMAGE_GICO_CLOCK_RNG_PROTOCOL",
    "ImageGICOBackboneContextClockMixtureModel",
    "ImageGICOClockGroup",
    "ImageGICOClockLibrary",
    "ImageGICOClockMixtureModelConfig",
    "ImageGICOClockMixturePolicy",
    "ImageGICOClockRealization",
    "build_image_gico_clock_library",
    "build_image_gico_clock_library_from_time_grids",
    "derive_image_gico_clock_uniforms",
    "image_gico_clock_sample_key",
    "image_gico_clock_uniform",
    "image_clock_mixture_module_state_sha256",
    "image_clock_mixture_serialized_state_sha256",
]
