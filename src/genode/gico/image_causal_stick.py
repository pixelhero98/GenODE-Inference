"""Endpoint-aware finite-support stick encoding for causal-AR GICO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from genode.gico.image_supervision import (
    IMAGE_GICO_DENSITY_BIN_COUNT,
    IMAGE_GICO_TARGET_NFES,
)

CAUSAL_STICK_PROTOCOL = "image-gico-causal-stick-path-bank-v1"
CAUSAL_STICK_QUANTIZER = "endpoint-aware-256-token-cube-companded-v2"
PREFIX_TRIE_PROTOCOL = "image-gico-causal-prefix-trie-v1"
TARGET_NFES = IMAGE_GICO_TARGET_NFES
DENSITY_BIN_COUNT = IMAGE_GICO_DENSITY_BIN_COUNT
STICK_ACTION_COUNT = DENSITY_BIN_COUNT - 1
TOKEN_COUNT = 256
INTERIOR_TOKEN_COUNT = TOKEN_COUNT - 2
INVERSE_CDF_DENSITY_GUARD = 1e-8
MAXIMUM_CLOCK_NODE_DRIFT = 0.005
MASS_NORMALIZATION_ATOL = 1e-12
_BUILD_MARKER = object()


def _freeze(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=value.dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _float64_array(
    value: object,
    *,
    field_name: str,
    ndim: int | None = None,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{field_name} must be an exact numpy.ndarray.")
    array = value
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{field_name} must use numpy.float64.")
    if not array.flags.c_contiguous:
        raise ValueError(f"{field_name} must be C-contiguous.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{field_name} must have rank {ndim}.")
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError(f"{field_name} must be nonempty and finite.")
    return array


def _int64_array(
    value: object,
    *,
    field_name: str,
    ndim: int | None = None,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{field_name} must be an exact numpy.ndarray.")
    array = value
    if array.dtype != np.dtype(np.int64):
        raise TypeError(f"{field_name} must use numpy.int64.")
    if not array.flags.c_contiguous:
        raise ValueError(f"{field_name} must be C-contiguous.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{field_name} must have rank {ndim}.")
    if array.size == 0:
        raise ValueError(f"{field_name} must be nonempty.")
    if bool(np.any((array < 0) | (array >= TOKEN_COUNT))):
        raise ValueError(f"{field_name} must contain tokens in [0, {TOKEN_COUNT - 1}].")
    return array


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite real scalar.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite.")
    return parsed


def _target_nfe(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("target_nfe must be an integer.")
    parsed = int(value)
    if parsed not in TARGET_NFES:
        raise ValueError(f"target_nfe must be one of {TARGET_NFES}.")
    return parsed


def _validate_reference_time_grid(value: object) -> np.ndarray:
    reference = _float64_array(value, field_name="reference_time_grid", ndim=1)
    if reference.shape != (DENSITY_BIN_COUNT + 1,):
        raise ValueError(f"reference_time_grid must have shape [{DENSITY_BIN_COUNT + 1}].")
    if float(reference[0]) != 0.0 or float(reference[-1]) != 1.0:
        raise ValueError("reference_time_grid endpoints must be exactly 0 and 1.")
    if not bool(np.all(np.diff(reference) > 0.0)):
        raise ValueError("reference_time_grid must be strictly increasing.")
    canonical = np.linspace(0.0, 1.0, DENSITY_BIN_COUNT + 1, dtype=np.float64)
    if not np.array_equal(reference, canonical):
        raise ValueError("reference_time_grid must be the protocol-fixed uniform 64-bin grid.")
    return reference


def _validate_density_mass(value: object, *, field_name: str) -> np.ndarray:
    mass = _float64_array(value, field_name=field_name)
    if mass.ndim < 1 or mass.shape[-1] != DENSITY_BIN_COUNT:
        raise ValueError(f"{field_name} must have final dimension {DENSITY_BIN_COUNT}.")
    if bool(np.any(mass < 0.0)):
        raise ValueError(f"{field_name} must be nonnegative.")
    totals = np.sum(mass, axis=-1, dtype=np.float64)
    if not bool(np.allclose(totals, np.ones_like(totals), rtol=0.0, atol=MASS_NORMALIZATION_ATOL)):
        raise ValueError(f"{field_name} rows must sum to one.")
    return mass


def encode_cube_companded_action(action: object) -> int:
    parsed = _finite_real(action, field_name="action")
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("action must lie in [0, 1].")
    if parsed == 0.0:
        return 0
    if parsed == 1.0:
        return TOKEN_COUNT - 1
    interior = int(np.floor(INTERIOR_TOKEN_COUNT * np.cbrt(np.float64(parsed))))
    return 1 + min(INTERIOR_TOKEN_COUNT - 1, interior)


def decode_cube_companded_token(token: object) -> np.float64:
    if isinstance(token, bool) or not isinstance(token, Integral):
        raise TypeError("token must be an integer.")
    parsed = int(token)
    if not 0 <= parsed < TOKEN_COUNT:
        raise ValueError(f"token must lie in [0, {TOKEN_COUNT - 1}].")
    if parsed == 0:
        return np.float64(0.0)
    if parsed == TOKEN_COUNT - 1:
        return np.float64(1.0)
    midpoint = np.float64(parsed - 1 + 0.5) / np.float64(INTERIOR_TOKEN_COUNT)
    return np.float64(midpoint * midpoint * midpoint)


def encode_cube_companded_actions(stick_actions: object) -> np.ndarray:
    actions = _float64_array(stick_actions, field_name="stick_actions")
    if bool(np.any((actions < 0.0) | (actions > 1.0))):
        raise ValueError("stick_actions must lie in [0, 1].")
    tokens = np.empty(actions.shape, dtype=np.int64)
    zero = actions == 0.0
    one = actions == 1.0
    interior = ~(zero | one)
    tokens[zero] = 0
    tokens[one] = TOKEN_COUNT - 1
    if bool(np.any(interior)):
        bins = np.floor(np.float64(INTERIOR_TOKEN_COUNT) * np.cbrt(actions[interior])).astype(np.int64)
        tokens[interior] = 1 + np.minimum(bins, INTERIOR_TOKEN_COUNT - 1)
    return _freeze(tokens)


def decode_cube_companded_tokens(tokens: object) -> np.ndarray:
    encoded = _int64_array(tokens, field_name="tokens")
    actions = np.empty(encoded.shape, dtype=np.float64)
    zero = encoded == 0
    one = encoded == TOKEN_COUNT - 1
    interior = ~(zero | one)
    actions[zero] = 0.0
    actions[one] = 1.0
    if bool(np.any(interior)):
        midpoint = (encoded[interior].astype(np.float64) - np.float64(0.5)) / np.float64(INTERIOR_TOKEN_COUNT)
        actions[interior] = midpoint * midpoint * midpoint
    return _freeze(actions)


def density_to_stick_actions(density_mass: object) -> np.ndarray:
    mass = _validate_density_mass(density_mass, field_name="density_mass")
    rows = mass.reshape(-1, DENSITY_BIN_COUNT)
    result = np.empty((rows.shape[0], STICK_ACTION_COUNT), dtype=np.float64)
    for row_index, row in enumerate(rows):
        remaining = np.float64(1.0)
        for action_index in range(STICK_ACTION_COUNT):
            value = np.float64(row[action_index])
            if remaining == 0.0:
                if value != 0.0:
                    raise ValueError("Positive density follows an exhausted stick.")
                action = np.float64(0.0)
            else:
                action = np.float64(value / remaining)
                if not np.isfinite(action) or not 0.0 <= action <= 1.0:
                    raise ValueError("density_mass cannot be represented as a stick.")
            result[row_index, action_index] = action
            remaining = np.float64(remaining * (np.float64(1.0) - action))
        if abs(float(remaining) - float(row[-1])) > MASS_NORMALIZATION_ATOL:
            raise ValueError("Final density bin differs from the stick remainder.")
    return _freeze(result.reshape(mass.shape[:-1] + (STICK_ACTION_COUNT,)))


def stick_actions_to_density(stick_actions: object) -> np.ndarray:
    actions = _float64_array(stick_actions, field_name="stick_actions")
    if actions.ndim < 1 or actions.shape[-1] != STICK_ACTION_COUNT:
        raise ValueError(f"stick_actions must have final dimension {STICK_ACTION_COUNT}.")
    if bool(np.any((actions < 0.0) | (actions > 1.0))):
        raise ValueError("stick_actions must lie in [0, 1].")
    rows = actions.reshape(-1, STICK_ACTION_COUNT)
    result = np.empty((rows.shape[0], DENSITY_BIN_COUNT), dtype=np.float64)
    for row_index, row in enumerate(rows):
        remaining = np.float64(1.0)
        for action_index, action in enumerate(row):
            result[row_index, action_index] = np.float64(remaining * action)
            remaining = np.float64(remaining * (np.float64(1.0) - action))
        result[row_index, -1] = remaining
    shaped = np.ascontiguousarray(result.reshape(actions.shape[:-1] + (DENSITY_BIN_COUNT,)))
    _validate_density_mass(shaped, field_name="reconstructed_density_mass")
    return _freeze(shaped)


def guard_density_for_inverse_cdf(
    density_mass: object,
    *,
    density_guard: object = INVERSE_CDF_DENSITY_GUARD,
) -> np.ndarray:
    guard = _finite_real(density_guard, field_name="density_guard")
    if guard != INVERSE_CDF_DENSITY_GUARD:
        raise ValueError(f"density_guard is protocol-bound to {INVERSE_CDF_DENSITY_GUARD:g}.")
    mass = _validate_density_mass(density_mass, field_name="density_mass")
    guarded = np.maximum(mass, np.float64(guard))
    guarded = guarded / np.sum(guarded, axis=-1, keepdims=True, dtype=np.float64)
    result = np.ascontiguousarray(guarded, dtype=np.float64)
    _validate_density_mass(result, field_name="guarded_density_mass")
    return _freeze(result)


def inverse_cdf_clock_nodes(
    density_mass: object,
    target_nfe: object,
    reference_time_grid: object,
    *,
    density_guard: object = INVERSE_CDF_DENSITY_GUARD,
) -> np.ndarray:
    """Materialize equal-mass clock nodes after the protocol guard."""

    nfe = _target_nfe(target_nfe)
    reference = _validate_reference_time_grid(reference_time_grid)
    guarded = guard_density_for_inverse_cdf(density_mass, density_guard=density_guard)
    rows = guarded.reshape(-1, DENSITY_BIN_COUNT)
    grids = np.empty((rows.shape[0], nfe + 1), dtype=np.float64)
    grids[:, 0] = 0.0
    grids[:, -1] = 1.0
    quantiles = np.arange(1, nfe, dtype=np.float64) / np.float64(nfe)
    for row_index, row in enumerate(rows):
        cumulative = np.cumsum(row, dtype=np.float64)
        cumulative[-1] = 1.0
        for quantile_index, quantile in enumerate(quantiles, start=1):
            bin_index = min(
                int(np.searchsorted(cumulative, quantile, side="left")),
                DENSITY_BIN_COUNT - 1,
            )
            cumulative_left = np.float64(0.0) if bin_index == 0 else np.float64(cumulative[bin_index - 1])
            selected_mass = np.float64(row[bin_index])
            if selected_mass <= 0.0:
                raise FloatingPointError("Guarded inverse CDF selected a nonpositive density bin.")
            fraction = np.float64((quantile - cumulative_left) / selected_mass)
            left_edge = np.float64(reference[bin_index])
            right_edge = np.float64(reference[bin_index + 1])
            grids[row_index, quantile_index] = np.float64(left_edge + fraction * (right_edge - left_edge))
    if not bool(np.isfinite(grids).all()) or not bool(np.all(np.diff(grids, axis=-1) > 0.0)):
        raise FloatingPointError("Inverse-CDF clock nodes must be finite and increasing.")
    return _freeze(grids.reshape(guarded.shape[:-1] + (nfe + 1,)))


def _normalize_prefix(value: object) -> tuple[int, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
            raise TypeError("prefix arrays must be one-dimensional integers.")
        raw: Sequence[object] = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = value
    else:
        raise TypeError("prefix must be a sequence of integer tokens.")
    result: list[int] = []
    for token in raw:
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise TypeError("prefix must contain integer tokens.")
        parsed = int(token)
        if not 0 <= parsed < TOKEN_COUNT:
            raise ValueError(f"prefix tokens must lie in [0, {TOKEN_COUNT - 1}].")
        result.append(parsed)
    if len(result) > STICK_ACTION_COUNT:
        raise ValueError(f"prefix may contain at most {STICK_ACTION_COUNT} tokens.")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PrefixTrie:
    target_nfe: int
    children_by_prefix: Mapping[tuple[int, ...], tuple[int, ...]] = field(repr=False)
    aliases_by_complete_path: Mapping[tuple[int, ...], tuple[int, ...]] = field(repr=False)
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _BUILD_MARKER:
            raise TypeError("PrefixTrie must be created with from_token_paths().")
        _target_nfe(self.target_nfe)

    @classmethod
    def from_token_paths(cls, target_nfe: object, token_paths: object) -> PrefixTrie:
        nfe = _target_nfe(target_nfe)
        paths = _int64_array(token_paths, field_name="token_paths", ndim=2)
        if paths.shape[1] != STICK_ACTION_COUNT:
            raise ValueError(f"token_paths must have shape [path, {STICK_ACTION_COUNT}].")
        children: dict[tuple[int, ...], set[int]] = {}
        aliases: dict[tuple[int, ...], list[int]] = {}
        for schedule_index, row in enumerate(paths):
            path = tuple(int(value) for value in row)
            for depth, token in enumerate(path):
                children.setdefault(path[:depth], set()).add(token)
            children.setdefault(path, set())
            aliases.setdefault(path, []).append(schedule_index)
        return cls(
            target_nfe=nfe,
            children_by_prefix=MappingProxyType(
                {
                    prefix: tuple(sorted(values))
                    for prefix, values in sorted(children.items(), key=lambda item: (len(item[0]), item[0]))
                }
            ),
            aliases_by_complete_path=MappingProxyType(
                {path: tuple(indices) for path, indices in sorted(aliases.items())}
            ),
            _marker=_BUILD_MARKER,
        )

    def valid_children(self, prefix: object) -> tuple[int, ...]:
        normalized = _normalize_prefix(prefix)
        try:
            return self.children_by_prefix[normalized]
        except KeyError as exc:
            raise KeyError("Prefix is outside the frozen teacher support.") from exc

    def child_mask(self, prefix: object) -> np.ndarray:
        children = self.valid_children(prefix)
        mask = np.zeros((TOKEN_COUNT,), dtype=np.bool_)
        if children:
            mask[list(children)] = True
        return _freeze(mask)

    def alias_members(self, complete_path: object) -> tuple[int, ...]:
        path = _normalize_prefix(complete_path)
        if len(path) != STICK_ACTION_COUNT:
            raise ValueError(f"complete_path must contain {STICK_ACTION_COUNT} tokens.")
        try:
            return self.aliases_by_complete_path[path]
        except KeyError as exc:
            raise KeyError("Path is outside the frozen teacher support.") from exc


@dataclass(frozen=True, slots=True)
class CausalStickDiagnostics:
    maximum_clock_node_drift: np.ndarray
    clock_node_drift_by_nfe: tuple[np.ndarray, ...]
    maximum_density_absolute_error: np.ndarray
    density_l1: np.ndarray
    maximum_clock_node_drift_limit: float

    @property
    def maximum_observed_clock_node_drift(self) -> float:
        return float(np.max(self.maximum_clock_node_drift))

    def as_payload(self) -> dict[str, object]:
        return {
            "maximum_clock_node_drift": self.maximum_clock_node_drift.tolist(),
            "maximum_observed_clock_node_drift": self.maximum_observed_clock_node_drift,
            "maximum_density_absolute_error": self.maximum_density_absolute_error.tolist(),
            "density_l1": self.density_l1.tolist(),
            "maximum_clock_node_drift_limit": self.maximum_clock_node_drift_limit,
        }


@dataclass(frozen=True, slots=True)
class ImageGICOCausalPathBank:
    target_nfes: tuple[int, ...]
    reference_time_grid: np.ndarray
    canonical_density_paths: np.ndarray
    token_paths: np.ndarray
    decoded_density_paths: np.ndarray
    tries: tuple[PrefixTrie, ...]
    unique_token_paths_by_nfe: tuple[np.ndarray, ...]
    alias_member_indices_by_nfe: tuple[tuple[tuple[int, ...], ...], ...]
    schedule_to_alias_index: np.ndarray
    diagnostics: CausalStickDiagnostics
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._marker is not _BUILD_MARKER:
            raise TypeError("ImageGICOCausalPathBank must be created with build().")

    @classmethod
    def build(
        cls,
        canonical_density_paths: object,
        reference_time_grid: object,
        *,
        target_nfes: tuple[int, ...] = TARGET_NFES,
        maximum_clock_node_drift: object = MAXIMUM_CLOCK_NODE_DRIFT,
    ) -> ImageGICOCausalPathBank:
        nfes = tuple(target_nfes)
        if nfes != TARGET_NFES:
            raise ValueError(f"target_nfes must be exactly {TARGET_NFES}.")
        limit = _finite_real(maximum_clock_node_drift, field_name="maximum_clock_node_drift")
        if not 0.0 <= limit <= MAXIMUM_CLOCK_NODE_DRIFT:
            raise ValueError(f"maximum_clock_node_drift must lie in [0, {MAXIMUM_CLOCK_NODE_DRIFT}].")
        canonical = _float64_array(
            canonical_density_paths,
            field_name="canonical_density_paths",
            ndim=3,
        )
        if canonical.shape[0] != len(nfes) or canonical.shape[2] != DENSITY_BIN_COUNT:
            raise ValueError("canonical_density_paths must have shape [3, schedule, 64].")
        _validate_density_mass(canonical, field_name="canonical_density_paths")
        reference = _validate_reference_time_grid(reference_time_grid)
        actions = density_to_stick_actions(canonical)
        tokens = encode_cube_companded_actions(actions)
        decoded = stick_actions_to_density(decode_cube_companded_tokens(tokens))
        absolute_error = np.abs(decoded - canonical)

        drift_rows: list[np.ndarray] = []
        maximum_drift = np.empty(canonical.shape[:2], dtype=np.float64)
        for nfe_index, nfe in enumerate(nfes):
            expected = inverse_cdf_clock_nodes(np.ascontiguousarray(canonical[nfe_index]), nfe, reference)
            observed = inverse_cdf_clock_nodes(np.ascontiguousarray(decoded[nfe_index]), nfe, reference)
            drift = np.abs(observed - expected)
            drift_rows.append(_freeze(drift))
            maximum_drift[nfe_index] = np.max(drift, axis=-1)
        maximum_observed = float(np.max(maximum_drift))
        if maximum_observed > limit:
            raise ValueError(
                "Quantized stick preflight failed: maximum inverse-CDF clock-node "
                f"drift {maximum_observed:.17g} exceeds {limit:.17g}."
            )

        tries: list[PrefixTrie] = []
        unique_rows: list[np.ndarray] = []
        alias_rows: list[tuple[tuple[int, ...], ...]] = []
        schedule_to_alias = np.empty(canonical.shape[:2], dtype=np.int64)
        for nfe_index, nfe in enumerate(nfes):
            nfe_paths = np.ascontiguousarray(tokens[nfe_index])
            trie = PrefixTrie.from_token_paths(nfe, nfe_paths)
            tries.append(trie)
            lookup: dict[tuple[int, ...], int] = {}
            paths: list[tuple[int, ...]] = []
            members: list[list[int]] = []
            for schedule_index, row in enumerate(nfe_paths):
                path = tuple(int(value) for value in row)
                alias_index = lookup.get(path)
                if alias_index is None:
                    alias_index = len(paths)
                    lookup[path] = alias_index
                    paths.append(path)
                    members.append([])
                members[alias_index].append(schedule_index)
                schedule_to_alias[nfe_index, schedule_index] = alias_index
            unique_rows.append(_freeze(np.ascontiguousarray(np.asarray(paths, dtype=np.int64))))
            alias_rows.append(tuple(tuple(row) for row in members))
        diagnostics = CausalStickDiagnostics(
            maximum_clock_node_drift=_freeze(maximum_drift),
            clock_node_drift_by_nfe=tuple(drift_rows),
            maximum_density_absolute_error=_freeze(np.max(absolute_error, axis=-1)),
            density_l1=_freeze(np.sum(absolute_error, axis=-1, dtype=np.float64)),
            maximum_clock_node_drift_limit=limit,
        )
        return cls(
            target_nfes=nfes,
            reference_time_grid=_freeze(reference),
            canonical_density_paths=_freeze(canonical),
            token_paths=tokens,
            decoded_density_paths=decoded,
            tries=tuple(tries),
            unique_token_paths_by_nfe=tuple(unique_rows),
            alias_member_indices_by_nfe=tuple(alias_rows),
            schedule_to_alias_index=_freeze(schedule_to_alias),
            diagnostics=diagnostics,
            _marker=_BUILD_MARKER,
        )

    @property
    def schedule_count(self) -> int:
        return int(self.canonical_density_paths.shape[1])

    def nfe_index(self, target_nfe: object) -> int:
        return self.target_nfes.index(_target_nfe(target_nfe))

    def valid_next_token_mask(self, target_nfe: object, prefix: object) -> np.ndarray:
        return self.tries[self.nfe_index(target_nfe)].child_mask(prefix)

    def decode_supported_path(self, target_nfe: object, token_path: object) -> np.ndarray:
        trie = self.tries[self.nfe_index(target_nfe)]
        normalized = _normalize_prefix(token_path)
        trie.alias_members(normalized)
        encoded = np.ascontiguousarray(np.asarray(normalized, dtype=np.int64))
        return stick_actions_to_density(decode_cube_companded_tokens(encoded))

    def aggregate_teacher_weights(self, teacher_weights: object) -> tuple[np.ndarray, ...]:
        weights = _float64_array(teacher_weights, field_name="teacher_weights", ndim=3)
        expected = (
            len(self.target_nfes),
            weights.shape[1],
            self.schedule_count,
        )
        if weights.shape != expected:
            raise ValueError("teacher_weights must have shape [3, context, schedule].")
        if bool(np.any(weights < 0.0)):
            raise ValueError("teacher_weights must be nonnegative.")
        row_totals = np.sum(weights, axis=-1, dtype=np.float64)
        if not bool(
            np.allclose(
                row_totals,
                np.ones_like(row_totals),
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise ValueError("teacher_weights rows must sum to one.")
        aggregated_rows: list[np.ndarray] = []
        for nfe_index, groups in enumerate(self.alias_member_indices_by_nfe):
            columns = [
                np.sum(
                    np.take(weights[nfe_index], members, axis=-1),
                    axis=-1,
                    dtype=np.float64,
                )
                for members in groups
            ]
            aggregated_rows.append(_freeze(np.stack(columns, axis=-1)))
        return tuple(aggregated_rows)

    def as_metadata_payload(self) -> dict[str, object]:
        return {
            "protocol": CAUSAL_STICK_PROTOCOL,
            "quantizer": CAUSAL_STICK_QUANTIZER,
            "target_nfes": list(self.target_nfes),
            "schedule_count": self.schedule_count,
            "density_bin_count": DENSITY_BIN_COUNT,
            "stick_action_count": STICK_ACTION_COUNT,
            "token_count": TOKEN_COUNT,
            "endpoint_tokens": {"zero": 0, "one": TOKEN_COUNT - 1},
            "interior_encode_rule": (
                f"floor({INTERIOR_TOKEN_COUNT}*cuberoot(action))+1_with_interior_cap_{INTERIOR_TOKEN_COUNT}"
            ),
            "interior_decode_rule": (f"((token-1+0.5)/{INTERIOR_TOKEN_COUNT})^3"),
            "inverse_cdf_density_guard": INVERSE_CDF_DENSITY_GUARD,
            "maximum_clock_node_drift": MAXIMUM_CLOCK_NODE_DRIFT,
            "alias_member_indices_by_nfe": [
                [list(members) for members in groups] for groups in self.alias_member_indices_by_nfe
            ],
            "diagnostics": self.diagnostics.as_payload(),
        }


__all__ = [
    "CAUSAL_STICK_PROTOCOL",
    "CAUSAL_STICK_QUANTIZER",
    "CausalStickDiagnostics",
    "DENSITY_BIN_COUNT",
    "INTERIOR_TOKEN_COUNT",
    "INVERSE_CDF_DENSITY_GUARD",
    "ImageGICOCausalPathBank",
    "MAXIMUM_CLOCK_NODE_DRIFT",
    "PrefixTrie",
    "STICK_ACTION_COUNT",
    "TARGET_NFES",
    "TOKEN_COUNT",
    "decode_cube_companded_token",
    "decode_cube_companded_tokens",
    "density_to_stick_actions",
    "encode_cube_companded_action",
    "encode_cube_companded_actions",
    "guard_density_for_inverse_cdf",
    "inverse_cdf_clock_nodes",
    "stick_actions_to_density",
]
