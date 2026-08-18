from __future__ import annotations

import importlib.util
import itertools
import pickle
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType, ModuleType

import torch
from torch import Tensor, nn

from genode.artifacts.identity import semantic_sha256
from genode.path_safety import resolve_portable_relative_path

from ._strict_json import loads_strict_json
from .registry import RFPP_CONFIG_IDENTITY_NAMESPACE, ImageBackboneSpec

_MAX_CONFIG_BYTES = 64 * 1024
_IMPORT_LOCK = threading.RLock()
_IMPORT_COUNTER = itertools.count()
_RESERVED_MODULE_ROOTS = ("network_edm", "persistence", "dnnlib")
_CONFIG_METADATA_FIELDS = {"num_params", "unet_type", "use_fp16"}
_SONG_UNET_DEFAULTS: dict[str, object] = {
    "label_dim": 0,
    "augment_dim": 0,
    "aux_dim": 0,
    "model_channels": 128,
    "channel_mult": [1, 2, 2, 2],
    "channel_mult_emb": 4,
    "num_blocks": 4,
    "attn_resolutions": [16],
    "dropout": 0.10,
    "label_dropout": 0,
    "embedding_type": "positional",
    "channel_mult_noise": 1,
    "encoder_type": "standard",
    "decoder_type": "standard",
    "resample_filter": [1, 1],
    "aug": False,
    "aug_dim": 32,
}
_DHARIWAL_UNET_DEFAULTS: dict[str, object] = {
    "label_dim": 0,
    "augment_dim": 0,
    "model_channels": 192,
    "channel_mult": [1, 2, 3, 4],
    "channel_mult_emb": 4,
    "num_blocks": 3,
    "attn_resolutions": [32, 16, 8],
    "dropout": 0.10,
    "label_dropout": 0,
}
_REQUIRED_UNET_FIELDS = {"img_resolution", "in_channels", "out_channels"}


def _source_member(root: Path, relative_path: str, *, label: str) -> Path:
    path = resolve_portable_relative_path(
        root,
        relative_path,
        label=label,
        reject_links=True,
    )
    if not path.is_file():
        raise ValueError(f"RF++ source root is missing regular {label} {relative_path!r}.")
    return path


def _load_bound_config(root: Path, spec: ImageBackboneSpec) -> dict[str, object]:
    config_path = _source_member(root, spec.source_config_path, label="model config")
    if config_path.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("RF++ model config is unexpectedly large.")
    try:
        text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("RF++ model config must be readable UTF-8 text.") from exc
    config = loads_strict_json(text, label="RF++ model config")
    if not isinstance(config, dict):
        raise ValueError("RF++ model config must be a JSON object.")
    identity = semantic_sha256(config, namespace=RFPP_CONFIG_IDENTITY_NAMESPACE)
    if identity != spec.source_config_identity:
        raise ValueError("RF++ model config does not match the registry-bound official config.")

    expected_unet = "songunet" if spec.architecture == "SongUNet+EDMPrecondVel" else "adm"
    expected_resolution = spec.image_shape[1]
    required_bindings = {
        "unet_type": expected_unet,
        "img_resolution": expected_resolution,
        "in_channels": spec.image_shape[0],
        "out_channels": spec.image_shape[0],
        "label_dim": spec.num_conditioning_classes,
    }
    for key, expected in required_bindings.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError(f"RF++ model config field {key!r} conflicts with the backbone registry.")
    use_fp16 = config.get("use_fp16", False)
    if not isinstance(use_fp16, bool):
        raise ValueError("RF++ model config field 'use_fp16' must be boolean when present.")
    return config


def _module_origin_within(module: object, root: Path) -> bool:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str):
        return False
    try:
        return Path(raw_origin).resolve().is_relative_to(root)
    except OSError:
        return False


def _assert_import_environment_clean(root: Path) -> None:
    conflicting_modules = sorted(
        name
        for name in sys.modules
        if name.startswith("_genode_rfpp_")
        or any(name == reserved or name.startswith(f"{reserved}.") for reserved in _RESERVED_MODULE_ROOTS)
    )
    if conflicting_modules:
        listed = ", ".join(conflicting_modules)
        raise RuntimeError(f"Conflicting RF++ modules are already loaded: {listed}.")

    for entry in sys.path:
        candidate = Path.cwd() if entry == "" else Path(entry).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved == root:
            raise RuntimeError("RF++ source root is already present on sys.path.")


@contextmanager
def _isolated_network_module(root: Path, spec: ImageBackboneSpec) -> Iterator[ModuleType]:
    network_path = _source_member(root, "network_edm.py", label="network module")
    _source_member(root, "persistence.py", label="persistence module")
    _source_member(root, "dnnlib/__init__.py", label="dnnlib package")
    unique_name = f"_genode_rfpp_{spec.source_revision[:12]}_{next(_IMPORT_COUNTER)}"

    with _IMPORT_LOCK:
        _assert_import_environment_clean(root)
        original_path = list(sys.path)
        original_modules = set(sys.modules)
        try:
            sys.path.insert(0, str(root))
            module_spec = importlib.util.spec_from_file_location(unique_name, network_path)
            if module_spec is None or module_spec.loader is None:
                raise RuntimeError("Could not create an import specification for RF++ network_edm.py.")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[unique_name] = module
            module_spec.loader.exec_module(module)

            if "network_edm" in sys.modules:
                raise RuntimeError("RF++ network module polluted the generic 'network_edm' module name.")
            for required_name in ("persistence", "dnnlib"):
                required_module = sys.modules.get(required_name)
                if required_module is None or not _module_origin_within(required_module, root):
                    raise RuntimeError(
                        f"RF++ import {required_name!r} did not resolve inside the verified source root."
                    )
            yield module
        finally:
            sys.path[:] = original_path
            for name, loaded_module in list(sys.modules.items()):
                if name in original_modules:
                    continue
                if (
                    name == unique_name
                    or name.startswith(f"{unique_name}.")
                    or any(name == reserved or name.startswith(f"{reserved}.") for reserved in _RESERVED_MODULE_ROOTS)
                    or _module_origin_within(loaded_module, root)
                ):
                    sys.modules.pop(name, None)


def _load_plain_tensor_state_dict(checkpoint_path: Path) -> dict[str, Tensor] | OrderedDict[str, Tensor]:
    before = checkpoint_path.stat()
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("RF++ checkpoint could not be loaded as weights-only data.") from exc
    after = checkpoint_path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError("RF++ checkpoint changed while it was being loaded.")

    if type(payload) not in {dict, OrderedDict}:
        raise ValueError("RF++ checkpoint must contain a direct state_dict object.")
    if not payload:
        raise ValueError("RF++ checkpoint state_dict must not be empty.")
    for key, value in payload.items():
        if type(key) is not str or not key:
            raise ValueError("RF++ checkpoint state_dict keys must be non-empty strings.")
        if not isinstance(value, Tensor):
            raise ValueError("RF++ checkpoint state_dict values must all be tensors.")
    return payload


def _required_model_class(module: ModuleType, name: str) -> type[nn.Module]:
    value = getattr(module, name, None)
    if not isinstance(value, type) or not issubclass(value, nn.Module):
        raise ValueError(f"Verified RF++ network module does not define nn.Module class {name!r}.")
    return value


def _unet_class_name(spec: ImageBackboneSpec) -> str:
    if spec.architecture == "SongUNet+EDMPrecondVel":
        return "SongUNet"
    if spec.architecture == "DhariwalUNet+EDMPrecondVel":
        return "DhariwalUNet"
    raise ValueError(f"Unsupported registry architecture {spec.architecture!r}.")


def _random_unet_constructor_config(
    config: Mapping[str, object],
    *,
    unet_class_name: str,
    augment_dim: int,
) -> dict[str, object]:
    if isinstance(augment_dim, bool) or not isinstance(augment_dim, Integral):
        raise TypeError("augment_dim must be a positive integer.")
    normalized_augment_dim = int(augment_dim)
    if normalized_augment_dim <= 0:
        raise ValueError("augment_dim must be positive.")
    defaults = _SONG_UNET_DEFAULTS if unet_class_name == "SongUNet" else _DHARIWAL_UNET_DEFAULTS
    allowed = set(defaults) | _REQUIRED_UNET_FIELDS
    unexpected = set(config) - allowed - _CONFIG_METADATA_FIELDS
    if unexpected:
        raise ValueError(f"RF++ official config contains unsupported UNet constructor fields: {sorted(unexpected)}.")
    missing = _REQUIRED_UNET_FIELDS - set(config)
    if missing:
        raise ValueError(f"RF++ official config is missing required UNet constructor fields: {sorted(missing)}.")
    constructor = deepcopy(defaults)
    constructor.update({key: deepcopy(value) for key, value in config.items() if key in allowed})
    constructor["augment_dim"] = normalized_augment_dim
    return constructor


def _immutable_config_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable_config_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_immutable_config_value(item) for item in value)
    return value


def _immutable_constructor_config(
    config: Mapping[str, object],
) -> Mapping[str, object]:
    return MappingProxyType({key: _immutable_config_value(value) for key, value in config.items()})


@dataclass(frozen=True, slots=True)
class RandomRFPPUNetBuild:
    """A randomly initialized official RF++ UNet and its effective config."""

    model: nn.Module
    network_class: str
    constructor_config: Mapping[str, object]


def build_rfpp_random_unet(
    *,
    source_root: Path,
    spec: ImageBackboneSpec,
    augment_dim: int,
) -> RandomRFPPUNetBuild:
    """Construct a trainable FP32 UNet from the pinned official config.

    ``source_root`` must already have passed the public checkout verifier.
    External modules are loaded only inside the same isolated import boundary
    used for checkpoint-backed RF++ construction.
    """

    root = source_root.resolve(strict=True)
    config = _load_bound_config(root, spec)
    unet_class_name = _unet_class_name(spec)
    constructor_config = _random_unet_constructor_config(
        config,
        unet_class_name=unet_class_name,
        augment_dim=augment_dim,
    )
    with _isolated_network_module(root, spec) as network_module:
        unet_class = _required_model_class(network_module, unet_class_name)
        model = unet_class(**constructor_config)
        if not isinstance(model, nn.Module):
            raise TypeError(f"RF++ class {unet_class_name!r} did not construct an nn.Module.")
    model.to(dtype=torch.float32)
    for name, value in itertools.chain(
        model.named_parameters(),
        model.named_buffers(),
    ):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise TypeError(f"RF++ random UNet tensor {name!r} is not float32.")
    model.train()
    model.requires_grad_(True)
    return RandomRFPPUNetBuild(
        model=model,
        network_class=unet_class_name,
        constructor_config=_immutable_constructor_config(constructor_config),
    )


def verify_rfpp_source_configuration(
    source_root: str | Path,
    spec: ImageBackboneSpec,
) -> Path:
    """Verify the registry-bound RF++ JSON config without importing upstream code."""

    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("RF++ source root must be a directory.")
    _load_bound_config(root, spec)
    return root


def build_rfpp_native_model(
    *,
    source_root: Path,
    checkpoint_path: Path,
    spec: ImageBackboneSpec,
) -> nn.Module:
    """Construct RF++ from verified external source and a direct tensor state_dict."""

    root = source_root.resolve(strict=True)
    config = _load_bound_config(root, spec)
    state_dict = _load_plain_tensor_state_dict(checkpoint_path)
    unet_class_name = _unet_class_name(spec)

    with _isolated_network_module(root, spec) as network_module:
        unet_class = _required_model_class(network_module, unet_class_name)
        preconditioner_class = _required_model_class(network_module, "EDMPrecondVel")
        unet = unet_class(**config)
        if not isinstance(unet, nn.Module):
            raise TypeError(f"RF++ class {unet_class_name!r} did not construct an nn.Module.")
        model = preconditioner_class(
            unet,
            use_fp16=bool(config.get("use_fp16", False)),
        )
        if not isinstance(model, nn.Module):
            raise TypeError("RF++ EDMPrecondVel did not construct an nn.Module.")
        try:
            incompatible = model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError("RF++ checkpoint state_dict does not strictly match the constructed model.") from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("RF++ strict state_dict loading reported incompatible keys.")
        model.eval()
        model.requires_grad_(False)
        return model


__all__ = [
    "RandomRFPPUNetBuild",
    "build_rfpp_native_model",
    "build_rfpp_random_unet",
    "verify_rfpp_source_configuration",
]
