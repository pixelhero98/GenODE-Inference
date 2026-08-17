from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

ConditioningMode = Literal["unconditional", "class_conditional"]

RFPP_REPOSITORY_URL = "https://github.com/sangyun884/rfpp"
RFPP_PINNED_REVISION = "f7b10a5d71c6a0079e0bd46eccce2dbf99836f09"
RFPP_CONFIG_IDENTITY_NAMESPACE = "genode-rfpp-config-v1"
IMAGENET64_EDM_REPOSITORY_URL = "https://github.com/NVlabs/edm"
IMAGENET64_EDM_PREPARATION_REVISION = "008a4e5316c8e3bfe61a62f874bddba254295afb"
IMAGENET64_EDM_DATASET_TOOL_PATH = "dataset_tool.py"


@dataclass(frozen=True, slots=True)
class ImageDatasetSpec:
    key: str
    display_name: str
    image_shape: tuple[int, int, int]
    semantic_num_classes: int
    source_url: str
    license_note: str
    preparation_reference: Mapping[str, str] | None = None

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "image_shape": list(self.image_shape),
            "semantic_num_classes": self.semantic_num_classes,
            "source_url": self.source_url,
            "license_note": self.license_note,
            "preparation_reference": (None if self.preparation_reference is None else dict(self.preparation_reference)),
        }


@dataclass(frozen=True, slots=True)
class ImageBackboneSpec:
    key: str
    display_name: str
    dataset_key: str
    method_key: str
    conditioning: ConditioningMode
    num_conditioning_classes: int
    architecture: str
    source_config_path: str
    source_config_identity: str
    checkpoint_filename: str
    checkpoint_url: str
    checkpoint_published_size_bytes: int
    repository_url: str = RFPP_REPOSITORY_URL
    source_revision: str = RFPP_PINNED_REVISION
    repository_license: str = "BSD-3-Clause-Clear"
    network_implementation_license: str = "CC-BY-NC-SA-4.0"
    checkpoint_license_status: str = "no_separate_license_notice_found"
    checkpoint_distribution_policy: str = "external_user_supplied_not_redistributed"
    native_time_direction: str = "data_at_t0_to_noise_at_t1"
    native_velocity_parameterization: str = "noise_minus_data"

    @property
    def dataset(self) -> ImageDatasetSpec:
        return get_image_dataset_spec(self.dataset_key)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self.dataset.image_shape

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "dataset_key": self.dataset_key,
            "method_key": self.method_key,
            "conditioning": self.conditioning,
            "num_conditioning_classes": self.num_conditioning_classes,
            "architecture": self.architecture,
            "source_config_path": self.source_config_path,
            "source_config_identity": self.source_config_identity,
            "checkpoint_filename": self.checkpoint_filename,
            "checkpoint_url": self.checkpoint_url,
            "checkpoint_published_size_bytes": (self.checkpoint_published_size_bytes),
            "checkpoint_digest_provenance": ("local_sha256_no_upstream_published_digest"),
            "repository_url": self.repository_url,
            "source_revision": self.source_revision,
            "repository_license": self.repository_license,
            "network_implementation_license": self.network_implementation_license,
            "checkpoint_license_status": self.checkpoint_license_status,
            "checkpoint_distribution_policy": self.checkpoint_distribution_policy,
            "native_time_direction": self.native_time_direction,
            "native_velocity_parameterization": self.native_velocity_parameterization,
        }


_DATASETS = {
    "cifar10": ImageDatasetSpec(
        key="cifar10",
        display_name="CIFAR-10",
        image_shape=(3, 32, 32),
        semantic_num_classes=10,
        source_url="https://www.cs.toronto.edu/~kriz/cifar.html",
        license_note="Upstream dataset terms apply; the approved RF++ backbone is unconditional.",
    ),
    "imagenet64": ImageDatasetSpec(
        key="imagenet64",
        display_name="ImageNet-64",
        image_shape=(3, 64, 64),
        semantic_num_classes=1000,
        source_url="https://image-net.org/challenges/LSVRC/2012/2012-downloads.php",
        license_note=(
            "Authorized ILSVRC2012 access and usage terms apply; this is not "
            "the separately distributed Chrabaszcz box-resize dataset."
        ),
        preparation_reference={
            "repository_url": IMAGENET64_EDM_REPOSITORY_URL,
            "revision": IMAGENET64_EDM_PREPARATION_REVISION,
            "path": IMAGENET64_EDM_DATASET_TOOL_PATH,
            "transform": "center-crop",
            "resolution": "64x64",
        },
    ),
}

_BACKBONES = {
    "cifar10_rfpp_config_g": ImageBackboneSpec(
        key="cifar10_rfpp_config_g",
        display_name="RF++ Config G on CIFAR-10",
        dataset_key="cifar10",
        method_key="rfpp",
        conditioning="unconditional",
        num_conditioning_classes=0,
        architecture="SongUNet+EDMPrecondVel",
        source_config_path="configs_unet/cifar10_ve_aug.json",
        source_config_identity=(
            "genode-rfpp-config-v1:f155a987a33b2bc05c699ca8e5b3f95a4e227bd4d250223a866552e0e0f8570a"
        ),
        checkpoint_filename="cifar-configG.pth",
        checkpoint_url="https://drive.google.com/file/d/14LnXZXJYJgGzMxgn72lzdl6cc6z5JPZj/view",
        checkpoint_published_size_bytes=225_666_671,
    ),
    "imagenet64_rfpp_config_e": ImageBackboneSpec(
        key="imagenet64_rfpp_config_e",
        display_name="RF++ Config E on ImageNet-64",
        dataset_key="imagenet64",
        method_key="rfpp",
        conditioning="class_conditional",
        num_conditioning_classes=1000,
        architecture="DhariwalUNet+EDMPrecondVel",
        source_config_path="configs_unet/imagenet64.json",
        source_config_identity=(
            "genode-rfpp-config-v1:8307e13bdbd66c82cdc9f62175a5b972808506cf7f7030a1c39ad38b0ea4fdd7"
        ),
        checkpoint_filename="imagenet-configE.pth",
        checkpoint_url="https://drive.google.com/file/d/13uzQWuUOOBij2vG4HV5MQSvsPMcB9Ojx/view",
        checkpoint_published_size_bytes=1_183_698_031,
    ),
    "cifar10_edm_ve_as_1rf": ImageBackboneSpec(
        key="cifar10_edm_ve_as_1rf",
        display_name="EDM VE interpreted as 1-RF on CIFAR-10",
        dataset_key="cifar10",
        method_key="edm_ve_as_1rf",
        conditioning="unconditional",
        num_conditioning_classes=0,
        architecture="SongUNet+EDMPrecondVel",
        source_config_path="configs_unet/cifar10_ve_aug.json",
        source_config_identity=(
            "genode-rfpp-config-v1:f155a987a33b2bc05c699ca8e5b3f95a4e227bd4d250223a866552e0e0f8570a"
        ),
        checkpoint_filename="edm_cifar_ve_uncond.pth",
        checkpoint_url="https://drive.google.com/file/d/18gNlRw4rUzUU_HPv-MTlj0k5WqnZE7yn/view",
        checkpoint_published_size_bytes=225_744_545,
    ),
    "imagenet64_edm_ve_as_1rf": ImageBackboneSpec(
        key="imagenet64_edm_ve_as_1rf",
        display_name="EDM VE interpreted as 1-RF on ImageNet-64",
        dataset_key="imagenet64",
        method_key="edm_ve_as_1rf",
        conditioning="class_conditional",
        num_conditioning_classes=1000,
        architecture="DhariwalUNet+EDMPrecondVel",
        source_config_path="configs_unet/imagenet64.json",
        source_config_identity=(
            "genode-rfpp-config-v1:8307e13bdbd66c82cdc9f62175a5b972808506cf7f7030a1c39ad38b0ea4fdd7"
        ),
        checkpoint_filename="edm_imagenet64_ve_cond.pth",
        checkpoint_url="https://drive.google.com/file/d/1DGDKqlkA3-YKJiw7-aimA-K2bLUxld44/view",
        checkpoint_published_size_bytes=1_183_785_741,
    ),
}

IMAGE_DATASET_REGISTRY: Mapping[str, ImageDatasetSpec] = MappingProxyType(_DATASETS)
IMAGE_BACKBONE_REGISTRY: Mapping[str, ImageBackboneSpec] = MappingProxyType(_BACKBONES)


def get_image_dataset_spec(key: str) -> ImageDatasetSpec:
    try:
        return IMAGE_DATASET_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(IMAGE_DATASET_REGISTRY))
        raise ValueError(f"Unknown image dataset key {key!r}; expected one of: {available}.") from exc


def get_image_backbone_spec(key: str) -> ImageBackboneSpec:
    try:
        return IMAGE_BACKBONE_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(IMAGE_BACKBONE_REGISTRY))
        raise ValueError(f"Unknown image backbone key {key!r}; expected one of: {available}.") from exc


__all__ = [
    "ConditioningMode",
    "IMAGE_BACKBONE_REGISTRY",
    "IMAGE_DATASET_REGISTRY",
    "ImageBackboneSpec",
    "ImageDatasetSpec",
    "IMAGENET64_EDM_DATASET_TOOL_PATH",
    "IMAGENET64_EDM_PREPARATION_REVISION",
    "IMAGENET64_EDM_REPOSITORY_URL",
    "RFPP_PINNED_REVISION",
    "RFPP_CONFIG_IDENTITY_NAMESPACE",
    "RFPP_REPOSITORY_URL",
    "get_image_backbone_spec",
    "get_image_dataset_spec",
]
