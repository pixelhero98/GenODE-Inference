"""Verified image-backbone registry, binding, adapters, and loaders."""

from .adapter import (
    IMAGE_BACKBONE_CONTEXT_DIM,
    IMAGE_BACKBONE_CONTEXT_SELECTOR,
    CanonicalNoiseToDataAdapter,
)
from .checkpoint import (
    CheckpointBinding,
    bind_checkpoint,
    validate_formal_image_checkpoint_binding,
    verify_checkpoint_binding,
)
from .loading import (
    UserSuppliedRFPPFactory,
    load_verified_image_backbone,
    verify_user_supplied_rfpp_source_root,
)
from .protocol import (
    DEFAULT_NATIVE_TIME_EPSILON,
    IMAGE_BACKBONE_PROTOCOL_SCHEMA,
    IMAGE_EVALUATION_NFES,
    IMAGE_EVALUATION_SOLVER,
    ImageBackboneManifest,
    build_image_backbone_manifest,
)
from .registry import (
    IMAGE_BACKBONE_REGISTRY,
    IMAGE_DATASET_REGISTRY,
    RFPP_CONFIG_IDENTITY_NAMESPACE,
    RFPP_PINNED_REVISION,
    RFPP_REPOSITORY_URL,
    ImageBackboneSpec,
    ImageDatasetSpec,
    get_image_backbone_spec,
    get_image_dataset_spec,
)
from .rfpp_factory import (
    RandomRFPPUNetBuild,
    build_rfpp_native_model,
    build_rfpp_random_unet,
    verify_rfpp_source_configuration,
)

__all__ = [
    "CanonicalNoiseToDataAdapter",
    "CheckpointBinding",
    "DEFAULT_NATIVE_TIME_EPSILON",
    "IMAGE_BACKBONE_CONTEXT_DIM",
    "IMAGE_BACKBONE_CONTEXT_SELECTOR",
    "IMAGE_BACKBONE_PROTOCOL_SCHEMA",
    "IMAGE_BACKBONE_REGISTRY",
    "IMAGE_DATASET_REGISTRY",
    "IMAGE_EVALUATION_NFES",
    "IMAGE_EVALUATION_SOLVER",
    "ImageBackboneManifest",
    "ImageBackboneSpec",
    "ImageDatasetSpec",
    "RFPP_CONFIG_IDENTITY_NAMESPACE",
    "RFPP_PINNED_REVISION",
    "RFPP_REPOSITORY_URL",
    "RandomRFPPUNetBuild",
    "UserSuppliedRFPPFactory",
    "bind_checkpoint",
    "build_image_backbone_manifest",
    "build_rfpp_native_model",
    "build_rfpp_random_unet",
    "get_image_backbone_spec",
    "get_image_dataset_spec",
    "load_verified_image_backbone",
    "validate_formal_image_checkpoint_binding",
    "verify_checkpoint_binding",
    "verify_rfpp_source_configuration",
    "verify_user_supplied_rfpp_source_root",
]
