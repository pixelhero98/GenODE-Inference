from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral
from types import MappingProxyType
from typing import Any

from genode.artifacts.identity import semantic_sha256
from genode.benchmarks.image.feature_protocol import (
    TORCH_FIDELITY_DISTRIBUTION,
    TORCH_FIDELITY_FEATURE_EXTRACTOR,
    TORCH_FIDELITY_FEATURE_LAYER,
    TORCH_FIDELITY_VERSION,
)
from genode.schedule_transfer.reference_clocks import (
    DEFAULT_REFERENCE_CLOCK_KEYS,
    reference_clock_keys,
    reference_clock_provenance,
)
from genode.schedules.fixed import FIXED_SCHEDULE_TARGET_NFES
from genode.schedules.specification import ScheduleSpecification

IMAGE_PROTOCOL_VERSION = 6
IMAGE_PROTOCOL_KEY = "image_euler_248_v6"
IMAGE_GICO_TEACHER_SCORE_WEIGHT = 0.01
IMAGE_GICO_TEACHER_SCORE_WARMUP_FRACTION = 0.60
IMAGE_GICO_TEACHER_SCORE_CLIP = 5.0

CIFAR10_DATASET_KEY = "cifar10"
IMAGENET64_DATASET_KEY = "imagenet64"
IMAGE_DATASET_KEYS: tuple[str, ...] = (
    CIFAR10_DATASET_KEY,
    IMAGENET64_DATASET_KEY,
)

IMAGE_SOLVER_KEY = "euler"
IMAGE_TARGET_NFES: tuple[int, ...] = FIXED_SCHEDULE_TARGET_NFES
IMAGE_SCHEDULE_KEYS: tuple[str, ...] = DEFAULT_REFERENCE_CLOCK_KEYS

KID_BLOCK_SIZE = 1_000
KID_REWARD_TRAIN_BLOCKS = 20
KID_SELECTION_SCREENING_BLOCKS = 10
KID_SURVIVOR_CONFIRMATION_BLOCKS = 10
KID_SCREENING_SURVIVORS_PER_NFE = 3
KID_SELECTION_RULE_KEY = "top3_unique_grid_then_unique_confirmation_minimum"

CONDITIONAL_KID_REWARD_TRAIN_BLOCKS = 64
CONDITIONAL_KID_SELECTION_SCREENING_BLOCKS = 32
CONDITIONAL_KID_SURVIVOR_CONFIRMATION_BLOCKS = 64

LOCKED_SAMPLE_COUNT = 50_000
LOCKED_INCEPTION_SPLITS = 10
LOCKED_PRECISION_RECALL_NEIGHBORHOOD = 3
LOCKED_PRECISION_RECALL_BATCH_SIZE = 10_000
LOCKED_METRIC_EXECUTION_PROTOCOL = "image_locked_metric_execution_v1"
FID_COVARIANCE_EPSILON = 1e-6
FID_COMPLEX_IMAGINARY_TOLERANCE = 1e-3
FID_NEGATIVE_RELATIVE_TOLERANCE = 1e-8
LOCKED_TORCH_FIDELITY_FIXED_OPTIONS: Mapping[str, object] = MappingProxyType(
    {
        "isc": True,
        "fid": False,
        "kid": False,
        "prc": True,
        "isc_splits": LOCKED_INCEPTION_SPLITS,
        "prc_neighborhood": LOCKED_PRECISION_RECALL_NEIGHBORHOOD,
        "prc_batch_size": LOCKED_PRECISION_RECALL_BATCH_SIZE,
        "samples_shuffle": False,
        "feature_extractor": TORCH_FIDELITY_FEATURE_EXTRACTOR,
        "feature_layer_isc": "logits_unbiased",
        "feature_layer_fid": TORCH_FIDELITY_FEATURE_LAYER,
        "feature_layer_kid": TORCH_FIDELITY_FEATURE_LAYER,
        "feature_layer_prc": TORCH_FIDELITY_FEATURE_LAYER,
        "feature_extractor_internal_dtype": "float32",
        "feature_extractor_compile": False,
        "save_cpu_ram": True,
        "cache": False,
        "datasets_download": False,
        "verbose": False,
    }
)

PANEL_PHASE_REWARD_TRAIN = "reward_train"
PANEL_PHASE_SELECTION_SCREENING = "selection_screening"
PANEL_PHASE_SURVIVOR_CONFIRMATION = "survivor_confirmation"
PANEL_PHASE_LOCKED_MAIN = "locked_main"
PANEL_PHASE_CONDITIONAL_REWARD_TRAIN = "conditional_reward_train"
PANEL_PHASE_CONDITIONAL_SELECTION_SCREENING = "conditional_selection_screening"
PANEL_PHASE_CONDITIONAL_SURVIVOR_CONFIRMATION = "conditional_survivor_confirmation"
PANEL_PHASES: tuple[str, ...] = (
    PANEL_PHASE_REWARD_TRAIN,
    PANEL_PHASE_SELECTION_SCREENING,
    PANEL_PHASE_SURVIVOR_CONFIRMATION,
    PANEL_PHASE_LOCKED_MAIN,
    PANEL_PHASE_CONDITIONAL_REWARD_TRAIN,
    PANEL_PHASE_CONDITIONAL_SELECTION_SCREENING,
    PANEL_PHASE_CONDITIONAL_SURVIVOR_CONFIRMATION,
)

_PANEL_SHAPES: Mapping[str, tuple[int, int]] = {
    PANEL_PHASE_REWARD_TRAIN: (KID_REWARD_TRAIN_BLOCKS, KID_BLOCK_SIZE),
    PANEL_PHASE_SELECTION_SCREENING: (
        KID_SELECTION_SCREENING_BLOCKS,
        KID_BLOCK_SIZE,
    ),
    PANEL_PHASE_SURVIVOR_CONFIRMATION: (
        KID_SURVIVOR_CONFIRMATION_BLOCKS,
        KID_BLOCK_SIZE,
    ),
    PANEL_PHASE_LOCKED_MAIN: (LOCKED_SAMPLE_COUNT // KID_BLOCK_SIZE, KID_BLOCK_SIZE),
    PANEL_PHASE_CONDITIONAL_REWARD_TRAIN: (
        CONDITIONAL_KID_REWARD_TRAIN_BLOCKS,
        KID_BLOCK_SIZE,
    ),
    PANEL_PHASE_CONDITIONAL_SELECTION_SCREENING: (
        CONDITIONAL_KID_SELECTION_SCREENING_BLOCKS,
        KID_BLOCK_SIZE,
    ),
    PANEL_PHASE_CONDITIONAL_SURVIVOR_CONFIRMATION: (
        CONDITIONAL_KID_SURVIVOR_CONFIRMATION_BLOCKS,
        KID_BLOCK_SIZE,
    ),
}


@dataclass(frozen=True)
class ImageBenchmarkSpec:
    key: str
    resolution: int
    class_count: int
    conditioning: str

    @property
    def is_class_conditional(self) -> bool:
        return self.class_count > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "resolution": int(self.resolution),
            "class_count": int(self.class_count),
            "conditioning": self.conditioning,
        }


_BENCHMARK_SPECS: Mapping[str, ImageBenchmarkSpec] = {
    CIFAR10_DATASET_KEY: ImageBenchmarkSpec(
        key=CIFAR10_DATASET_KEY,
        resolution=32,
        class_count=0,
        conditioning="unconditional",
    ),
    IMAGENET64_DATASET_KEY: ImageBenchmarkSpec(
        key=IMAGENET64_DATASET_KEY,
        resolution=64,
        class_count=1_000,
        conditioning="balanced_class_conditional",
    ),
}


def image_benchmark_spec(dataset_key: str) -> ImageBenchmarkSpec:
    key = str(dataset_key).strip().lower()
    try:
        return _BENCHMARK_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported image dataset {dataset_key!r}; expected one of {IMAGE_DATASET_KEYS}.") from exc


def normalize_image_solver(value: str) -> str:
    key = str(value).strip().lower()
    if key != IMAGE_SOLVER_KEY:
        raise ValueError(f"The image protocol is Euler-only; got solver_key={value!r}.")
    return key


def normalize_image_nfe(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Image target_nfe must be one of {IMAGE_TARGET_NFES}, got {value!r}.")
    parsed = int(value)
    if parsed not in IMAGE_TARGET_NFES:
        raise ValueError(
            f"Image target_nfe must be one of {IMAGE_TARGET_NFES}; "
            f"unseen-NFE evaluation is not part of this protocol, got {parsed}."
        )
    return parsed


def panel_shape(phase: str) -> tuple[int, int]:
    key = str(phase).strip().lower()
    try:
        return _PANEL_SHAPES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown image sample-panel phase {phase!r}; expected one of {PANEL_PHASES}.") from exc


@dataclass(frozen=True)
class EulerImageWorkload:
    pair_count: int
    evidence_images: int
    backbone_image_evaluations: int
    survivor_confirmation_images: int = 0
    survivor_confirmation_backbone_evaluations: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "solver_key": IMAGE_SOLVER_KEY,
            "pair_count": int(self.pair_count),
            "evidence_images": int(self.evidence_images),
            "backbone_image_evaluations": int(self.backbone_image_evaluations),
            "survivor_confirmation_images": int(self.survivor_confirmation_images),
            "survivor_confirmation_backbone_evaluations": int(self.survivor_confirmation_backbone_evaluations),
        }


def image_schedule_keys(
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> tuple[str, ...]:
    return reference_clock_keys(extra_late_p_values)


def euler_image_workload(
    *,
    pair_count: int = 1,
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> EulerImageWorkload:
    if isinstance(pair_count, bool) or not isinstance(pair_count, Integral):
        raise ValueError("pair_count must be a positive integer.")
    pairs = int(pair_count)
    if pairs <= 0:
        raise ValueError("pair_count must be a positive integer.")
    schedule_count = len(image_schedule_keys(extra_late_p_values))
    images_per_density_cell = (KID_REWARD_TRAIN_BLOCKS + KID_SELECTION_SCREENING_BLOCKS) * KID_BLOCK_SIZE
    per_pair_images = schedule_count * len(IMAGE_TARGET_NFES) * images_per_density_cell
    per_pair_evaluations = schedule_count * images_per_density_cell * sum(IMAGE_TARGET_NFES)
    return EulerImageWorkload(
        pair_count=pairs,
        evidence_images=pairs * per_pair_images,
        backbone_image_evaluations=pairs * per_pair_evaluations,
    )


def survivor_confirmation_workload(
    survivors_by_nfe: Mapping[int, int],
    *,
    pair_count: int = 1,
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> EulerImageWorkload:
    base = euler_image_workload(
        pair_count=pair_count,
        extra_late_p_values=extra_late_p_values,
    )
    normalized: dict[int, int] = dict.fromkeys(IMAGE_TARGET_NFES, 0)
    for raw_nfe, raw_count in survivors_by_nfe.items():
        nfe = normalize_image_nfe(raw_nfe)
        if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
            raise ValueError("Survivor counts must be nonnegative integers.")
        count = int(raw_count)
        if count < 0:
            raise ValueError("Survivor counts must be nonnegative integers.")
        normalized[nfe] += count
    samples_per_survivor = KID_SURVIVOR_CONFIRMATION_BLOCKS * KID_BLOCK_SIZE
    confirmation_images = samples_per_survivor * sum(normalized.values())
    confirmation_evaluations = samples_per_survivor * sum(nfe * count for nfe, count in normalized.items())
    return EulerImageWorkload(
        pair_count=base.pair_count,
        evidence_images=base.evidence_images,
        backbone_image_evaluations=base.backbone_image_evaluations,
        survivor_confirmation_images=base.pair_count * confirmation_images,
        survivor_confirmation_backbone_evaluations=(base.pair_count * confirmation_evaluations),
    )


def image_protocol_metadata(
    *,
    extra_late_p_values: str | Sequence[Decimal | float | int | str] = (),
) -> dict[str, Any]:
    schedule_keys = image_schedule_keys(extra_late_p_values)
    workload = euler_image_workload(
        extra_late_p_values=extra_late_p_values,
    )
    metadata: dict[str, Any] = {
        "protocol_key": IMAGE_PROTOCOL_KEY,
        "protocol_version": IMAGE_PROTOCOL_VERSION,
        "solver_key": IMAGE_SOLVER_KEY,
        "target_nfes": list(IMAGE_TARGET_NFES),
        "unseen_nfe_evaluation": False,
        "schedule_keys": list(schedule_keys),
        "schedule_count": len(schedule_keys),
        "schedule_specifications": [ScheduleSpecification(key).as_payload() for key in schedule_keys],
        "reference_clock_provenance": [reference_clock_provenance(key) for key in schedule_keys],
        "kid": {
            "reward_metric": "kernel_inception_distance",
            "estimator": "unbiased_mmd2",
            "kernel": "polynomial_degree_3",
            "block_size": KID_BLOCK_SIZE,
            "reward_train_blocks": KID_REWARD_TRAIN_BLOCKS,
            "selection_screening_blocks": KID_SELECTION_SCREENING_BLOCKS,
            "survivor_confirmation_blocks": KID_SURVIVOR_CONFIRMATION_BLOCKS,
            "reward_direction": "lower_is_better",
            "reward_transform": "signed_robust_uniform_advantage",
        },
        "locked_metrics": {
            "sample_count": LOCKED_SAMPLE_COUNT,
            "fid": "fid50k",
            "inception_score_splits": LOCKED_INCEPTION_SPLITS,
            "precision_recall_neighborhood": (LOCKED_PRECISION_RECALL_NEIGHBORHOOD),
            "precision_recall_batch_size": (LOCKED_PRECISION_RECALL_BATCH_SIZE),
            "same_generated_panel": True,
            "execution": locked_metric_execution_spec(),
        },
        "selection": {
            "rule_key": KID_SELECTION_RULE_KEY,
            "screening_survivors_per_nfe": (KID_SCREENING_SURVIVORS_PER_NFE),
            "duplicate_handling": "group_by_exact_time_grid_sha256",
            "screening_statistic": "mean_unbiased_block_kid",
            "screening_direction": "lower_is_better",
            "confirmation_statistic": "mean_unbiased_block_kid",
            "confirmation_direction": "lower_is_better",
            "tie_policy": "abort",
            "locked_tuning": False,
        },
        "gico_student": {
            "primary_target": "kid_reward_soft_density_mixture",
            "objective": "target_to_policy_kl_minus_teacher_score",
            "teacher_evidence_phase": "reward_train",
            "teacher_score_weight": IMAGE_GICO_TEACHER_SCORE_WEIGHT,
            "teacher_score_schedule": "zero_then_linear_late_ramp",
            "teacher_score_warmup_fraction": (IMAGE_GICO_TEACHER_SCORE_WARMUP_FRACTION),
            "teacher_score_clip": IMAGE_GICO_TEACHER_SCORE_CLIP,
            "unseen_nfe_distillation": False,
        },
        "datasets": {key: image_benchmark_spec(key).as_dict() for key in IMAGE_DATASET_KEYS},
        "workload_per_dataset_checkpoint_pair": workload.as_dict(),
    }
    metadata["protocol_sha256"] = semantic_sha256(
        metadata,
        namespace="image-benchmark-protocol",
    )
    return metadata


def locked_metric_execution_spec() -> dict[str, Any]:
    """Return the complete report-bound numerical metric contract."""

    payload: dict[str, Any] = {
        "protocol": LOCKED_METRIC_EXECUTION_PROTOCOL,
        "backend": {
            "distribution": TORCH_FIDELITY_DISTRIBUTION,
            "version": TORCH_FIDELITY_VERSION,
        },
        "dynamic_bindings": {
            "input1": "locked_generated_metric_dataset",
            "input2": "bound_real_metric_dataset",
            "cuda": "execution_environment.device_type",
            "batch_size": ("fid_reference.feature_extraction_batch_size"),
            "rng_seed": "locked_sample_panel.seed_start",
            "feature_extractor_weights_path": ("verified_feature_weights.path"),
        },
        "torch_fidelity_fixed_options": dict(LOCKED_TORCH_FIDELITY_FIXED_OPTIONS),
        "fid50k": {
            "generated_moments": ("ordered_fixed_1024_row_float64_chunks"),
            "real_moments": "bound_full_training_fid_reference",
            "distance": ("scipy_sqrtm_of_covariance_product_trace"),
            "covariance_epsilon": FID_COVARIANCE_EPSILON,
            "fallback": ("add_epsilon_identity_to_both_covariances"),
            "complex_imaginary_tolerance": (FID_COMPLEX_IMAGINARY_TOLERANCE),
            "negative_relative_tolerance": (FID_NEGATIVE_RELATIVE_TOLERANCE),
            "negative_within_tolerance": "clamp_to_zero",
        },
    }
    payload["spec_sha256"] = semantic_sha256(
        payload,
        namespace="image-locked-metric-execution",
    )
    return payload


def finite_temperature(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("temperature must be a finite positive number.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError("temperature must be a finite positive number.")
    return parsed


__all__ = [
    "CIFAR10_DATASET_KEY",
    "CONDITIONAL_KID_REWARD_TRAIN_BLOCKS",
    "CONDITIONAL_KID_SELECTION_SCREENING_BLOCKS",
    "CONDITIONAL_KID_SURVIVOR_CONFIRMATION_BLOCKS",
    "EulerImageWorkload",
    "FID_COMPLEX_IMAGINARY_TOLERANCE",
    "FID_COVARIANCE_EPSILON",
    "FID_NEGATIVE_RELATIVE_TOLERANCE",
    "IMAGE_DATASET_KEYS",
    "IMAGE_PROTOCOL_KEY",
    "IMAGE_PROTOCOL_VERSION",
    "IMAGE_GICO_TEACHER_SCORE_WEIGHT",
    "IMAGE_GICO_TEACHER_SCORE_WARMUP_FRACTION",
    "IMAGE_GICO_TEACHER_SCORE_CLIP",
    "IMAGE_SCHEDULE_KEYS",
    "IMAGE_SOLVER_KEY",
    "IMAGE_TARGET_NFES",
    "IMAGENET64_DATASET_KEY",
    "ImageBenchmarkSpec",
    "KID_BLOCK_SIZE",
    "KID_REWARD_TRAIN_BLOCKS",
    "KID_SCREENING_SURVIVORS_PER_NFE",
    "KID_SELECTION_RULE_KEY",
    "KID_SELECTION_SCREENING_BLOCKS",
    "KID_SURVIVOR_CONFIRMATION_BLOCKS",
    "LOCKED_INCEPTION_SPLITS",
    "LOCKED_METRIC_EXECUTION_PROTOCOL",
    "LOCKED_PRECISION_RECALL_NEIGHBORHOOD",
    "LOCKED_PRECISION_RECALL_BATCH_SIZE",
    "LOCKED_SAMPLE_COUNT",
    "LOCKED_TORCH_FIDELITY_FIXED_OPTIONS",
    "PANEL_PHASE_LOCKED_MAIN",
    "PANEL_PHASE_CONDITIONAL_REWARD_TRAIN",
    "PANEL_PHASE_CONDITIONAL_SELECTION_SCREENING",
    "PANEL_PHASE_CONDITIONAL_SURVIVOR_CONFIRMATION",
    "PANEL_PHASE_REWARD_TRAIN",
    "PANEL_PHASE_SELECTION_SCREENING",
    "PANEL_PHASE_SURVIVOR_CONFIRMATION",
    "PANEL_PHASES",
    "euler_image_workload",
    "finite_temperature",
    "image_benchmark_spec",
    "image_protocol_metadata",
    "image_schedule_keys",
    "locked_metric_execution_spec",
    "normalize_image_nfe",
    "normalize_image_solver",
    "panel_shape",
    "survivor_confirmation_workload",
]
