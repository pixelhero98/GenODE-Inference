from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping

import torch

from genode.models.config import OTFlowConfig
from genode.data.otflow_datasets import (
    DEFAULT_SYNTHETIC_LENGTH,
    build_dataset_splits_from_cryptos,
    build_dataset_splits_from_es_mbp_10,
    build_dataset_splits_from_npz_l2,
    build_dataset_splits_from_optiver,
    build_dataset_splits_synthetic,
    default_cryptos_npz_path,
    default_es_mbp_10_npz_path,
    default_optiver_npz_path,
)
from genode.data.otflow_medical_datasets import (
    SLEEP_EDF_DATASET_KEY,
    build_dataset_splits_from_sleep_edf,
    default_sleep_edf_data_path,
)

DATASET_CHOICES = ("synthetic", "npz_l2", "optiver", "cryptos", "es_mbp_10", SLEEP_EDF_DATASET_KEY)
OTFLOW_PAPER_DATASET_CHOICES = ("cryptos", "es_mbp_10", SLEEP_EDF_DATASET_KEY)
OTFLOW_PAPER_BACKBONE_PRESETS: Mapping[str, Mapping[str, object]] = {
    "cryptos": {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": True,
        "use_time_gaps": False,
    },
    "es_mbp_10": {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": True,
        "use_time_gaps": False,
    },
    SLEEP_EDF_DATASET_KEY: {
        "levels": 1,
        "token_dim": 3,
        "history_len": 12_000,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": False,
        "use_time_gaps": False,
        "use_cond_features": True,
        "cond_standardize": False,
        "rollout_mode": "non_ar",
        "future_block_len": 3_000,
    },
}
OTFLOW_PAPER_BACKBONE_PRESET: Mapping[str, object] = OTFLOW_PAPER_BACKBONE_PRESETS["cryptos"]

OTFLOW_QUALITY_PRESETS: Mapping[str, Dict[str, object]] = {
    "synthetic": {
        "levels": 10,
        "history_len": 128,
        "eval_nfe": 2,
        "solver": "euler",
        "ctx_encoder": "transformer",
        "ctx_causal": True,
        "ctx_local_kernel": 5,
        "ctx_pool_scales": "4,16",
    },
    "optiver": {
        "levels": 2,
        "history_len": 128,
        "eval_nfe": 4,
        "solver": "dpmpp2m",
        "ctx_encoder": "transformer",
        "ctx_causal": True,
        "ctx_local_kernel": 5,
        "ctx_pool_scales": "4,16",
    },
    "cryptos": {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "eval_nfe": 1,
        "solver": "dpmpp2m",
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
    },
    "es_mbp_10": {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "eval_nfe": 1,
        "solver": "euler",
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
    },
    SLEEP_EDF_DATASET_KEY: {
        "levels": 1,
        "token_dim": 3,
        "history_len": 12_000,
        "eval_nfe": 1,
        "solver": "euler",
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_cond_features": True,
        "cond_standardize": False,
        "use_time_features": False,
        "use_time_gaps": False,
        "rollout_mode": "non_ar",
        "future_block_len": 3_000,
    },
}

OTFLOW_SPEED_PRESETS: Mapping[str, Dict[str, object]] = {
    dataset: {
        **copy.deepcopy(dict(preset)),
        "eval_nfe": 1,
    }
    for dataset, preset in OTFLOW_QUALITY_PRESETS.items()
}

OTFLOW_PRESET_VARIANTS: Mapping[str, Mapping[str, Dict[str, object]]] = {
    "quality": OTFLOW_QUALITY_PRESETS,
    "speed": OTFLOW_SPEED_PRESETS,
}


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    dataset: str
    levels: int
    horizons: tuple[int, int, int]
    history_options: tuple[int, ...]
    nfe_options: tuple[int, ...]
    train_steps_tune: int
    train_steps_final: int
    eval_windows_tune: int
    eval_windows_final: int
    data_path: str = ""
    synthetic_length: int = 2_000_000
    train_frac: float = 0.7
    val_frac: float = 0.1
    test_frac: float = 0.2
    stride_train: int = 1
    stride_eval: int = 1
    batch_size: int = 64


DATASET_PLANS: Mapping[str, DatasetPlan] = {
    "cryptos": DatasetPlan(
        name="cryptos",
        dataset="cryptos",
        levels=10,
        horizons=(60, 300, 900),
        history_options=(256,),
        nfe_options=(1, 2),
        train_steps_tune=6000,
        train_steps_final=12000,
        eval_windows_tune=10,
        eval_windows_final=20,
    ),
    "es_mbp_10": DatasetPlan(
        name="es_mbp_10",
        dataset="es_mbp_10",
        levels=10,
        horizons=(60, 300, 900),
        history_options=(256,),
        nfe_options=(1,),
        train_steps_tune=6000,
        train_steps_final=12000,
        eval_windows_tune=10,
        eval_windows_final=20,
    ),
    SLEEP_EDF_DATASET_KEY: DatasetPlan(
        name=SLEEP_EDF_DATASET_KEY,
        dataset=SLEEP_EDF_DATASET_KEY,
        levels=1,
        horizons=(3000, 3000, 3000),
        history_options=(12000,),
        nfe_options=(1,),
        train_steps_tune=4000,
        train_steps_final=12000,
        eval_windows_tune=6,
        eval_windows_final=12,
        data_path=default_sleep_edf_data_path(),
        stride_train=3000,
        stride_eval=3000,
        batch_size=2,
    ),
}

_OPTIONAL_CFG_OVERRIDES = (
    "token_dim",
    "hidden_dim",
    "ctx_encoder",
    "ctx_causal",
    "ctx_local_kernel",
    "diffusion_steps",
    "fu_net_type",
    "fu_net_layers",
    "fu_net_heads",
    "rollout_mode",
    "future_block_len",
    "adaptive_context",
    "adaptive_context_ratio",
    "adaptive_context_min",
    "adaptive_context_max",
    "train_variable_context",
    "train_context_min",
    "train_context_max",
    "use_time_features",
    "use_time_gaps",
    "use_minibatch_ot",
    "cfg_scale",
    "solver",
    "use_amp",
    "grad_accum_steps",
)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def get_otflow_paper_backbone_preset(dataset: str) -> Dict[str, object]:
    dataset_key = str(dataset).strip()
    if dataset_key not in OTFLOW_PAPER_DATASET_CHOICES:
        raise ValueError(f"No OTFlow paper preset defined for dataset={dataset!r}")
    return copy.deepcopy(dict(OTFLOW_PAPER_BACKBONE_PRESETS[dataset_key]))


def get_otflow_dataset_preset(dataset: str, variant: str = "quality") -> Dict[str, object]:
    variant_key = str(variant).strip().lower()
    if variant_key not in OTFLOW_PRESET_VARIANTS:
        raise ValueError(f"Unknown OTFlow preset variant={variant!r}")
    dataset_key = str(dataset).strip()
    presets = OTFLOW_PRESET_VARIANTS[variant_key]
    if dataset_key not in presets:
        raise ValueError(f"No OTFlow preset defined for dataset={dataset!r}")
    return copy.deepcopy(dict(presets[dataset_key]))


def apply_otflow_dataset_preset(args, variant: str | None = None):
    dataset = getattr(args, "dataset", None)
    if dataset not in OTFLOW_QUALITY_PRESETS:
        return args
    variant_name = str(
        variant
        or getattr(args, "otflow_variant", None)
        or "quality"
    ).strip().lower()
    preset = get_otflow_dataset_preset(str(dataset), variant=variant_name)
    for key, value in preset.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) is None:
            setattr(args, key, value)
    if hasattr(args, "otflow_variant") and getattr(args, "otflow_variant") is None:
        setattr(args, "otflow_variant", variant_name)
    return args


def build_cfg_from_args(args) -> OTFlowConfig:
    cfg = OTFlowConfig()
    cfg.apply_overrides(
        device=torch.device(args.device),
        levels=args.levels,
        history_len=args.history_len,
        steps=int(getattr(args, "steps", cfg.train.steps)),
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        standardize=args.standardize,
        use_cond_features=args.use_cond_features,
        cond_standardize=args.cond_standardize,
    )

    for name in _OPTIONAL_CFG_OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            cfg.apply_overrides(**{name: value})

    ctx_pool_scales = getattr(args, "ctx_pool_scales", None)
    if ctx_pool_scales:
        cfg.apply_overrides(ctx_pool_scales=tuple(parse_int_list(ctx_pool_scales)))

    cond_depths = getattr(args, "cond_depths", "")
    if cond_depths:
        cfg.apply_overrides(cond_depths=tuple(parse_int_list(cond_depths)))

    cond_vol_window = getattr(args, "cond_vol_window", None)
    if cond_vol_window is not None:
        cfg.apply_overrides(cond_vol_window=cond_vol_window)

    return cfg


def build_dataset_splits(args, cfg: OTFlowConfig):
    dataset = args.dataset
    if dataset == "npz_l2":
        if not args.data_path:
            raise ValueError("--data_path is required when --dataset npz_l2")
        return build_dataset_splits_from_npz_l2(
            path=args.data_path,
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == "cryptos":
        return build_dataset_splits_from_cryptos(
            path=args.data_path or default_cryptos_npz_path(),
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == "optiver":
        return build_dataset_splits_from_optiver(
            path=args.data_path or default_optiver_npz_path(),
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == "es_mbp_10":
        return build_dataset_splits_from_es_mbp_10(
            path=args.data_path or default_es_mbp_10_npz_path(),
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == "synthetic":
        return build_dataset_splits_synthetic(
            cfg=cfg,
            length=args.synthetic_length,
            seed=args.seed,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == SLEEP_EDF_DATASET_KEY:
        return build_dataset_splits_from_sleep_edf(
            path=args.data_path or default_sleep_edf_data_path(),
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    raise ValueError(f"Unknown dataset={dataset}")


__all__ = [
    "DATASET_CHOICES",
    "OTFLOW_PAPER_DATASET_CHOICES",
    "OTFLOW_PAPER_BACKBONE_PRESETS",
    "OTFLOW_PAPER_BACKBONE_PRESET",
    "DATASET_PLANS",
    "DatasetPlan",
    "DEFAULT_SYNTHETIC_LENGTH",
    "OTFLOW_QUALITY_PRESETS",
    "OTFLOW_SPEED_PRESETS",
    "mkdir",
    "parse_int_list",
    "parse_float_list",
    "get_otflow_paper_backbone_preset",
    "get_otflow_dataset_preset",
    "apply_otflow_dataset_preset",
    "build_cfg_from_args",
    "build_dataset_splits",
]
