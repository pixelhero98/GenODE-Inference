from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Mapping

from genode.models.config import OTFlowConfig
from genode.data.otflow_datasets import (
    LOBSTER_SYNTHETIC_DATASET_KEY,
    build_dataset_splits_from_cryptos,
    build_dataset_splits_from_lobster_synthetic,
)
from genode.data.otflow_medical_constants import (
    LONG_TERM_ST_DATASET_KEY,
    LONG_TERM_ST_STRIDE,
    LONG_TERM_ST_HISTORY_LEN,
    LONG_TERM_ST_HORIZON_LEN,
)
from genode.data.otflow_paths import cryptos_data_path, lobster_synthetic_profile_path, long_term_st_data_path

OTFLOW_REFERENCE_DATASET_CHOICES = ("cryptos", LOBSTER_SYNTHETIC_DATASET_KEY, LONG_TERM_ST_DATASET_KEY)
OTFLOW_REFERENCE_BACKBONE_PRESETS: Mapping[str, Mapping[str, object]] = {
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
        "rollout_mode": "non_ar",
        "future_block_len": 128,
    },
    LOBSTER_SYNTHETIC_DATASET_KEY: {
        "levels": 10,
        "token_dim": 4,
        "history_len": 256,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": True,
        "use_time_gaps": False,
        "rollout_mode": "non_ar",
        "future_block_len": 128,
    },
    LONG_TERM_ST_DATASET_KEY: {
        "levels": 1,
        "token_dim": 1,
        "history_len": LONG_TERM_ST_HISTORY_LEN,
        "ctx_encoder": "hybrid",
        "ctx_causal": True,
        "ctx_local_kernel": 7,
        "ctx_pool_scales": "8,32",
        "use_time_features": False,
        "use_time_gaps": False,
        "use_cond_features": False,
        "cond_standardize": False,
        "rollout_mode": "non_ar",
        "future_block_len": LONG_TERM_ST_HORIZON_LEN,
    },
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
        horizons=(128, 128, 128),
        history_options=(256,),
        nfe_options=(1, 2),
        train_steps_tune=6000,
        train_steps_final=12000,
        eval_windows_tune=10,
        eval_windows_final=20,
    ),
    LOBSTER_SYNTHETIC_DATASET_KEY: DatasetPlan(
        name=LOBSTER_SYNTHETIC_DATASET_KEY,
        dataset=LOBSTER_SYNTHETIC_DATASET_KEY,
        levels=10,
        horizons=(128, 128, 128),
        history_options=(256,),
        nfe_options=(1,),
        train_steps_tune=6000,
        train_steps_final=12000,
        eval_windows_tune=10,
        eval_windows_final=20,
        data_path=lobster_synthetic_profile_path(),
    ),
    LONG_TERM_ST_DATASET_KEY: DatasetPlan(
        name=LONG_TERM_ST_DATASET_KEY,
        dataset=LONG_TERM_ST_DATASET_KEY,
        levels=1,
        horizons=(LONG_TERM_ST_HORIZON_LEN, LONG_TERM_ST_HORIZON_LEN, LONG_TERM_ST_HORIZON_LEN),
        history_options=(LONG_TERM_ST_HISTORY_LEN,),
        nfe_options=(1,),
        train_steps_tune=4000,
        train_steps_final=12000,
        eval_windows_tune=4,
        eval_windows_final=8,
        data_path=long_term_st_data_path(),
        stride_train=LONG_TERM_ST_STRIDE,
        stride_eval=LONG_TERM_ST_STRIDE,
        batch_size=2,
    ),
}

def get_otflow_reference_backbone_preset(dataset: str) -> Dict[str, object]:
    dataset_key = str(dataset).strip()
    if dataset_key not in OTFLOW_REFERENCE_BACKBONE_PRESETS:
        raise ValueError(f"No OTFlow reference preset defined for dataset={dataset!r}")
    return copy.deepcopy(dict(OTFLOW_REFERENCE_BACKBONE_PRESETS[dataset_key]))


def build_dataset_splits(args, cfg: OTFlowConfig):
    dataset = args.dataset
    if dataset == "cryptos":
        return build_dataset_splits_from_cryptos(
            path=args.data_path or cryptos_data_path(),
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == LOBSTER_SYNTHETIC_DATASET_KEY:
        return build_dataset_splits_from_lobster_synthetic(
            profile_path=args.data_path or lobster_synthetic_profile_path(),
            cfg=cfg,
            length=args.synthetic_length,
            seed=args.seed,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    if dataset == LONG_TERM_ST_DATASET_KEY:
        from genode.data.otflow_medical_datasets import build_dataset_splits_from_long_term_st

        return build_dataset_splits_from_long_term_st(
            path=args.data_path or long_term_st_data_path(),
            cfg=cfg,
            stride_train=args.stride_train,
            stride_eval=args.stride_eval,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
    raise ValueError(f"Unknown dataset={dataset}")


__all__ = [
    "OTFLOW_REFERENCE_DATASET_CHOICES",
    "OTFLOW_REFERENCE_BACKBONE_PRESETS",
    "DATASET_PLANS",
    "DatasetPlan",
    "get_otflow_reference_backbone_preset",
    "build_dataset_splits",
]
