from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from genode.data.experiment_common import DATASET_PLANS, build_dataset_splits, get_otflow_paper_backbone_preset
from genode.evaluation.fm_backbone_registry import (
    BACKBONE_NAME_OTFLOW,
    build_backbone_checkpoint_id,
    find_backbone_artifact,
    load_backbone_manifest,
    train_budget_label,
)
from genode.data.otflow_experiment_plan import (
    CANONICAL_CONDITIONAL_GENERATION_PAPER_DATASETS,
    CANONICAL_FORECAST_PAPER_DATASETS,
    experiment_plan_by_key,
)
from genode.data.otflow_forecast_data import build_monash_forecast_splits
from genode.data.otflow_medical_datasets import SLEEP_EDF_DATASET_KEY
from genode.models.otflow_model import OTFlow
from genode.data.otflow_paths import (
    default_cryptos_data_path,
    default_es_mbp_10_data_path,
    default_sleep_edf_data_path,
    project_results_root,
    resolve_project_path,
)
from genode.evaluation.otflow_sampling_support import _apply_sample_overrides, _restore_sample_overrides
from genode.runtime import ProgressBar
from genode.schedule_transfer.otflow_schedule_diagnostics import _collect_calibration
from genode.schedule_transfer.otflow_signal_traces import (
    NATIVE_INFO_GROWTH_TRACE_KEY,
    compute_info_growth_hardness_numpy,
    resolved_info_growth_scale,
)
from genode.models.otflow_train_val import save_json, seed_all

FORECAST_FAMILY = "forecast_extrapolation"
CONDITIONAL_GENERATION_FAMILY = "conditional_generation"
VALIDATION_PHASE = "validation_tuning"
LOCKED_TEST_PHASE = "locked_test"
TRAIN_TUNING_PHASE = "train_tuning"

TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION = "train_window_fraction"
TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED = "validation_normalized"
TRAIN_TUNING_SAMPLING_MODES = (
    TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
)
TRAIN_TUNING_SAMPLER_WINDOW_FRACTION_V1 = "temporal_stratified_hash_v1"
TRAIN_TUNING_SAMPLER_VALIDATION_NORMALIZED_V2 = "temporal_stratified_validation_normalized_v2"
DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION = 0.70
DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION = 0.10

UNIFORM_SCHEDULER_KEY = "uniform"
DEFAULT_SIGNAL_TRACE_KEY = NATIVE_INFO_GROWTH_TRACE_KEY
DEFAULT_SHARED_BACKBONE_ROOT = (
    project_results_root() / "shared_backbones" / "otflow_fullhorizon_seed0"
)
DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE = "transformer"
DEFAULT_CONDITIONAL_GENERATION_TRAIN_STEPS = 20_000
CONDITIONAL_GENERATION_PHYSICAL_BATCH_SIZE_BY_DATASET: Dict[str, int] = {
    "cryptos": 8,
    "es_mbp_10": 8,
    SLEEP_EDF_DATASET_KEY: 2,
}
IGNORED_BASELINE_MODEL_CONFIG_KEYS = {
    "baseline_latent_dim",
    "vae_kl_weight",
    "timegan_supervision_weight",
    "timegan_moment_weight",
    "kovae_pred_weight",
    "kovae_ridge",
    "gan_noise_dim",
    "cgan_recon_weight",
}

DEFAULT_FORECAST_DATASETS = tuple(CANONICAL_FORECAST_PAPER_DATASETS)
SUPPORTED_FORECAST_DATASETS = tuple(CANONICAL_FORECAST_PAPER_DATASETS)
DEFAULT_CONDITIONAL_GENERATION_DATASETS = tuple(CANONICAL_CONDITIONAL_GENERATION_PAPER_DATASETS)
ALL_SOLVER_ORDER: Tuple[str, ...] = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
SOLVER_RUNTIME_NAMES: Dict[str, str] = {
    "euler": "euler",
    "heun": "heun",
    "midpoint_rk2": "midpoint_rk2",
    "dpmpp2m": "dpmpp2m",
}

def _empirical_crps(samples: np.ndarray, target: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    term1 = np.mean(np.abs(samples - target[None, :]), axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))
    return float(np.mean(term1 - term2))


def _point_mase(prediction: np.ndarray, target: np.ndarray, scale: float) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    safe_scale = max(float(scale), 1e-12)
    return float(np.mean(np.abs(prediction - target)) / safe_scale)


def parse_csv(text: str) -> List[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_int_csv(text: str) -> List[int]:
    return [int(part) for part in parse_csv(text)]


def parse_float_csv(text: str) -> List[float]:
    return [float(part) for part in parse_csv(text)]


def parse_forecast_datasets(text: str) -> List[str]:
    names = parse_csv(text)
    unknown = [name for name in names if name not in SUPPORTED_FORECAST_DATASETS]
    if unknown:
        raise ValueError(f"Unknown forecast datasets: {unknown}")
    return names


def parse_conditional_generation_datasets(text: str) -> List[str]:
    names = parse_csv(text)
    unknown = [name for name in names if name not in DEFAULT_CONDITIONAL_GENERATION_DATASETS]
    if unknown:
        raise ValueError(f"Unknown conditional-generation datasets: {unknown}")
    return names


def _forecast_time_feature_mode(cfg) -> str:
    use_elapsed = bool(getattr(cfg.model, "use_time_features", False))
    use_gap_only = bool(getattr(cfg.model, "use_time_gaps", False))
    if use_elapsed and use_gap_only:
        raise ValueError("Time features must use exactly one mode: none, gap_only, or gap_elapsed.")
    if use_elapsed:
        return "gap_elapsed"
    if use_gap_only:
        return "gap_only"
    return "none"


def choose_forecast_example_indices(ds, n_examples: int, seed: int) -> np.ndarray:
    total = int(len(ds))
    if total <= 0:
        raise ValueError("Forecast evaluation split is empty.")
    requested = int(n_examples)
    if requested <= 0 or requested >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.asarray(rng.choice(np.arange(total, dtype=np.int64), size=requested, replace=False), dtype=np.int64)


def normalize_train_tuning_sampling_mode(mode: str) -> str:
    value = str(mode).strip() or TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION
    if value not in TRAIN_TUNING_SAMPLING_MODES:
        raise ValueError(f"Unknown train-tuning sampling mode {mode!r}.")
    return value


def train_tuning_sampler_key(mode: str) -> str:
    normalized = normalize_train_tuning_sampling_mode(str(mode))
    if normalized == TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED:
        return TRAIN_TUNING_SAMPLER_VALIDATION_NORMALIZED_V2
    return TRAIN_TUNING_SAMPLER_WINDOW_FRACTION_V1


def train_tuning_target_example_count(
    train_examples: int,
    *,
    fraction: float,
    sampling_mode: str = TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    reference_examples: Optional[int] = None,
    train_split_fraction: float = DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    val_split_fraction: float = DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
) -> int:
    total = int(train_examples)
    if total <= 0:
        raise ValueError("Forecast train split is empty.")
    frac = float(fraction)
    if not math.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"train-tuning fraction must be positive, got {fraction!r}.")
    mode = normalize_train_tuning_sampling_mode(str(sampling_mode))
    if mode == TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION:
        if frac >= 1.0:
            return total
        return min(total, max(1, int(math.floor(float(total) * frac + 0.5))))
    ref = int(reference_examples or 0)
    if ref <= 0:
        raise ValueError("validation_normalized train-tuning sampling requires positive reference_examples.")
    train_frac = float(train_split_fraction)
    val_frac = float(val_split_fraction)
    if not math.isfinite(train_frac) or train_frac <= 0.0:
        raise ValueError(f"train_tuning_train_split_fraction must be positive, got {train_split_fraction!r}.")
    if not math.isfinite(val_frac) or val_frac <= 0.0:
        raise ValueError(f"train_tuning_val_split_fraction must be positive, got {val_split_fraction!r}.")
    target = int(math.floor(float(ref) * frac * train_frac / val_frac + 0.5))
    return min(total, max(1, target))


def _strata_ranges(total: int, strata: int) -> List[Tuple[int, int]]:
    strata_count = max(1, min(int(strata), int(total)))
    ranges: List[Tuple[int, int]] = []
    for stratum in range(strata_count):
        start = int(math.floor(float(stratum) * float(total) / float(strata_count)))
        end = int(math.floor(float(stratum + 1) * float(total) / float(strata_count)))
        if end > start:
            ranges.append((start, end))
    return ranges


def _allocate_stratified_target_counts(total: int, strata: int, target: int) -> List[int]:
    ranges = _strata_ranges(int(total), int(strata))
    capped_target = min(int(total), max(1, int(target)))
    raw_counts = []
    counts: List[int] = []
    for idx, (start, end) in enumerate(ranges):
        size = int(end - start)
        raw = float(capped_target) * float(size) / float(total)
        base = min(size, int(math.floor(raw)))
        counts.append(base)
        raw_counts.append((raw - float(base), idx))
    remaining = int(capped_target) - int(sum(counts))
    while remaining > 0:
        progressed = False
        for _, idx in sorted(raw_counts, key=lambda item: (-item[0], item[1])):
            size = int(ranges[idx][1] - ranges[idx][0])
            if counts[idx] >= size:
                continue
            counts[idx] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return counts


def choose_forecast_train_tuning_indices(
    ds,
    *,
    fraction: float = 0.20,
    seed: int = 0,
    strata: int = 20,
    dataset: str = "",
    salt: str = "train_tuning_v1",
    sampling_mode: str = TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION,
    reference_examples: Optional[int] = None,
    train_split_fraction: float = DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION,
    val_split_fraction: float = DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION,
) -> np.ndarray:
    """Deterministically sample train windows by temporal strata."""
    total = int(len(ds))
    if total <= 0:
        raise ValueError("Forecast train split is empty.")
    frac = float(fraction)
    if not math.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"train-tuning fraction must be positive, got {fraction!r}.")
    mode = normalize_train_tuning_sampling_mode(str(sampling_mode))
    if mode == TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION:
        if frac >= 1.0:
            return np.arange(total, dtype=np.int64)
        strata_count = max(1, min(int(strata), total))
        selected: List[int] = []
        for stratum in range(strata_count):
            start = int(math.floor(float(stratum) * float(total) / float(strata_count)))
            end = int(math.floor(float(stratum + 1) * float(total) / float(strata_count)))
            if end <= start:
                continue
            size = int(end - start)
            keep = max(1, int(round(float(size) * frac)))
            keep = min(size, keep)
            scored = []
            for idx in range(start, end):
                token = f"{salt}|{dataset}|{int(seed)}|{stratum}|{idx}"
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                scored.append((int(digest[:16], 16), idx))
            scored.sort(key=lambda item: (item[0], item[1]))
            selected.extend(idx for _, idx in scored[:keep])
        return np.asarray(sorted(set(selected)), dtype=np.int64)

    target = train_tuning_target_example_count(
        total,
        fraction=frac,
        sampling_mode=mode,
        reference_examples=reference_examples,
        train_split_fraction=train_split_fraction,
        val_split_fraction=val_split_fraction,
    )
    if target >= total:
        return np.arange(total, dtype=np.int64)
    ranges = _strata_ranges(total, int(strata))
    counts = _allocate_stratified_target_counts(total, int(strata), int(target))
    selected = []
    for stratum, ((start, end), keep) in enumerate(zip(ranges, counts)):
        if int(keep) <= 0:
            continue
        size = int(end - start)
        token = f"{salt}|{mode}|{dataset}|{int(seed)}|{stratum}|{start}|{end}|{target}"
        local_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(local_seed)
        chosen = rng.choice(size, size=int(keep), replace=False) + int(start)
        selected.extend(int(idx) for idx in chosen.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _parse_forecast_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Any]:
    if len(batch) == 3:
        hist, tgt, meta = batch
        return hist, tgt, None, meta
    if len(batch) == 4:
        hist, tgt, fut, meta = batch
        return hist, tgt, fut, meta
    raise ValueError(f"Unexpected forecast batch format with {len(batch)} items.")


def selection_metric_for_family(benchmark_family: str) -> str:
    if str(benchmark_family) == FORECAST_FAMILY:
        return "crps"
    if str(benchmark_family) == CONDITIONAL_GENERATION_FAMILY:
        return "score_main"
    raise ValueError(f"Unsupported benchmark_family={benchmark_family}")


def solver_eval_multiplier(solver_key: str) -> int:
    key = str(solver_key)
    if key in {"euler", "dpmpp2m"}:
        return 1
    if key in {"heun", "midpoint_rk2"}:
        return 2
    raise ValueError(f"Unsupported solver_key={solver_key}")


def solver_macro_steps(solver_key: str, target_nfe: int) -> int:
    if str(solver_key) in {"heun", "midpoint_rk2"}:
        if int(target_nfe) % 2 != 0:
            raise ValueError(f"{solver_key} requires an even target_nfe, got {target_nfe}")
        return int(target_nfe) // 2
    multiplier = int(solver_eval_multiplier(str(solver_key)))
    if multiplier == 1:
        return int(target_nfe)
    return max(1, int(round(float(target_nfe) / float(multiplier))))


def solver_experiment_scope(solver_key: str) -> str:
    return "solver_transfer" if str(solver_key) == "dpmpp2m" else "main"


def solver_order_p(solver_key: str) -> float:
    key = str(solver_key)
    if key == "euler":
        return 1.0
    if key in {"heun", "midpoint_rk2", "dpmpp2m"}:
        return 2.0
    raise ValueError(f"Unsupported solver order mapping for {solver_key}")


def resolve_reference_macro_steps(
    requested_macro_steps: int,
    runtime_nfe: int,
    *,
    reference_macro_factor: float = 4.0,
) -> int:
    requested = int(requested_macro_steps)
    if requested > 0:
        return requested
    factor = float(reference_macro_factor)
    if factor <= 0.0:
        raise ValueError(f"reference_macro_factor must be positive, got {reference_macro_factor}")
    return max(32, int(round(factor * int(runtime_nfe))))


def _resolved_backbone_manifest_path(cli_args: argparse.Namespace) -> Optional[Path]:
    raw = str(getattr(cli_args, "backbone_manifest", "") or "").strip()
    if not raw:
        return None
    path = resolve_project_path(raw)
    if not path.exists():
        return None
    return path


def _load_ready_backbone_manifest(cli_args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    manifest_path = _resolved_backbone_manifest_path(cli_args)
    if manifest_path is None:
        return None
    return load_backbone_manifest(manifest_path)


def _resolved_manifest_artifact(
    cli_args: argparse.Namespace,
    *,
    benchmark_family: str,
    dataset_key: str,
) -> Optional[Dict[str, Any]]:
    manifest_payload = _load_ready_backbone_manifest(cli_args)
    if manifest_payload is None:
        return None
    return find_backbone_artifact(
        manifest_payload,
        backbone_name=BACKBONE_NAME_OTFLOW,
        benchmark_family=str(benchmark_family),
        dataset_key=str(dataset_key),
        train_steps=int(getattr(cli_args, "otflow_train_steps", 20000)),
        status="ready",
    )


def _safe_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_checkpoint_path(path: str | Path) -> Path:
    return resolve_project_path(str(path))


def _missing_shared_checkpoint_paths(
    *,
    shared_backbone_root: Path,
    forecast_datasets: Sequence[str],
    conditional_generation_datasets: Sequence[str],
) -> List[Path]:
    missing: List[Path] = []
    for dataset in forecast_datasets:
        ckpt_path = shared_backbone_root / "forecast" / str(dataset) / "model.pt"
        if not ckpt_path.exists():
            missing.append(ckpt_path)
    for dataset in conditional_generation_datasets:
        ckpt_path = (
            shared_backbone_root
            / "conditional_generation"
            / str(dataset)
            / DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
            / "model.pt"
        )
        if not ckpt_path.exists():
            missing.append(ckpt_path)
    return missing


def validate_execution_preflight(cli_args: argparse.Namespace) -> None:
    forecast_datasets = parse_forecast_datasets(str(cli_args.forecast_datasets))
    conditional_generation_datasets = parse_conditional_generation_datasets(
        str(cli_args.conditional_generation_datasets)
    )
    shared_backbone_root = resolve_project_path(str(cli_args.shared_backbone_root))
    errors: List[str] = []
    manifest_payload = _load_ready_backbone_manifest(cli_args)
    if manifest_payload is not None:
        missing_artifacts: List[str] = []
        missing_manifest_checkpoints: List[Path] = []
        for dataset in forecast_datasets:
            try:
                artifact = find_backbone_artifact(
                    manifest_payload,
                    backbone_name=BACKBONE_NAME_OTFLOW,
                    benchmark_family=FORECAST_FAMILY,
                    dataset_key=str(dataset),
                    train_steps=int(getattr(cli_args, "otflow_train_steps", 20000)),
                    status="ready",
                )
                ckpt_path = _resolve_checkpoint_path(str(artifact["checkpoint_path"]))
                if not ckpt_path.exists():
                    missing_manifest_checkpoints.append(ckpt_path)
            except KeyError:
                missing_artifacts.append(f"{FORECAST_FAMILY}:{dataset}")
        for dataset in conditional_generation_datasets:
            try:
                artifact = find_backbone_artifact(
                    manifest_payload,
                    backbone_name=BACKBONE_NAME_OTFLOW,
                    benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                    dataset_key=str(dataset),
                    train_steps=int(getattr(cli_args, "otflow_train_steps", 20000)),
                    status="ready",
                )
                ckpt_path = _resolve_checkpoint_path(str(artifact["checkpoint_path"]))
                if not ckpt_path.exists():
                    missing_manifest_checkpoints.append(ckpt_path)
            except KeyError:
                missing_artifacts.append(f"{CONDITIONAL_GENERATION_FAMILY}:{dataset}")
        if missing_artifacts:
            errors.append(
                "Backbone manifest is missing ready OTFlow checkpoints for the selected datasets and train_steps="
                f"{int(getattr(cli_args, 'otflow_train_steps', 20000))}: {', '.join(missing_artifacts)}"
            )
        if missing_manifest_checkpoints:
            missing_lines = ", ".join(str(path) for path in missing_manifest_checkpoints)
            errors.append(
                "Backbone manifest contains ready OTFlow artifacts whose checkpoint files are missing: "
                f"{missing_lines}"
            )
    else:
        missing_checkpoints = _missing_shared_checkpoint_paths(
            shared_backbone_root=shared_backbone_root,
            forecast_datasets=forecast_datasets,
            conditional_generation_datasets=conditional_generation_datasets,
        )
        if missing_checkpoints:
            missing_lines = ", ".join(str(path) for path in missing_checkpoints)
            errors.append(
                "Missing shared checkpoints for the selected datasets. "
                f"Provide checkpoint-ready datasets or produce the missing checkpoints first: {missing_lines}"
            )
    if errors:
        raise RuntimeError("Execution preflight failed:\n- " + "\n- ".join(errors))


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(arr.shape[0], dtype=np.float64)
    start = 0
    while start < order.size:
        end = start
        while end + 1 < order.size and abs(float(arr[order[end + 1]]) - float(arr[order[start]])) <= 1e-12:
            end += 1
        avg_rank = 0.5 * (float(start + 1) + float(end + 1))
        for idx in range(start, end + 1):
            ranks[order[idx]] = float(avg_rank)
        start = end + 1
    return ranks


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")
    x_rank = _rankdata_average(x_arr)
    y_rank = _rankdata_average(y_arr)
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    return float(corr)


def _resolved_conditional_generation_physical_batch_size(dataset: str) -> int:
    dataset_key = str(dataset)
    default_value = int(CONDITIONAL_GENERATION_PHYSICAL_BATCH_SIZE_BY_DATASET[dataset_key])
    if dataset_key != SLEEP_EDF_DATASET_KEY:
        return default_value
    raw = str(os.environ.get("OTFLOW_SLEEP_EDF_PHYSICAL_BATCH_SIZE", "") or "").strip()
    if not raw:
        return default_value
    try:
        override = int(raw)
    except ValueError:
        return default_value
    return max(1, int(override))


def _dataset_data_path(cli_args: argparse.Namespace, dataset: str) -> str:
    if str(dataset) == "cryptos":
        return str(getattr(cli_args, "cryptos_path", "") or default_cryptos_data_path())
    if str(dataset) == "es_mbp_10":
        return str(getattr(cli_args, "es_path", "") or default_es_mbp_10_data_path())
    if str(dataset) == SLEEP_EDF_DATASET_KEY:
        return str(getattr(cli_args, "sleep_edf_path", "") or default_sleep_edf_data_path())
    raise ValueError(f"Unknown conditional-generation dataset: {dataset}")


def resolved_train_steps(cli_args: argparse.Namespace, dataset: str) -> int:
    del dataset
    return int(cli_args.steps) if int(cli_args.steps) > 0 else int(DEFAULT_CONDITIONAL_GENERATION_TRAIN_STEPS)


def resolved_eval_horizon(cli_args: argparse.Namespace, dataset: str) -> int:
    spec = experiment_plan_by_key()[str(dataset)]
    return int(cli_args.eval_horizon) if int(cli_args.eval_horizon) > 0 else int(spec.experiment_horizon)


def resolved_eval_windows(cli_args: argparse.Namespace, dataset: str, split: str) -> int:
    assert split in {"val", "test"}
    raw = int(cli_args.eval_windows_val if split == "val" else cli_args.eval_windows_test)
    if raw > 0:
        return raw
    plan = DATASET_PLANS[str(dataset)]
    return int(plan.eval_windows_final)


def build_conditional_generation_dataset_args_from_cfg(
    cli_args: argparse.Namespace,
    dataset: str,
    field_network_type: str,
    cfg,
) -> argparse.Namespace:
    plan = DATASET_PLANS[str(dataset)]
    experiment_spec = experiment_plan_by_key()[str(dataset)]
    preset = get_otflow_paper_backbone_preset(str(dataset))
    batch_size = int(_resolved_conditional_generation_physical_batch_size(str(dataset)))
    grad_accum_steps = max(1, int(math.ceil(32.0 / float(max(1, batch_size)))))
    locked_future_block_len = (
        int(cli_args.future_block_len)
        if int(getattr(cli_args, "future_block_len", 0)) > 0
        else int(experiment_spec.future_block_len)
    )
    locked_history_len = int(experiment_spec.history_len)
    args = argparse.Namespace(
        dataset=str(dataset),
        data_path=_dataset_data_path(cli_args, str(dataset)),
        synthetic_length=int(plan.synthetic_length),
        seed=int(cli_args.dataset_seed),
        device=str(cli_args.device),
        steps=int(resolved_train_steps(cli_args, str(dataset))),
        train_frac=plan.train_frac,
        val_frac=plan.val_frac,
        test_frac=plan.test_frac,
        stride_train=plan.stride_train,
        stride_eval=plan.stride_eval,
        levels=int(preset["levels"]),
        token_dim=int(preset.get("token_dim", 4)),
        history_len=int(locked_history_len),
        batch_size=int(batch_size),
        lr=float(cli_args.lr),
        weight_decay=float(cli_args.weight_decay),
        grad_clip=float(cli_args.grad_clip),
        standardize=True,
        use_cond_features=bool(preset.get("use_cond_features", False)),
        cond_standardize=bool(preset.get("cond_standardize", True)),
        hidden_dim=int(cli_args.hidden_dim),
        ctx_encoder=str(preset["ctx_encoder"]),
        ctx_causal=bool(preset["ctx_causal"]),
        ctx_local_kernel=int(preset["ctx_local_kernel"]),
        ctx_pool_scales=str(preset["ctx_pool_scales"]),
        use_time_features=bool(preset.get("use_time_features", preset.get("use_time_gaps", False))),
        use_time_gaps=bool(preset.get("use_time_gaps", False)),
        fu_net_type=str(field_network_type),
        fu_net_layers=int(cli_args.fu_net_layers),
        fu_net_heads=int(cli_args.fu_net_heads),
        rollout_mode=str(cli_args.rollout_mode),
        future_block_len=int(locked_future_block_len),
        adaptive_context=False,
        adaptive_context_ratio=None,
        adaptive_context_min=None,
        adaptive_context_max=None,
        train_variable_context=False,
        train_context_min=None,
        train_context_max=None,
        use_minibatch_ot=True,
        solver="euler",
        use_amp=True,
        grad_accum_steps=int(grad_accum_steps),
    )
    args.steps = int(getattr(cfg.train, "steps", getattr(args, "steps", resolved_train_steps(cli_args, dataset))))
    args.levels = int(cfg.levels)
    args.token_dim = int(getattr(cfg, "token_dim", getattr(args, "token_dim", 4)))
    checkpoint_history_len = int(getattr(cfg, "history_len", args.history_len))
    if checkpoint_history_len != int(locked_history_len):
        raise RuntimeError(
            f"Conditional-generation checkpoint config history_len={checkpoint_history_len} does not match locked "
            f"{dataset} history_len={int(locked_history_len)}."
        )
    args.history_len = int(locked_history_len)
    args.batch_size = int(cfg.batch_size)
    args.use_cond_features = bool(getattr(cfg, "use_cond_features", getattr(args, "use_cond_features", False)))
    args.cond_standardize = bool(getattr(cfg, "cond_standardize", getattr(args, "cond_standardize", True)))
    args.hidden_dim = int(cfg.hidden_dim)
    args.ctx_encoder = str(getattr(cfg.model, "ctx_encoder", args.ctx_encoder))
    args.ctx_causal = bool(getattr(cfg.model, "ctx_causal", args.ctx_causal))
    args.ctx_local_kernel = int(getattr(cfg.model, "ctx_local_kernel", args.ctx_local_kernel))
    args.use_time_features = bool(getattr(cfg.model, "use_time_features", getattr(args, "use_time_features", False)))
    args.use_time_gaps = bool(getattr(cfg.model, "use_time_gaps", getattr(args, "use_time_gaps", False)))
    args.fu_net_type = str(getattr(cfg.model, "fu_net_type", field_network_type))
    args.fu_net_layers = int(getattr(cfg.model, "fu_net_layers", args.fu_net_layers))
    args.fu_net_heads = int(getattr(cfg.model, "fu_net_heads", args.fu_net_heads))
    args.rollout_mode = str(getattr(cfg.model, "rollout_mode", args.rollout_mode))
    checkpoint_future_block_len = int(getattr(cfg.model, "future_block_len", args.future_block_len))
    if checkpoint_future_block_len != int(locked_future_block_len):
        raise RuntimeError(
            f"Conditional-generation checkpoint config future_block_len={checkpoint_future_block_len} does not match locked "
            f"{dataset} future_block_len={int(locked_future_block_len)}."
        )
    args.future_block_len = int(locked_future_block_len)
    return args


def load_checkpoint_model(ckpt_path: Path, device: torch.device) -> Tuple[OTFlow, Any]:
    from genode.models.config import OTFlowConfig

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = OTFlowConfig()
    section_types = {
        "data": type(cfg.data),
        "model": type(cfg.model),
        "fm": type(cfg.fm),
        "train": type(cfg.train),
        "sample": type(cfg.sample),
    }
    for section_name, cls in section_types.items():
        section_values = {field.name: getattr(getattr(cfg, section_name), field.name) for field in fields(cls)}
        checkpoint_values = dict(cfg_dict.get(section_name, {}))
        if section_name == "model":
            checkpoint_values = {
                key: value for key, value in checkpoint_values.items() if key not in IGNORED_BASELINE_MODEL_CONFIG_KEYS
            }
        section_values.update(checkpoint_values)
        if section_name == "train":
            section_values["device"] = device
        valid_keys = {field.name for field in fields(cls)}
        unknown_keys = set(section_values) - valid_keys
        if unknown_keys:
            raise TypeError(
                f"Checkpoint config section {section_name!r} contains unsupported keys: {sorted(unknown_keys)}"
            )
        setattr(cfg, section_name, cls(**section_values))
    cfg.train.device = device

    model = OTFlow(cfg).to(device)
    model_state = dict(ckpt["model_state"])
    load_result = model.load_state_dict(model_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint model_state is incompatible: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model, cfg


def _metadata_path_for_checkpoint(manifest_artifact: Optional[Mapping[str, Any]], ckpt_path: Path) -> Path:
    if manifest_artifact is not None and str(manifest_artifact.get("metadata_path", "") or "").strip():
        return Path(str(manifest_artifact["metadata_path"])).resolve()
    return Path(ckpt_path).resolve().with_name("checkpoint_metadata.json")


def _metadata_split_cond_dim(metadata: Mapping[str, Any]) -> int:
    split_stats = dict(metadata.get("split_stats", {}) or {})
    if "cond_dim" in split_stats:
        return int(split_stats.get("cond_dim") or 0)
    return int(metadata.get("cond_dim") or 0)


def _required_metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"checkpoint metadata field {key!r} is required.")
    return value


def _validate_metadata_identity(
    *,
    metadata: Mapping[str, Any],
    ckpt_path: Path,
    dataset: str,
    benchmark_family: str,
    expected_train_steps: int,
    expected_history_len: int,
    expected_future_block_len: int,
    expected_field_network_type: Optional[str] = None,
) -> None:
    if not metadata:
        raise RuntimeError(f"Checkpoint {ckpt_path} is missing checkpoint_metadata.json for {dataset}.")
    checks = (
        ("dataset_key", str(dataset), str),
        ("benchmark_family", str(benchmark_family), str),
        ("train_steps", int(expected_train_steps), int),
        ("history_len", int(expected_history_len), int),
        ("future_block_len", int(expected_future_block_len), int),
    )
    for key, expected, caster in checks:
        observed = caster(_required_metadata_value(metadata, key))
        if observed != expected:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} metadata mismatch for {dataset}: "
                f"{key}={observed!r}, expected {expected!r}."
            )
    if expected_field_network_type is not None:
        observed_field = str(_required_metadata_value(metadata, "field_network_type"))
        if observed_field != str(expected_field_network_type):
            raise RuntimeError(
                f"Checkpoint {ckpt_path} metadata mismatch for {dataset}: "
                f"field_network_type={observed_field!r}, expected {str(expected_field_network_type)!r}."
            )


def _validate_forecast_checkpoint_task(
    *,
    dataset: str,
    ckpt_path: Path,
    metadata: Mapping[str, Any],
    checkpoint_model_cond_dim: int,
    checkpoint_train_steps: int,
    checkpoint_history_len: int,
    checkpoint_future_block_len: int,
    expected_train_steps: int,
) -> None:
    experiment_spec = experiment_plan_by_key()[str(dataset)]
    _validate_metadata_identity(
        metadata=metadata,
        ckpt_path=ckpt_path,
        dataset=str(dataset),
        benchmark_family=FORECAST_FAMILY,
        expected_train_steps=int(expected_train_steps),
        expected_history_len=int(experiment_spec.history_len),
        expected_future_block_len=int(experiment_spec.future_block_len),
    )
    metadata_cond_dim = _metadata_split_cond_dim(metadata)
    if int(checkpoint_model_cond_dim) != int(metadata_cond_dim):
        raise RuntimeError(
            f"Forecast checkpoint {ckpt_path} cond_dim mismatch for {dataset}: "
            f"checkpoint={int(checkpoint_model_cond_dim)}, metadata={int(metadata_cond_dim)}."
        )
    if int(checkpoint_train_steps) != int(expected_train_steps):
        raise RuntimeError(
            f"Forecast checkpoint {ckpt_path} train_steps mismatch for {dataset}: "
            f"checkpoint={int(checkpoint_train_steps)}, expected={int(expected_train_steps)}."
        )
    if int(checkpoint_history_len) != int(experiment_spec.history_len):
        raise RuntimeError(
            f"Forecast checkpoint {ckpt_path} history_len mismatch for {dataset}: "
            f"checkpoint={int(checkpoint_history_len)}, expected={int(experiment_spec.history_len)}."
        )
    if int(checkpoint_future_block_len) != int(experiment_spec.future_block_len):
        raise RuntimeError(
            f"Forecast checkpoint {ckpt_path} future_block_len mismatch for {dataset}: "
            f"checkpoint={int(checkpoint_future_block_len)}, expected={int(experiment_spec.future_block_len)}."
        )


def _validate_conditional_generation_checkpoint_task(
    *,
    dataset: str,
    ckpt_path: Path,
    metadata: Mapping[str, Any],
    checkpoint_model_cond_dim: int,
    checkpoint_train_steps: int,
    checkpoint_history_len: int,
    checkpoint_future_block_len: int,
    expected_train_steps: int,
    expected_history_len: int,
    expected_future_block_len: int,
    splits: Optional[Mapping[str, Any]] = None,
) -> None:
    _validate_metadata_identity(
        metadata=metadata,
        ckpt_path=ckpt_path,
        dataset=str(dataset),
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        expected_train_steps=int(expected_train_steps),
        expected_history_len=int(expected_history_len),
        expected_future_block_len=int(expected_future_block_len),
        expected_field_network_type=DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE,
    )
    metadata_cond_dim = _metadata_split_cond_dim(metadata)
    if int(checkpoint_model_cond_dim) != int(metadata_cond_dim):
        raise RuntimeError(
            f"Conditional-generation checkpoint {ckpt_path} is incompatible with split metadata for {dataset}: "
            f"model.cond_dim={int(checkpoint_model_cond_dim)}, split_stats.cond_dim={int(metadata_cond_dim)}."
        )
    if int(checkpoint_train_steps) != int(expected_train_steps):
        raise RuntimeError(
            f"Conditional-generation checkpoint {ckpt_path} train_steps mismatch: "
            f"checkpoint={int(checkpoint_train_steps)}, expected={int(expected_train_steps)}."
        )
    if int(checkpoint_history_len) != int(expected_history_len):
        raise RuntimeError(
            f"Conditional-generation checkpoint {ckpt_path} history_len mismatch: "
            f"checkpoint={int(checkpoint_history_len)}, expected={int(expected_history_len)}."
        )
    if int(checkpoint_future_block_len) != int(expected_future_block_len):
        raise RuntimeError(
            f"Conditional-generation checkpoint {ckpt_path} future_block_len mismatch: "
            f"checkpoint={int(checkpoint_future_block_len)}, expected={int(expected_future_block_len)}."
        )
    if splits is None:
        return
    stats = dict(splits.get("stats", {}) or {})
    split_cond_dim = int(stats.get("cond_dim") or 0)
    if split_cond_dim != int(checkpoint_model_cond_dim):
        raise RuntimeError(
            f"Conditional-generation checkpoint {ckpt_path} architecture does not match rebuilt {dataset} split: "
            f"model.cond_dim={int(checkpoint_model_cond_dim)}, split_stats.cond_dim={int(split_cond_dim)}."
        )
    split_history = stats.get("history_len")
    if split_history is not None and int(split_history) != int(checkpoint_history_len):
        raise RuntimeError(
            f"Conditional-generation checkpoint {ckpt_path} history_len does not match rebuilt {dataset} split: "
            f"checkpoint={int(checkpoint_history_len)}, split_stats.history_len={int(split_history)}."
        )


def load_forecast_checkpoint_splits(
    *,
    cli_args: argparse.Namespace,
    dataset_root: Path,
    shared_backbone_root: Path,
    dataset: str,
    device: torch.device,
) -> Dict[str, Any]:
    expected_train_steps = int(getattr(cli_args, "otflow_train_steps", 20000))
    manifest_artifact = _resolved_manifest_artifact(cli_args, benchmark_family=FORECAST_FAMILY, dataset_key=str(dataset))
    if manifest_artifact is not None:
        ckpt_path = _resolve_checkpoint_path(str(manifest_artifact["checkpoint_path"]))
        checkpoint_id = str(manifest_artifact["checkpoint_id"])
        resolved_train_steps = int(manifest_artifact["train_steps"])
        resolved_budget_label = str(manifest_artifact["train_budget_label"])
        backbone_name = str(manifest_artifact.get("backbone_name", BACKBONE_NAME_OTFLOW))
    else:
        ckpt_path = shared_backbone_root / "forecast" / str(dataset) / "model.pt"
        metadata = _safe_json(shared_backbone_root / "forecast" / str(dataset) / "checkpoint_metadata.json") or {}
        resolved_train_steps = int(metadata.get("train_steps", int(getattr(cli_args, "otflow_train_steps", 20000))))
        resolved_budget_label = str(metadata.get("train_budget_label", train_budget_label(resolved_train_steps)))
        checkpoint_id = str(
            metadata.get("checkpoint_id")
            or build_backbone_checkpoint_id(
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=FORECAST_FAMILY,
                dataset_key=str(dataset),
                train_steps=resolved_train_steps,
            )
        )
        backbone_name = BACKBONE_NAME_OTFLOW
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Forecast checkpoint not found: {ckpt_path}")
    metadata_path = _metadata_path_for_checkpoint(manifest_artifact, ckpt_path)
    metadata = _safe_json(metadata_path) or {}
    model, cfg = load_checkpoint_model(ckpt_path, device=device)
    _validate_forecast_checkpoint_task(
        dataset=str(dataset),
        ckpt_path=ckpt_path,
        metadata=metadata,
        checkpoint_model_cond_dim=int(getattr(cfg.model, "cond_dim", 0)),
        checkpoint_train_steps=int(getattr(cfg.train, "steps", 0)),
        checkpoint_history_len=int(cfg.history_len),
        checkpoint_future_block_len=int(getattr(cfg.model, "future_block_len", 1)),
        expected_train_steps=int(expected_train_steps),
    )
    time_feature_mode = _forecast_time_feature_mode(cfg)
    splits = build_monash_forecast_splits(
        dataset_root=dataset_root,
        dataset_key=str(dataset),
        cfg=cfg,
        history_len=int(cfg.history_len),
        horizon=int(cfg.prediction_horizon),
        stride_train=1,
        time_feature_mode=str(time_feature_mode),
    )
    return {
        "model": model,
        "cfg": cfg,
        "splits": splits,
        "checkpoint_path": ckpt_path,
        "checkpoint_id": str(checkpoint_id),
        "backbone_name": str(backbone_name),
        "train_steps": int(resolved_train_steps),
        "train_budget_label": str(resolved_budget_label),
    }


def load_conditional_generation_checkpoint_splits(
    *,
    cli_args: argparse.Namespace,
    shared_backbone_root: Path,
    dataset: str,
    device: torch.device,
) -> Dict[str, Any]:
    expected_train_steps = int(getattr(cli_args, "otflow_train_steps", 20000))
    experiment_spec = experiment_plan_by_key()[str(dataset)]
    expected_history_len = int(experiment_spec.history_len)
    expected_future_block_len = (
        int(getattr(cli_args, "future_block_len", 0))
        if int(getattr(cli_args, "future_block_len", 0)) > 0
        else int(experiment_spec.future_block_len)
    )
    manifest_artifact = _resolved_manifest_artifact(
        cli_args,
        benchmark_family=CONDITIONAL_GENERATION_FAMILY,
        dataset_key=str(dataset),
    )
    if manifest_artifact is not None:
        ckpt_path = _resolve_checkpoint_path(str(manifest_artifact["checkpoint_path"]))
        checkpoint_id = str(manifest_artifact["checkpoint_id"])
        resolved_train_steps = int(manifest_artifact["train_steps"])
        resolved_budget_label = str(manifest_artifact["train_budget_label"])
        backbone_name = str(manifest_artifact.get("backbone_name", BACKBONE_NAME_OTFLOW))
    else:
        ckpt_path = (
            shared_backbone_root
            / "conditional_generation"
            / str(dataset)
            / DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
            / "model.pt"
        )
        metadata = (
            _safe_json(
                shared_backbone_root
                / "conditional_generation"
                / str(dataset)
                / DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE
                / "checkpoint_metadata.json"
            )
            or {}
        )
        resolved_train_steps = int(metadata.get("train_steps", int(getattr(cli_args, "otflow_train_steps", 20000))))
        resolved_budget_label = str(metadata.get("train_budget_label", train_budget_label(resolved_train_steps)))
        checkpoint_id = str(
            metadata.get("checkpoint_id")
            or build_backbone_checkpoint_id(
                backbone_name=BACKBONE_NAME_OTFLOW,
                benchmark_family=CONDITIONAL_GENERATION_FAMILY,
                dataset_key=str(dataset),
                train_steps=resolved_train_steps,
                field_network_type=DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE,
            )
        )
        backbone_name = BACKBONE_NAME_OTFLOW
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Conditional-generation checkpoint not found: {ckpt_path}")
    metadata_path = _metadata_path_for_checkpoint(manifest_artifact, ckpt_path)
    metadata = _safe_json(metadata_path) or {}
    model, cfg = load_checkpoint_model(ckpt_path, device=device)
    checkpoint_model_cond_dim = int(getattr(cfg.model, "cond_dim", 0))
    checkpoint_train_steps = int(getattr(cfg.train, "steps", 0))
    checkpoint_history_len = int(cfg.history_len)
    checkpoint_future_block_len = int(getattr(cfg.model, "future_block_len", 1))
    _validate_conditional_generation_checkpoint_task(
        dataset=str(dataset),
        ckpt_path=ckpt_path,
        metadata=metadata,
        checkpoint_model_cond_dim=int(checkpoint_model_cond_dim),
        checkpoint_train_steps=int(checkpoint_train_steps),
        checkpoint_history_len=int(checkpoint_history_len),
        checkpoint_future_block_len=int(checkpoint_future_block_len),
        expected_train_steps=int(expected_train_steps),
        expected_history_len=int(expected_history_len),
        expected_future_block_len=int(expected_future_block_len),
    )
    dataset_args = build_conditional_generation_dataset_args_from_cfg(
        cli_args,
        str(dataset),
        DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE,
        cfg,
    )
    splits = build_dataset_splits(dataset_args, cfg)
    _validate_conditional_generation_checkpoint_task(
        dataset=str(dataset),
        ckpt_path=ckpt_path,
        metadata=metadata,
        checkpoint_model_cond_dim=int(checkpoint_model_cond_dim),
        checkpoint_train_steps=int(checkpoint_train_steps),
        checkpoint_history_len=int(checkpoint_history_len),
        checkpoint_future_block_len=int(checkpoint_future_block_len),
        expected_train_steps=int(expected_train_steps),
        expected_history_len=int(expected_history_len),
        expected_future_block_len=int(expected_future_block_len),
        splits=splits,
    )
    return {
        "model": model,
        "cfg": cfg,
        "splits": splits,
        "checkpoint_path": ckpt_path,
        "checkpoint_id": str(checkpoint_id),
        "backbone_name": str(backbone_name),
        "train_steps": int(resolved_train_steps),
        "train_budget_label": str(resolved_budget_label),
    }



def collect_forecast_calibration(model: OTFlow, ds_val, cfg, *, macro_steps: int, solver_name: str, seed: int, calibration_trace_samples: int = 1, info_growth_scale_multiplier: float = 1.0) -> Dict[str, Any]:
    trace_samples=int(calibration_trace_samples)
    if trace_samples<=0: raise ValueError(f"calibration_trace_samples must be positive, got {calibration_trace_samples}")
    reference_time_grid: Optional[np.ndarray]=None; disagreement_rows=[]; residual_rows=[]; oracle_rows=[]; trace_rows=[]; device=cfg.train.device
    for example_idx in range(len(ds_val)):
        hist_t, _, _, _ = _parse_forecast_batch(ds_val[int(example_idx)])
        hist=hist_t[None].to(device).float(); dsamps=[]; rsamps=[]; osamps=[]
        for sample_idx in range(trace_samples):
            seed_all(int(seed)+int(example_idx)+1_000_000*int(sample_idx))
            _,trace=model.sample_future_trace(hist,steps=int(macro_steps),solver=str(solver_name),oracle_local_error=True)
            grid=trace["time_grid"].detach().cpu().numpy().astype(np.float64)
            if reference_time_grid is None: reference_time_grid=grid
            elif not np.allclose(reference_time_grid,grid,atol=1e-8,rtol=1e-8): raise ValueError("Forecast calibration trace time grids must match across validation examples.")
            dsamps.append(trace["disagreement"][0].detach().cpu().numpy().astype(np.float64)); rsamps.append(trace["residual_norm"][0].detach().cpu().numpy().astype(np.float64)); osamps.append(trace["oracle_local_error"][0].detach().cpu().numpy().astype(np.float64))
        d=np.stack(dsamps,axis=0).mean(axis=0); r=np.stack(rsamps,axis=0).mean(axis=0); o=np.stack(osamps,axis=0).mean(axis=0)
        disagreement_rows.append(d); residual_rows.append(r); oracle_rows.append(o)
        for step_idx,(dv,rv,ov) in enumerate(zip(d.tolist(),r.tolist(),o.tolist())): trace_rows.append({"example_index":int(example_idx),"step_index":int(step_idx),"disagreement":float(dv),"residual_norm":float(rv),"oracle_local_error":float(ov)})
    if not disagreement_rows: raise ValueError("Forecast validation split is empty; cannot calibrate native info-growth trace.")
    disagreement_arr=np.stack(disagreement_rows,axis=0); residual_arr=np.stack(residual_rows,axis=0); oracle_arr=np.stack(oracle_rows,axis=0)
    base_scale=resolved_info_growth_scale(residual_arr.reshape(-1)); effective_scale=float(base_scale)*float(info_growth_scale_multiplier)
    if effective_scale<=0.0: raise ValueError(f"info_growth_scale_multiplier must keep scale positive, got {info_growth_scale_multiplier}")
    info_growth_arr=compute_info_growth_hardness_numpy(residual_arr,disagreement_arr,scale=float(effective_scale))
    if reference_time_grid is None: reference_time_grid=np.linspace(0.0,1.0,int(macro_steps)+1,dtype=np.float64)
    corr_signal=info_growth_arr[:,1:].reshape(-1); corr_oracle=oracle_arr[:,1:].reshape(-1)
    return {"macro_steps":int(macro_steps),"solver":str(solver_name),"n_windows":int(info_growth_arr.shape[0]),"calibration_trace_samples":int(trace_samples),"reference_time_grid":[float(x) for x in reference_time_grid.tolist()],"reference_time_alignment":"left_endpoint","base_info_growth_scale":float(base_scale),"info_growth_scale":float(effective_scale),"info_growth_scale_multiplier":float(info_growth_scale_multiplier),"rows":trace_rows,"disagreement_by_step":[float(x) for x in disagreement_arr.mean(axis=0).tolist()],"residual_norm_by_step":[float(x) for x in residual_arr.mean(axis=0).tolist()],"oracle_local_error_by_step":[float(x) for x in oracle_arr.mean(axis=0).tolist()],NATIVE_INFO_GROWTH_TRACE_KEY:[float(x) for x in info_growth_arr.mean(axis=0).tolist()],"signal_correlations_vs_oracle":{NATIVE_INFO_GROWTH_TRACE_KEY:{"spearman":safe_spearman(corr_signal,corr_oracle)}}}


@torch.no_grad()
def evaluate_forecast_schedule(
    model: OTFlow,
    ds,
    cfg,
    *,
    solver_name: str,
    runtime_nfe: int,
    time_grid: Sequence[float],
    num_eval_samples: int,
    seed: int,
    example_indices: Optional[Sequence[int]] = None,
    batch_size: int = 1,
    progress_label: str = "",
) -> Dict[str, Any]:
    device = cfg.train.device
    mse_values: List[float] = []
    crps_values: List[float] = []
    mase_values: List[float] = []
    latencies: List[float] = []
    effective_batch_size = max(1, int(batch_size))
    backup = _apply_sample_overrides(model, cfg, solver=str(solver_name), time_grid=tuple(float(x) for x in time_grid))
    try:
        selected_examples = (
            choose_forecast_example_indices(ds, 0, int(seed))
            if example_indices is None
            else np.asarray([int(idx) for idx in example_indices], dtype=np.int64)
        )
        if selected_examples.ndim != 1 or selected_examples.size <= 0:
            raise ValueError("example_indices must be a non-empty 1D sequence.")
        invalid = [int(idx) for idx in selected_examples.tolist() if int(idx) < 0 or int(idx) >= int(len(ds))]
        if invalid:
            raise ValueError(f"example_indices contains out-of-range entries: {invalid}")
        example_list = [int(idx) for idx in selected_examples.tolist()]
        chunk_starts = list(range(0, len(example_list), effective_batch_size))
        progress_enabled = bool(str(progress_label).strip()) and len(chunk_starts) > 1
        with ProgressBar(len(chunk_starts), str(progress_label), enabled=progress_enabled) as progress:
            for chunk_start in chunk_starts:
                chunk_indices = example_list[chunk_start : chunk_start + effective_batch_size]
                hist_rows: List[torch.Tensor] = []
                true_blocks: List[np.ndarray] = []
                mase_scales: List[float] = []
                for example_idx in chunk_indices:
                    hist_t, tgt_t, fut_t, _ = _parse_forecast_batch(ds[int(example_idx)])
                    hist_rows.append(hist_t.float())
                    true_parts = [tgt_t[None, :]]
                    if fut_t is not None:
                        true_parts.append(fut_t)
                    true_block_norm = torch.cat(true_parts, dim=0).cpu().numpy()
                    true_blocks.append(ds.denormalize_block(true_block_norm, int(example_idx)).reshape(-1))
                    mase_scales.append(float(ds.mase_denom(int(example_idx))))
                hist = torch.stack(hist_rows, dim=0).to(device).float()
                chunk_draws: List[np.ndarray] = []
                for sample_idx in range(int(num_eval_samples)):
                    seed_all(int(seed) + 1_000_000 * int(chunk_start) + int(sample_idx))
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    pred_norm = model.sample_future(hist, steps=int(runtime_nfe), solver=str(solver_name))
                    if device.type == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    latencies.append(float(elapsed) / float(max(1, len(chunk_indices))))
                    pred_np = pred_norm.detach().cpu().numpy()
                    pred_rows = [
                        ds.denormalize_block(pred_np[row_idx], int(example_idx)).reshape(-1).astype(np.float32)
                        for row_idx, example_idx in enumerate(chunk_indices)
                    ]
                    chunk_draws.append(np.stack(pred_rows, axis=0))
                samples_by_example = np.stack(chunk_draws, axis=0)
                for row_idx, true_block_raw in enumerate(true_blocks):
                    samples = samples_by_example[:, row_idx, :]
                    pred_mean = samples.mean(axis=0)
                    mse_values.append(float(np.mean((pred_mean - true_block_raw) ** 2)))
                    crps_values.append(_empirical_crps(samples, true_block_raw))
                    mase_values.append(_point_mase(pred_mean, true_block_raw, mase_scales[row_idx]))
                progress.update()
    finally:
        _restore_sample_overrides(model, cfg, backup)
    latency_arr = np.asarray(latencies, dtype=np.float64)
    example_payload = [int(idx) for idx in selected_examples.tolist()]
    encoded_protocol = json.dumps(
        {
            "batch_size": int(effective_batch_size),
            "example_indices": example_payload,
            "num_eval_samples": int(num_eval_samples),
            "runtime_nfe": int(runtime_nfe),
            "solver_name": str(solver_name),
            "time_grid": [float(x) for x in time_grid],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    chosen_examples_hash = hashlib.sha256(json.dumps(example_payload, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "crps": float(np.mean(np.asarray(crps_values, dtype=np.float64))) if crps_values else float("nan"),
        "mse": float(np.mean(np.asarray(mse_values, dtype=np.float64))) if mse_values else float("nan"),
        "mase": float(np.mean(np.asarray(mase_values, dtype=np.float64))) if mase_values else float("nan"),
        "latency_ms_per_sample": float(1000.0 * latency_arr.mean()) if latency_arr.size > 0 else float("nan"),
        "eval_examples": int(len(example_payload)),
        "eval_horizon": int(getattr(ds, "horizon", 1)),
        "evaluation_protocol_hash": hashlib.sha256(encoded_protocol.encode("utf-8")).hexdigest(),
        "chosen_examples_hash": str(chosen_examples_hash),
        "num_eval_samples": int(num_eval_samples),
        "realized_nfe": int(runtime_nfe) * int(solver_eval_multiplier(str(solver_name))),
    }


__all__ = [
    "ALL_SOLVER_ORDER",
    "CONDITIONAL_GENERATION_FAMILY",
    "CONDITIONAL_GENERATION_PHYSICAL_BATCH_SIZE_BY_DATASET",
    "DEFAULT_CONDITIONAL_GENERATION_TRAIN_STEPS",
    "DEFAULT_CONDITIONAL_GENERATION_DATASETS",
    "DEFAULT_CONDITIONAL_GENERATION_FIELD_NETWORK_TYPE",
    "DEFAULT_FORECAST_DATASETS",
    "DEFAULT_SHARED_BACKBONE_ROOT",
    "DEFAULT_TRAIN_TUNING_TRAIN_SPLIT_FRACTION",
    "DEFAULT_TRAIN_TUNING_VAL_SPLIT_FRACTION",
    "DEFAULT_SIGNAL_TRACE_KEY",
    "FORECAST_FAMILY",
    "LOCKED_TEST_PHASE",
    "SOLVER_RUNTIME_NAMES",
    "SUPPORTED_FORECAST_DATASETS",
    "TRAIN_TUNING_PHASE",
    "TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED",
    "TRAIN_TUNING_SAMPLING_MODE_WINDOW_FRACTION",
    "TRAIN_TUNING_SAMPLING_MODES",
    "UNIFORM_SCHEDULER_KEY",
    "VALIDATION_PHASE",
    "collect_forecast_calibration",
    "choose_forecast_example_indices",
    "choose_forecast_train_tuning_indices",
    "evaluate_forecast_schedule",
    "normalize_train_tuning_sampling_mode",
    "train_tuning_sampler_key",
    "train_tuning_target_example_count",
    "load_checkpoint_model",
    "load_conditional_generation_checkpoint_splits",
    "load_forecast_checkpoint_splits",
    "parse_conditional_generation_datasets",
    "parse_csv",
    "parse_float_csv",
    "parse_forecast_datasets",
    "parse_int_csv",
    "resolve_reference_macro_steps",
    "resolved_eval_horizon",
    "resolved_eval_windows",
    "resolved_train_steps",
    "safe_spearman",
    "save_json",
    "selection_metric_for_family",
    "solver_eval_multiplier",
    "solver_experiment_scope",
    "solver_macro_steps",
    "solver_order_p",
    "validate_execution_preflight",
]
