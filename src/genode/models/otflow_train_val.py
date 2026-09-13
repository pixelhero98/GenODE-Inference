"""Shared training and deterministic seeding for OTFlow backbones."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from genode.data.otflow_datasets import WindowedParamSequenceDataset
from genode.models.config import OTFlowConfig
from genode.models.modules import EMAModel
from genode.models.otflow_model import OTFlow

SUPPORTED_MODEL_NAMES = ("otflow",)
_NUMPY_SEED_MODULUS = 2**32


def _bounded_numpy_seed(seed: int) -> int:
    seed_i = int(seed)
    if seed_i < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")
    return int(seed_i % _NUMPY_SEED_MODULUS)


def seed_all(seed: int = 0):
    normalized_seed = _bounded_numpy_seed(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    torch.cuda.manual_seed_all(normalized_seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state for restart checkpoints."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "protocol": "numpy_random_state_v1",
            "bit_generator": str(numpy_state[0]),
            "keys": [int(value) for value in numpy_state[1].tolist()],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    if not state:
        return
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    cuda_state = state.get("cuda")
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            "protocol",
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise ValueError("Restart checkpoint contains an invalid NumPy RNG state.")
        if numpy_state.get("protocol") != "numpy_random_state_v1":
            raise ValueError("Restart checkpoint contains an unsupported NumPy RNG protocol.")
        keys = np.asarray(numpy_state.get("keys"), dtype=np.uint32)
        if keys.ndim != 1 or keys.size == 0:
            raise ValueError("Restart checkpoint contains invalid NumPy RNG keys.")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                keys,
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def make_loader(
    ds: WindowedParamSequenceDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a DataLoader with safe defaults for small split sizes.

    `drop_last=False` avoids creating an empty loader when len(ds) < batch_size,
    which would otherwise cause training loops to fail with repeated StopIteration.
    """
    sampler = None
    effective_shuffle = bool(shuffle)
    if bool(shuffle) and getattr(ds, "sampler_weights", None) is not None:
        weights = torch.as_tensor(ds.sampler_weights, dtype=torch.double)
        num_samples = getattr(ds, "sampler_num_samples", None)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=int(num_samples) if num_samples is not None else int(len(weights)),
            replacement=True,
            generator=generator,
        )
        effective_shuffle = False
    elif bool(shuffle) and bool(getattr(ds, "sampler_replacement", False)):
        num_samples = getattr(ds, "sampler_num_samples", None)
        if num_samples is not None and int(num_samples) > 0:
            sampler = RandomSampler(ds, replacement=True, num_samples=int(num_samples), generator=generator)
            effective_shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
    )


def _parse_batch(batch):
    """Unpack the dataset tuple emitted by WindowedParamSequenceDataset.

    Supports:
      - (hist, tgt, meta)
      - (hist, tgt, cond, meta)
      - (hist, tgt, fut, meta)
      - (hist, tgt, fut, cond, meta)

    Batched future horizons are rank-3 tensors [B, H_fut, D], batched
    conditioning features are rank-2 tensors [B, C]. Unbatched examples use
    rank-2 futures [H_fut, D] and rank-1 conditions [C].
    """
    if len(batch) == 3:
        hist, tgt, meta = batch
        return (hist, tgt, None, None, meta)
    if len(batch) == 4:
        hist, tgt, a, meta = batch
        hist_rank = int(hist.dim())
        arg_rank = int(a.dim())
        if hist_rank == 3 and arg_rank == 2:
            return (hist, tgt, None, a, meta)
        if hist_rank == 3 and arg_rank == 3:
            return (hist, tgt, a, None, meta)
        if hist_rank == 2 and arg_rank == 1:
            return (hist, tgt, None, a, meta)
        if hist_rank == 2 and arg_rank == 2:
            return (hist, tgt, a, None, meta)
        raise ValueError(f"Unexpected 4-item batch tensor ranks: hist={hist_rank}, item={arg_rank}.")
    if len(batch) == 5:
        hist, tgt, fut, cond, meta = batch
        return (hist, tgt, fut, cond, meta)
    raise ValueError("Unexpected batch format.")


def _torch_sync(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _amp_enabled(cfg: OTFlowConfig, device: torch.device) -> bool:
    return bool(getattr(cfg.train, "use_amp", False)) and device.type == "cuda" and torch.cuda.is_available()


@contextmanager
def _autocast_context(cfg: OTFlowConfig, device: torch.device):
    if _amp_enabled(cfg, device):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            yield
        return
    yield


@contextmanager
def _temporary_eval_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    normalized_seed = _bounded_numpy_seed(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    torch.cuda.manual_seed_all(normalized_seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def resolve_context_length(max_available: int, *, horizon: int, cfg: OTFlowConfig | None) -> int:
    max_available = max(1, int(max_available))
    if cfg is None or not bool(getattr(cfg, "adaptive_context", False)):
        return max_available
    ratio = float(getattr(cfg, "adaptive_context_ratio", 1.5))
    ctx_min = int(getattr(cfg, "adaptive_context_min", 1))
    ctx_max = int(getattr(cfg, "adaptive_context_max", max_available))
    desired = int(round(max(1, int(horizon)) * ratio))
    desired = max(ctx_min, desired)
    desired = min(desired, ctx_max, max_available)
    return max(1, desired)


def crop_history_window(hist: torch.Tensor, context_len: int) -> torch.Tensor:
    context_len = max(1, int(context_len))
    if hist.dim() == 3:
        return hist[:, -context_len:, :]
    if hist.dim() == 2:
        return hist[-context_len:, :]
    raise ValueError(f"Unsupported history tensor rank: {hist.dim()}")


def sample_training_context_length(max_available: int, cfg: OTFlowConfig | None) -> int:
    max_available = max(1, int(max_available))
    if cfg is None or not bool(getattr(cfg, "train_variable_context", False)):
        return max_available
    min_len = max(1, int(getattr(cfg, "train_context_min", 1)))
    max_len = int(getattr(cfg, "train_context_max", max_available))
    min_len = min(min_len, max_available)
    max_len = min(max(max_len, min_len), max_available)
    return int(np.random.randint(min_len, max_len + 1))


def _model_prediction_horizon(model: torch.nn.Module) -> int:
    model_cfg = getattr(model, "cfg", None)
    if model_cfg is None:
        return 1
    return int(max(1, int(getattr(model_cfg, "prediction_horizon", 1))))


def _build_scheduler(opt: torch.optim.Optimizer, cfg: OTFlowConfig, total_steps: int):
    """Build an optional LR scheduler (warmup + cosine decay)."""
    schedule = getattr(cfg, "lr_schedule", "constant").lower()
    warmup = int(getattr(cfg, "lr_warmup_steps", 0))
    if schedule == "constant" and warmup <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        if schedule == "cosine":
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()
    if normalized not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Only model_name='otflow' is supported, got {model_name!r}.")
    return normalized


def _validate_model_dataset_support(model_name: str, cfg: OTFlowConfig) -> None:
    del cfg
    _normalize_model_name(model_name)


def _build_model(model_name: str, cfg: OTFlowConfig, device: torch.device) -> torch.nn.Module:
    _validate_model_dataset_support(model_name, cfg)
    return OTFlow(cfg).to(device)


def _compute_training_loss(
    model: torch.nn.Module,
    *,
    tgt: torch.Tensor,
    hist: torch.Tensor,
    fut: torch.Tensor | None,
    cond: torch.Tensor | None,
    meta: Any,
    loss_mode: str | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(model, OTFlow):
        return model.loss(tgt, hist, fut=fut, cond=cond, meta=meta)
    del loss_mode
    raise RuntimeError("Unexpected model type; OTFlow is the only supported model.")


@torch.no_grad()
def evaluate_average_loss(
    ds: WindowedParamSequenceDataset,
    model: torch.nn.Module,
    cfg: OTFlowConfig,
    *,
    model_name: str = "otflow",
    max_batches: int | None = None,
    loss_mode: str | None = None,
    shuffle: bool = False,
) -> dict[str, Any]:
    """Evaluate the mean training objective on a dataset split."""
    _normalize_model_name(model_name)
    device = cfg.device
    loader = make_loader(ds, cfg.batch_size, shuffle=shuffle, drop_last=False)
    if len(loader) == 0:
        raise ValueError("Evaluation loader is empty.")
    was_training = bool(model.training)
    model.eval()
    total_loss = 0.0
    total_examples = 0
    batches = 0
    try:
        for batch in loader:
            hist, tgt, fut, cond, meta = _parse_batch(batch)
            hist = hist.to(device).float()
            tgt = tgt.to(device).float()
            fut = fut.to(device).float() if fut is not None else None
            cond = cond.to(device).float() if cond is not None else None
            context_len = resolve_context_length(hist.shape[1], horizon=_model_prediction_horizon(model), cfg=cfg)
            hist = crop_history_window(hist, context_len)
            loss, logs = _compute_training_loss(
                model, tgt=tgt, hist=hist, fut=fut, cond=cond, meta=meta, loss_mode=loss_mode
            )
            batch_size = int(hist.shape[0])
            loss_value = float(logs.get("loss", float(loss.detach())))
            total_loss += loss_value * float(batch_size)
            total_examples += batch_size
            batches += 1
            if max_batches is not None and batches >= int(max_batches):
                break
    finally:
        if was_training:
            model.train()
    if total_examples <= 0:
        raise ValueError("Evaluation produced no examples.")
    return {"loss": float(total_loss / float(total_examples)), "examples": int(total_examples), "batches": int(batches)}


_MAX_TORCH_GENERATOR_SEED = 2**63 - 1


def _loader_epoch_seed(base_seed: int, epoch: int) -> int:
    return int((int(base_seed) + (int(epoch) + 1) * 11400714819323198485) % _MAX_TORCH_GENERATOR_SEED)


def _tensor_mapping_to_cpu(mapping: Mapping[str, torch.Tensor] | None) -> dict[str, torch.Tensor] | None:
    if mapping is None:
        return None
    return {str(key): value.detach().cpu().clone() for key, value in mapping.items()}


def train_loop(
    ds: WindowedParamSequenceDataset,
    cfg: OTFlowConfig,
    model_name: str = "otflow",
    steps: int = 10000,
    log_every: int = 200,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    loss_mode: str | None = None,
    shuffle: bool = True,
    on_step: Callable[[int, torch.nn.Module, float, dict[str, float]], None] | None = None,
    initial_model_state: Mapping[str, torch.Tensor] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    scaler_state: Mapping[str, Any] | None = None,
    ema_state: Mapping[str, torch.Tensor] | None = None,
    swa_model_state: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    loader_state: Mapping[str, Any] | None = None,
    start_step: int = 0,
    on_training_state: Callable[
        [int, torch.nn.Module, torch.optim.Optimizer, Any | None, torch.cuda.amp.GradScaler, dict[str, Any]], None
    ]
    | None = None,
) -> torch.nn.Module:
    """Train a model on next-step prediction in normalized param space.

    Features:
    - EMA model averaging (cfg.ema_decay > 0)
    - LR warmup + cosine decay (cfg.lr_schedule, cfg.lr_warmup_steps)
    """
    total_steps = int(steps)
    if total_steps <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}.")
    start_step = int(start_step)
    if start_step < 0 or start_step > total_steps:
        raise ValueError(f"start_step must be in [0, {total_steps}], got {start_step}.")
    device = cfg.device
    model_name = _normalize_model_name(model_name)
    _validate_model_dataset_support(model_name, cfg)
    model = _build_model(model_name, cfg, device) if model is None else model.to(device)
    if initial_model_state is not None:
        model.load_state_dict(dict(initial_model_state))
    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if optimizer_state is not None:
        opt.load_state_dict(dict(optimizer_state))
    accum_steps = max(1, int(getattr(cfg.train, "grad_accum_steps", 1)))
    use_amp = _amp_enabled(cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if scaler_state is not None:
        scaler.load_state_dict(dict(scaler_state))
    scheduler = _build_scheduler(opt, cfg, total_steps)
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(dict(scheduler_state))
    ema_decay = float(getattr(cfg, "ema_decay", 0.0))
    ema = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None
    if ema is not None and ema_state is not None:
        ema.shadow = {str(key): value.detach().clone().to(device) for key, value in ema_state.items()}
    use_swa = getattr(cfg, "use_swa", False)
    swa_model = AveragedModel(model) if use_swa else None
    if swa_model is not None and swa_model_state is not None:
        swa_model.load_state_dict(dict(swa_model_state))
    swa_start = int(0.75 * total_steps)
    restore_rng_state(rng_state)
    raw_loader_state = dict(loader_state or {})
    loader_seed = int(raw_loader_state.get("seed", torch.initial_seed()))
    loader_epoch = int(raw_loader_state.get("epoch", 0) or 0)
    loader_batch_index = int(raw_loader_state.get("batch_index", 0) or 0)

    def _make_epoch_loader(epoch: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(_loader_epoch_seed(loader_seed, int(epoch)))
        loader = make_loader(ds, cfg.batch_size, shuffle=shuffle, drop_last=False, generator=generator)
        if len(loader) == 0:
            raise ValueError(
                "Training loader is empty. Check dataset construction, history_len, split boundaries, and batch_size."
            )
        return loader

    loader = _make_epoch_loader(loader_epoch)
    loader_len = len(loader)
    if loader_batch_index >= loader_len:
        loader_epoch += int(loader_batch_index // loader_len)
        loader_batch_index = int(loader_batch_index % loader_len)
        loader = _make_epoch_loader(loader_epoch)
        loader_len = len(loader)
    it = iter(loader)
    for _ in range(loader_batch_index):
        next(it)
    model.train()
    opt.zero_grad(set_to_none=True)
    opt_step = start_step
    micro_step = 0
    while opt_step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            loader_epoch += 1
            loader_batch_index = 0
            loader = _make_epoch_loader(loader_epoch)
            loader_len = len(loader)
            it = iter(loader)
            batch = next(it)
        loader_batch_index += 1
        hist, tgt, fut, cond, meta = _parse_batch(batch)
        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        fut = fut.to(device).float() if fut is not None else None
        cond = cond.to(device).float() if cond is not None else None
        train_context_len = sample_training_context_length(hist.shape[1], cfg)
        hist = crop_history_window(hist, train_context_len)
        with _autocast_context(cfg, device):
            loss, logs = _compute_training_loss(
                model, tgt=tgt, hist=hist, fut=fut, cond=cond, meta=meta, loss_mode=loss_mode
            )
        micro_step += 1
        loss_for_backward = loss / float(accum_steps)
        scaler.scale(loss_for_backward).backward()
        if micro_step % accum_steps != 0:
            continue
        opt_step += 1
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        if swa_model is not None and opt_step >= swa_start:
            swa_model.update_parameters(model)
        latest_train_loss = float(logs.get("loss", float(loss.detach())))
        if on_step is not None:
            on_step(int(opt_step), model, latest_train_loss, dict(logs))
        if opt_step % log_every == 0:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"[{model_name}] step {opt_step}/{total_steps}  loss={latest_train_loss:.4f}  lr={lr_now:.2e}  details={logs}"
            )
        if on_training_state is not None:
            training_state = {
                "loader_state": {
                    "seed": int(loader_seed),
                    "epoch": int(loader_epoch),
                    "batch_index": int(loader_batch_index),
                    "loader_batches_per_epoch": int(loader_len),
                },
                "ema_state": _tensor_mapping_to_cpu(ema.shadow) if ema is not None else None,
                "swa_model_state": swa_model.state_dict() if swa_model is not None else None,
            }
            on_training_state(int(opt_step), model, opt, scheduler, scaler, training_state)
    if swa_model is not None:
        print(f"[{model_name}] Applying SWA weights tracked over the last {total_steps - swa_start + 1} steps")
        model.load_state_dict(swa_model.module.state_dict())
    elif ema is not None:
        ema.apply_shadow(model)
    return model.eval()


def save_json(obj: dict[str, Any], path: str):

    def _conv(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path_obj.with_name(f".{path_obj.name}.{time.time_ns()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_conv)
    tmp_path.replace(path_obj)
    print(f"Saved JSON -> {path_obj}")


__all__ = [
    "capture_rng_state",
    "crop_history_window",
    "evaluate_average_loss",
    "make_loader",
    "resolve_context_length",
    "restore_rng_state",
    "sample_training_context_length",
    "save_json",
    "seed_all",
    "train_loop",
]
