from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


_CONFIG_SECTIONS = ("data", "model", "fm", "train", "sample")


@dataclass
class SequenceDataConfig:
    levels: int = 10
    token_dim: int = 4
    history_len: int = 256
    context_feature_dim: int = 0
    standardize: bool = True
    use_cond_features: bool = False
    cond_depths: Tuple[int, ...] = (1, 3, 5, 10)
    cond_vol_window: int = 50
    cond_standardize: bool = True

    @property
    def snapshot_dim(self) -> int:
        return int(self.token_dim) * int(self.levels)

    @property
    def state_dim(self) -> int:
        return self.snapshot_dim


@dataclass
class SharedModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    cond_dim: int = 0
    ctx_encoder: str = "transformer"
    ctx_causal: bool = True
    ctx_local_kernel: int = 5
    ctx_pool_scales: Tuple[int, ...] = (4, 16)
    ctx_heads: int = 4
    ctx_layers: int = 2
    diffusion_steps: int = 32
    adaptive_context: bool = False
    adaptive_context_ratio: float = 1.5
    adaptive_context_min: int = 64
    adaptive_context_max: int = 256
    train_variable_context: bool = False
    train_context_min: int = 64
    train_context_max: int = 256
    use_time_gaps: bool = False
    use_time_features: bool = False
    use_res_mlp: bool = True
    fu_net_type: str = "transformer"
    fu_net_layers: int = 3
    fu_net_heads: int = 4
    rollout_mode: str = "autoregressive"
    future_block_len: int = 1


@dataclass
class FMConfig:
    use_minibatch_ot: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 64
    steps: int = 20_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    lr_warmup_steps: int = 500
    lr_schedule: str = "cosine"
    use_swa: bool = True
    use_amp: bool = True
    grad_accum_steps: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps: float = 1e-8


@dataclass
class SampleConfig:
    steps: int = 2
    cfg_scale: float = 1.0
    solver: str = "euler"
    time_grid: Tuple[float, ...] = ()
    adaptive_beta: float = 0.9
    adaptive_tau: float = 0.15
    adaptive_kappa: float = 12.0
    adaptive_gamma_max: float = 0.05
    adaptive_cooldown_steps: int = 0
    adaptive_noise_mode: str = "orthogonal"
    adaptive_trigger_mode: str = "adaptive"
    adaptive_disable_noise_frac: float = 0.1
    adaptive_rtol: float = 1e-3
    adaptive_atol: float = 1e-6
    adaptive_safety: float = 0.9
    adaptive_min_step: float = 1e-5
    adaptive_max_nfe: int = 512
    refine_beta: float = 0.9
    refine_trigger_mode: str = "zscore"
    refine_threshold_z: float = 1.5
    refine_threshold_raw: float = 0.0
    refine_step_mu: Tuple[float, ...] = ()
    refine_step_sigma: Tuple[float, ...] = ()
    refine_step_threshold: Tuple[float, ...] = ()
    refine_selected_steps: Tuple[int, ...] = ()
    refine_fixed_last_k: int = 0
    refine_sigma_eps: float = 1e-6
    refine_disallow_final_step: bool = True


@dataclass(init=False)
class OTFlowConfig:
    data: SequenceDataConfig = field(default_factory=SequenceDataConfig)
    model: SharedModelConfig = field(default_factory=SharedModelConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    def __init__(
        self,
        data: Optional[SequenceDataConfig] = None,
        model: Optional[SharedModelConfig] = None,
        fm: Optional[FMConfig] = None,
        train: Optional[TrainConfig] = None,
        sample: Optional[SampleConfig] = None,
        **flat_overrides: Any,
    ):
        object.__setattr__(self, "data", data if data is not None else SequenceDataConfig())
        object.__setattr__(self, "model", model if model is not None else SharedModelConfig())
        object.__setattr__(self, "fm", fm if fm is not None else FMConfig())
        object.__setattr__(self, "train", train if train is not None else TrainConfig())
        object.__setattr__(self, "sample", sample if sample is not None else SampleConfig())
        if flat_overrides:
            self.apply_overrides(**flat_overrides)

    def __getattr__(self, name: str) -> Any:
        for section_name in _CONFIG_SECTIONS:
            section = object.__getattribute__(self, section_name)
            if hasattr(section, name):
                return getattr(section, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def apply_overrides(self, **flat_overrides: Any) -> "OTFlowConfig":
        for key, value in flat_overrides.items():
            matched = False
            for section_name in _CONFIG_SECTIONS:
                section = getattr(self, section_name)
                if hasattr(section, key):
                    setattr(section, key, value)
                    matched = True
                    break
            if not matched:
                raise TypeError(f"Unknown config field: {key}")
        return self

    @property
    def state_dim(self) -> int:
        return self.data.state_dim

    @property
    def snapshot_dim(self) -> int:
        return self.data.snapshot_dim

    @property
    def context_dim(self) -> int:
        base_dim = int(getattr(self.data, "context_feature_dim", 0) or self.snapshot_dim)
        use_elapsed = bool(getattr(self.model, "use_time_features", False))
        use_gap_only = bool(getattr(self.model, "use_time_gaps", False))
        if use_elapsed and use_gap_only:
            raise ValueError("Time features must use exactly one mode: none, gap_only, or gap_elapsed.")
        if use_elapsed:
            extra_dim = 2
        elif use_gap_only:
            extra_dim = 1
        else:
            extra_dim = 0
        return int(base_dim) + int(extra_dim)

    @property
    def prediction_horizon(self) -> int:
        rollout_mode = str(self.model.rollout_mode).strip().lower()
        future_block_len = max(1, int(self.model.future_block_len))
        return future_block_len if rollout_mode == "non_ar" else 1

    @property
    def sample_state_dim(self) -> int:
        return int(self.snapshot_dim) * int(self.prediction_horizon)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        train_dict = asdict(self.train)
        if isinstance(train_dict.get("device"), torch.device):
            train_dict["device"] = str(train_dict["device"])
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "fm": asdict(self.fm),
            "train": train_dict,
            "sample": asdict(self.sample),
        }
