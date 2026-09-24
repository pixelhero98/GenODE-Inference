"""An AMP overflow must not consume an exact-budget backbone step."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

import genode.models.otflow_train_val as training
from genode.data.otflow_datasets import build_dataset_splits_from_arrays
from genode.models.config import OTFlowConfig
from genode.models.otflow_model import OTFlow


class SkippingScaler:
    def __init__(self, *, enabled: bool):
        self.enabled = enabled
        self.attempts = 0
        self.successes = 0
        self.scale_value = 8.0
        self.skipped = False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        self.attempts += 1
        self.skipped = self.attempts == 1
        if not self.skipped:
            optimizer.step()
            self.successes += 1

    def update(self):
        if self.skipped:
            self.scale_value /= 2


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_amp_overflow_does_not_advance_exact_budget_or_scheduler():
    cfg = OTFlowConfig(
        device=torch.device("cpu"),
        levels=1,
        token_dim=4,
        history_len=4,
        hidden_dim=16,
        dropout=0.0,
        ctx_heads=4,
        ctx_layers=1,
        fu_net_layers=1,
        fu_net_heads=4,
        rollout_mode="non_ar",
        future_block_len=2,
        use_cond_features=False,
        cond_standardize=False,
        cond_dim=0,
        use_amp=False,
        ema_decay=0.0,
    )
    params = np.random.default_rng(0).normal(size=(80, 4)).astype(np.float32)
    times = np.arange(80, dtype=np.float32)
    ds = build_dataset_splits_from_arrays(params, times, cfg, train_frac=0.6, val_frac=0.2)["train"]
    model = OTFlow(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = SkippingScaler(enabled=False)
    scheduler = CountingScheduler()
    callbacks = []

    def record(step, _model, _loss, _logs):
        optimizer_steps = max(int(state["step"]) for state in optimizer.state.values())
        callbacks.append((step, optimizer_steps, scheduler.steps))

    with (
        patch.object(training.torch.cuda.amp, "GradScaler", return_value=scaler),
        patch.object(training, "_build_scheduler", return_value=scheduler),
    ):
        training.train_loop(
            ds,
            cfg,
            model=model,
            optimizer=optimizer,
            steps=3,
            log_every=100,
            on_step=record,
        )

    assert scaler.attempts == 4
    assert scaler.successes == 3
    assert callbacks == [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
