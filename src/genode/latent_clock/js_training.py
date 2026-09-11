"""Budgeted JS optimization and a separate, versioned inference artifact."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from genode.latent_clock.artifacts import sha256_file, write_new_json
from genode.latent_clock.js_reinforce import (
    JS_PROTOCOL,
    MIN_INTERVAL,
    DirichletSchedulePolicy,
    JSArchitecture,
    materialize_js,
    reinforce_loss,
    sample_intervals,
)


@dataclass(frozen=True)
class JSFit:
    trajectories: int
    anchor_trajectories: int
    seed: int
    batch_contexts: int = 32
    rollouts: int = 2
    learning_rate: float = 5e-5

    def updates(self, context_count: int) -> int:
        remaining = self.trajectories - self.anchor_trajectories
        if self.batch_contexts < 3 or self.batch_contexts > context_count or self.rollouts < 2:
            raise ValueError("JS requires 3 <= batch contexts <= available contexts and at least two rollouts.")
        if self.anchor_trajectories != context_count or remaining <= 0:
            raise ValueError("Charge exactly one uniform anchor per prompt/noise context before JS rollouts.")
        if remaining % (self.batch_contexts * self.rollouts):
            raise ValueError("JS trajectory allowance must contain complete balanced rollout batches.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("JS learning rate must be positive and finite.")
        return remaining // (self.batch_contexts * self.rollouts)


def stack_inputs(inputs):
    result = {}
    for key in ("noise", "text", "pooled", "padding_mask"):
        values = [item[key] for item in inputs]
        if all(value is None for value in values):
            result[key] = None
        elif any(value is None for value in values):
            raise ValueError(f"Inconsistent JS native input availability: {key}")
        else:
            # Frozen encoder outputs may be inference tensors. A real copy outside
            # inference_mode permits saving them for the policy's backward pass.
            result[key] = torch.cat(values).detach().clone()
    return result


def fit_js(policy, *, context_count, inputs, evaluate, config: JSFit, on_update=None):
    """Callbacks supply frozen native inputs and a single terminal reward per action.

    Each batch contains distinct contexts and K independent clocks per context.
    Context batches cycle through shuffled training contexts; no validation data
    or teacher scores enter optimization. The final fixed-budget state is used.
    """
    updates = config.updates(context_count)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    generator = np.random.default_rng(config.seed)
    order, position = generator.permutation(context_count), 0
    policy.train()
    history = []
    for update in range(updates):
        if position + config.batch_contexts > context_count:
            order, position = generator.permutation(context_count), 0
        indices = order[position : position + config.batch_contexts].tolist()
        position += config.batch_contexts
        concentration = policy(**stack_inputs([inputs(index) for index in indices]))
        actions = sample_intervals(concentration, seed=config.seed + 1_000_003 * (update + 1), rollouts=config.rollouts)
        rewards = torch.tensor(
            [
                [
                    evaluate(index, materialize_js(action.cpu().numpy()), update, member)
                    for member, action in enumerate(actions[row])
                ]
                for row, index in enumerate(indices)
            ],
            dtype=torch.float64,
            device=concentration.device,
        )
        loss, diagnostics = reinforce_loss(concentration, actions, rewards)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = {
            "update": update + 1,
            "loss": loss.item(),
            "reward": rewards.mean().item(),
            "gradient_norm": gradient.item(),
            **diagnostics,
        }
        history.append(record)
        if on_update is not None:
            on_update(record)
    policy.eval()
    return history


def save_js(path, policy, metadata):
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    payload = {
        "protocol": JS_PROTOCOL,
        "architecture": asdict(policy.architecture),
        "semantics": {
            "solver": "euler",
            "skipped_interval": "initial",
            "terminal": 0.0,
            "minimum_interval": MIN_INTERVAL,
            "sampling": "one_joint_draw_per_image",
        },
        "metadata": metadata,
        "state_dict": {key: value.detach().cpu() for key, value in policy.state_dict().items()},
    }
    # Validate metadata is portable and safe for the weights-only loader.
    json.dumps(metadata, allow_nan=False)
    torch.save(payload, destination / "policy.pt")
    write_new_json(
        destination / "manifest.json",
        {"protocol": JS_PROTOCOL, "policy_sha256": sha256_file(destination / "policy.pt")},
    )


def load_js(path, *, device="cpu"):
    source = Path(path)
    manifest = json.loads((source.parent / "manifest.json").read_text())
    if manifest.get("protocol") != JS_PROTOCOL or manifest.get("policy_sha256") != sha256_file(source):
        raise ValueError("Incompatible or modified JS artifact.")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected = {
        "solver": "euler",
        "skipped_interval": "initial",
        "terminal": 0.0,
        "minimum_interval": MIN_INTERVAL,
        "sampling": "one_joint_draw_per_image",
    }
    if payload.get("protocol") != JS_PROTOCOL or payload.get("semantics") != expected:
        raise ValueError("Incompatible JS architecture/solver protocol; retrain this baseline.")
    with torch.random.fork_rng(devices=[]):
        policy = DirichletSchedulePolicy(JSArchitecture(**payload["architecture"]))
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.to(device).eval().requires_grad_(False)
    return policy, payload["metadata"]
