from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from genode.latent_clock.clocks import Clock
from genode.latent_clock.contracts import ExecutionTrace, FrozenContext

if TYPE_CHECKING:
    from genode.latent_clock.js_reinforce import JSSchedule


def _prepare_scheduler(scheduler: Any, clock: Clock | JSSchedule, device: Any) -> Any:
    import torch

    target = 1.0 - np.asarray(clock.nodes, dtype=np.float64)
    # Reset the upstream step/begin indices and inference-step metadata. Install
    # the complete grid directly: an inverse/forward shift round trip can merge
    # adjacent float32 nodes of the reversed p=8 reference at NFE 8.
    scheduler.set_timesteps(clock.target_nfe, device=device)
    scheduler.sigmas = torch.tensor(target, dtype=torch.float32, device=device)
    scheduler.timesteps = (
        torch.tensor(target[:-1], dtype=torch.float32, device=device) * scheduler.config.num_train_timesteps
    )
    if not bool(torch.all(torch.diff(scheduler.sigmas) < 0)) or not bool(
        torch.all(torch.diff(scheduler.timesteps) < 0)
    ):
        raise RuntimeError(f"Clock {clock.key} at NFE {clock.target_nfe} collapses in runtime precision.")
    return scheduler.timesteps


class SanaFlowEulerAdapter:
    """Clock-controlled form of NVlabs/Sana FlowEuler at pinned source revision."""

    solver_key = "euler"

    def __init__(
        self,
        *,
        sampler: Any,
        context_encoder: Callable[[str], tuple[Any, Any, np.ndarray]],
        latent_factory: Callable[[int, FrozenContext], Any],
        decoder: Callable[[Any], Any],
        backbone_revision: str,
    ) -> None:
        self.sampler = sampler
        self.context_encoder = context_encoder
        self.latent_factory = latent_factory
        self.decoder = decoder
        self.backbone_revision = str(backbone_revision)
        self._opaque_contexts: dict[str, tuple[Any, Any, dict[str, Any]]] = {}

    def encode_context(self, prompt_id: str, prompt: str) -> FrozenContext:
        encoded = self.context_encoder(str(prompt))
        if len(encoded) == 3:
            condition, uncondition, pooled = encoded
            model_kwargs: dict[str, Any] = {}
        else:
            condition, uncondition, pooled, model_kwargs = encoded
        context = FrozenContext(str(prompt_id), np.asarray(pooled), self.backbone_revision)
        self._opaque_contexts[context.context_id] = (condition, uncondition, dict(model_kwargs))
        return context

    def policy_inputs(self, *, noise_seed: int, context: FrozenContext) -> dict:
        """Expose frozen native inputs for the separate noise-conditioned JS baseline."""
        import torch

        if context.backbone_revision != self.backbone_revision or context.context_id not in self._opaque_contexts:
            raise ValueError("SANA context does not belong to this frozen adapter.")
        condition, _, kwargs = self._opaque_contexts[context.context_id]
        mask = kwargs.get("mask")
        return {
            "noise": self.latent_factory(int(noise_seed), context).detach(),
            "text": condition[:, 0].detach(),
            "pooled": torch.tensor(context.embedding.copy(), device=condition.device)[None],
            "padding_mask": None if mask is None else ~mask.bool(),
        }

    def sample(
        self, *, noise_seed: int, context: FrozenContext, clock: Clock | JSSchedule
    ) -> tuple[Any, ExecutionTrace]:
        if context.backbone_revision != self.backbone_revision or context.context_id not in self._opaque_contexts:
            raise ValueError("SANA context does not belong to this frozen adapter.")
        if clock.target_nfe <= 0:
            raise ValueError("SANA clock requires positive NFE.")
        import torch

        condition, uncondition, model_kwargs = self._opaque_contexts[context.context_id]
        self.sampler.condition, self.sampler.uncondition = condition, uncondition
        self.sampler.model_kwargs = model_kwargs
        latents = self.latent_factory(int(noise_seed), context)
        device = condition.device
        _prepare_scheduler(self.sampler.scheduler, clock, device)
        do_cfg = self.sampler.cfg_scale > 1
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        field_calls = 0
        original_model = self.sampler.model
        original_set_timesteps = self.sampler.scheduler.set_timesteps

        def counted_model(*args: Any, **kwargs: Any) -> Any:
            nonlocal field_calls
            field_calls += 1
            return original_model(*args, **kwargs)

        # FlowEuler owns the solver loop. Its first action normally reconstructs
        # a native grid, so suppress only that reconstruction after the complete
        # requested grid has been validated and installed above.
        self.sampler.model = counted_model
        self.sampler.scheduler.set_timesteps = lambda *args, **kwargs: None
        try:
            with torch.inference_mode():
                latents = self.sampler.sample(latents, steps=clock.target_nfe)
                image = self.decoder(latents)
        finally:
            self.sampler.model = original_model
            self.sampler.scheduler.set_timesteps = original_set_timesteps
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        trace = ExecutionTrace(
            "runtime",
            clock.key,
            "euler",
            clock.target_nfe,
            field_calls,
            field_calls,
            field_calls * (2 if do_cfg else 1),
            elapsed,
        )
        return image, trace
