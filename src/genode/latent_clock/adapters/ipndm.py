from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from genode.latent_clock.clocks import Clock
from genode.latent_clock.contracts import ExecutionTrace, FrozenContext


def compile_ipndm_times(clock: Clock, *, epsilon: float = 0.001) -> np.ndarray:
    eps = float(epsilon)
    if not 0 < eps < 1:
        raise ValueError("iPNDM epsilon must lie strictly between zero and one.")
    times = 1.0 - np.asarray(clock.nodes, dtype=np.float64) * (1.0 - eps)
    if not np.all(np.diff(times) < 0) or times[0] != 1.0 or not np.isclose(times[-1], eps):
        raise RuntimeError("Failed to compile a complete iPNDM time grid.")
    return times


class IPNDMAdapter:
    solver_key = "ipndm"

    def __init__(
        self,
        *,
        model_fn: Any,
        decoder: Callable[[Any], Any],
        solver: Any,
        noise_schedule: Any,
        context_encoder: Callable[[str], tuple[Any, Any, np.ndarray]],
        latent_factory: Callable[[int, FrozenContext], Any],
        backbone_revision: str,
        order: int = 2,
    ) -> None:
        self.model_fn, self.decoder, self.solver, self.noise_schedule = model_fn, decoder, solver, noise_schedule
        self.context_encoder, self.latent_factory = context_encoder, latent_factory
        self.backbone_revision, self.order = str(backbone_revision), int(order)
        if self.order != 2:
            raise ValueError("The preregistered official LD3 comparison uses iPNDM order 2.")
        self._opaque_contexts: dict[str, tuple[Any, Any]] = {}

    def encode_context(self, prompt_id: str, prompt: str) -> FrozenContext:
        condition, uncondition, pooled = self.context_encoder(str(prompt))
        context = FrozenContext(str(prompt_id), np.asarray(pooled), self.backbone_revision)
        self._opaque_contexts[context.context_id] = (condition, uncondition)
        return context

    def sample(self, *, noise_seed: int, context: FrozenContext, clock: Clock) -> tuple[Any, ExecutionTrace]:
        times = compile_ipndm_times(clock, epsilon=float(self.noise_schedule.eps))
        return self.sample_times(
            noise_seed=noise_seed, context=context, integration_times=times, evaluation_times=times, clock_key=clock.key
        )

    def sample_times(
        self, *, noise_seed: int, context: FrozenContext, integration_times: Any, evaluation_times: Any, clock_key: str
    ) -> tuple[Any, ExecutionTrace]:
        if context.backbone_revision != self.backbone_revision or context.context_id not in self._opaque_contexts:
            raise ValueError("SD1.5 context does not belong to this frozen adapter.")
        import torch

        condition, uncondition = self._opaque_contexts[context.context_id]
        latent = self.latent_factory(int(noise_seed), context)
        times = torch.tensor(
            integration_times,
            dtype=torch.float32,
            device=latent.device,
        )
        eval_times = torch.tensor(evaluation_times, dtype=torch.float32, device=latent.device)
        if times.shape != eval_times.shape:
            raise ValueError("Integration and model-evaluation grids must have matching lengths.")
        nfe = len(times) - 1
        field_calls = 0

        def counted_model(*args, **kwargs):
            nonlocal field_calls
            field_calls += 1
            return self.model_fn(*args, **kwargs)

        torch.cuda.synchronize(latent.device)
        started = time.perf_counter()
        with torch.inference_mode():
            sample = self.noise_schedule.prior_transformation(latent)
            sample = self.solver.sample_simple(
                counted_model,
                sample,
                times,
                eval_times,
                order=self.order,
                condition=condition,
                unconditional_condition=uncondition,
            )
            image = self.decoder(sample)
        torch.cuda.synchronize(latent.device)
        elapsed = time.perf_counter() - started
        trace = ExecutionTrace(
            "runtime",
            clock_key,
            "ipndm",
            nfe,
            field_calls,
            field_calls,
            2 * field_calls,
            elapsed,
        )
        return image, trace
