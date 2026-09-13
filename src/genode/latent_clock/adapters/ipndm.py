from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from genode.latent_clock.clocks import Clock
from genode.latent_clock.contracts import ExecutionTrace, FrozenContext


class VariableStepIPNDM:
    """Second-order AB integration in y=x/alpha, r=sigma/alpha.

    Unlike fixed-coefficient iPNDM, extrapolation uses the actual consecutive
    r intervals. The first update is Euler; every update makes one model call.
    """

    solver_key = "ipndm_v"
    protocol = "sigma_over_alpha_ab2_v1"

    def __init__(self, noise_schedule: Any) -> None:
        self.noise_schedule = noise_schedule

    def sample_simple(
        self,
        model_fn: Any,
        x: Any,
        timesteps: Any,
        evaluation_times: Any,
        *,
        order: int = 2,
        condition: Any = None,
        unconditional_condition: Any = None,
    ) -> Any:
        import torch

        if order != 2:
            raise ValueError("Variable-step iPNDM supports order 2 only.")
        if timesteps.ndim != 1 or timesteps.numel() < 2 or not torch.equal(timesteps, evaluation_times):
            raise ValueError("Variable-step iPNDM requires identical one-dimensional integration/evaluation grids.")
        if not torch.isfinite(timesteps).all() or not (timesteps[1:] < timesteps[:-1]).all():
            raise ValueError("Variable-step iPNDM requires finite, strictly decreasing times.")
        schedule = self.noise_schedule
        if timesteps[0] > schedule.T or timesteps[-1] < schedule.eps:
            raise ValueError("Variable-step iPNDM times lie outside the noise schedule.")
        # Compute interval ratios accurately, but preserve the latent/model dtype.
        times = timesteps.double()
        alpha = schedule.marginal_alpha(times)
        rho = schedule.marginal_std(times) / alpha
        intervals = rho[1:] - rho[:-1]
        if not torch.isfinite(rho).all() or not (alpha > 0).all() or not (intervals < 0).all():
            raise ValueError("Variable-step iPNDM requires finite, strictly decreasing sigma/alpha.")
        previous = None
        for step in range(len(intervals)):
            current = model_fn(x, timesteps[step].expand(x.shape[0]), condition, unconditional_condition)
            estimate = current
            if previous is not None:
                ratio = (intervals[step] / intervals[step - 1]).to(x)
                estimate = (1 + ratio / 2) * current - (ratio / 2) * previous
            x = (alpha[step + 1] / alpha[step]).to(x) * x + (alpha[step + 1] * intervals[step]).to(x) * estimate
            previous = current
        return x


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
        solver_key: str | None = None,
    ) -> None:
        self.model_fn, self.decoder, self.solver, self.noise_schedule = model_fn, decoder, solver, noise_schedule
        self.context_encoder, self.latent_factory = context_encoder, latent_factory
        self.backbone_revision, self.order = str(backbone_revision), int(order)
        declared_solver = getattr(solver, "solver_key", "ipndm")
        solver_key = declared_solver if solver_key is None else solver_key
        if solver_key not in {"ipndm", "ipndm_v"} or solver_key != declared_solver:
            raise ValueError("SD1.5 solver identity differs from its implementation.")
        self.solver_key = solver_key
        if self.order != 2:
            raise ValueError("SD1.5 iPNDM adapters support order 2 only.")
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
            self.solver_key,
            nfe,
            field_calls,
            field_calls,
            2 * field_calls,
            elapsed,
        )
        return image, trace
