from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from genode.latent_clock.clocks import Clock, bo_bounds, clock_from_interval_logits


@dataclass(frozen=True)
class BOObservation:
    query_index: int
    clock: Clock
    mean_utility: float
    variance_of_mean: float


def fit_bo_clock(
    target_nfe: int,
    evaluate: Callable[[Clock], tuple[float, float]],
    *,
    seed: int,
    total_queries: int = 25,
) -> tuple[Clock, tuple[BOObservation, ...]]:
    """Run the preregistered global BO search; each callback evaluates one full panel."""
    try:
        import torch
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise RuntimeError("BO fitting requires botorch and gpytorch in the clock-fitting environment.") from exc

    nfe = int(target_nfe)
    dimensions = nfe - 1
    if dimensions <= 0 or int(total_queries) != 25:
        raise ValueError("BO requires target_nfe >= 2 and exactly 25 full-panel queries.")
    lower, upper = bo_bounds(nfe)
    bounds = torch.tensor(np.stack([lower, upper]), dtype=torch.double)
    torch.manual_seed(int(seed))
    engine = torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=int(seed))
    initial_count = max(8, 2 * dimensions)
    minimum_width = 2.0**-24

    def width_constraint(x):
        logits = torch.cat((x, x.new_zeros(1)))
        return torch.log_softmax(logits, dim=-1).min() - math.log(minimum_width)

    def feasible_sobol(count):
        points = []
        while len(points) < count:
            values = bounds[0] + (bounds[1] - bounds[0]) * engine.draw(1)[0].double()
            if width_constraint(values) >= 0:
                points.append(values)
        return torch.stack(points)

    # Purely numerical proposals do not access prompts or rewards. Constrain
    # interval widths to float32 representability before any expensive query.
    initial = feasible_sobol(initial_count - 1)
    candidates = [np.zeros(dimensions, dtype=np.float64), *[row.numpy() for row in initial]]
    observations: list[BOObservation] = []

    def observe(values: np.ndarray) -> None:
        clock = clock_from_interval_logits(f"bo_query_{len(observations):02d}", values)
        mean, variance = (float(x) for x in evaluate(clock))
        if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0:
            raise ValueError("BO evaluator returned an invalid mean or variance.")
        observations.append(BOObservation(len(observations), clock, mean, max(variance, 1e-10)))

    for candidate in candidates:
        observe(np.asarray(candidate, dtype=np.float64))
    while len(observations) < int(total_queries):
        # Reconstruct interval logits; keeping this explicit protects the GP inputs from grid roundoff.
        from genode.latent_clock.clocks import interval_logits

        train_x = torch.tensor(np.stack([interval_logits(obs.clock) for obs in observations]), dtype=torch.double)
        train_y = torch.tensor([[obs.mean_utility] for obs in observations], dtype=torch.double)
        train_yvar = torch.tensor([[obs.variance_of_mean] for obs in observations], dtype=torch.double)
        covariance = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dimensions))
        model = SingleTaskGP(
            train_x,
            train_y,
            train_Yvar=train_yvar,
            covar_module=covariance,
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(d=dimensions, bounds=bounds),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acquisition = qLogNoisyExpectedImprovement(model=model, X_baseline=train_x)
        proposed, _ = optimize_acqf(
            acquisition,
            bounds=bounds,
            q=1,
            num_restarts=10,
            batch_initial_conditions=feasible_sobol(10).unsqueeze(1),
            nonlinear_inequality_constraints=[(width_constraint, True)],
            options={"seed": int(seed) + len(observations), "batch_limit": 1},
        )
        observe(proposed[0].detach().cpu().numpy())
    best = max(observations, key=lambda row: (row.mean_utility, -row.query_index))
    return Clock("bo_selected", best.clock.target_nfe, best.clock.nodes, "botorch_qlognei"), tuple(observations)
