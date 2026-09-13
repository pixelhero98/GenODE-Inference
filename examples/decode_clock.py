"""Asset-free decoder example: exact denoiser-call budgets, no quality evaluation."""

from genode.gico.clocks import materialize, reference_densities
from genode.solver_protocol import solver_macro_steps

for solver in ("euler", "heun"):
    nfe = 8
    mass = reference_densities(solver, nfe)["late_p_3"]
    grid = materialize(mass, solver, nfe)
    assert grid[0] == 0 and grid[-1] == 1
    assert len(grid) == solver_macro_steps(solver, nfe) + 1
    print(solver, "NFE", nfe, "grid", grid)
