from __future__ import annotations

from genode.solvers.euler import EulerSolver, integrate_euler
from genode.solvers.protocol import (
    EULER_SOLVER_KEY,
    Solver,
    SolverResult,
    SolverSpecification,
    VelocityField,
    validate_target_nfe,
)


__all__ = [
    "EULER_SOLVER_KEY",
    "EulerSolver",
    "Solver",
    "SolverResult",
    "SolverSpecification",
    "VelocityField",
    "integrate_euler",
    "validate_target_nfe",
]
