from __future__ import annotations

import json

import pytest

from genode.evaluation.diffusion_flow_time_reparameterization import _load_schedule_summary_cases
from genode.gico.schedule_hash import schedule_grid_hash


def write_summary(tmp_path, **prediction):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint_step": 4000,
                "schedules": [
                    {
                        "scheduler_key": "adaptive_density",
                        "predictions": [
                            {
                                "solver_key": "heun",
                                "target_nfe": 4,
                                "runtime_nfe": 2,
                                "time_grid": [0, 0.4, 1],
                                **prediction,
                            }
                        ],
                    }
                ],
            }
        )
    )
    return str(path)


def test_summary_grid_preserves_checkpoint_and_exact_nfe(tmp_path):
    cases = _load_schedule_summary_cases(write_summary(tmp_path))
    assert len(cases) == 1
    case = cases[0]
    assert case["checkpoint_step"] == 4000
    assert (case["runtime_nfe"], case["macro_steps"], case["realized_nfe"]) == (2, 2, 4)
    assert case["schedule_grid_hash"] == schedule_grid_hash([0, 0.4, 1])


@pytest.mark.parametrize(
    "prediction",
    [
        {"runtime_nfe": 4},
        {"realized_nfe": 2},
        {"time_grid": [0, 0.5, 0.4, 1]},
        {"time_grid": [0, 0.5]},
        {"time_grid": [0, float("nan"), 1]},
    ],
)
def test_summary_grid_rejects_invalid_solver_grid_and_nfe(tmp_path, prediction):
    with pytest.raises(ValueError):
        _load_schedule_summary_cases(write_summary(tmp_path, **prediction))
