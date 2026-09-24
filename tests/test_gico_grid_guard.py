"""One guarded density realization across GICO execution surfaces."""

import numpy as np
import pytest
import torch

from genode.gico.clocks import guarded_density_mass, materialize
from genode.gico.density_representation import density_mass_to_time_grid
from genode.gico.networks import DENSITY_BINS, DENSITY_MIXTURE
from genode.schedules.policy import ScheduleBatch


def test_guarded_grid_matches_exported_schedule_and_exact_nfe():
    raw = np.zeros(64, dtype=np.float64)
    raw[0], raw[-1] = 0.5, 0.5
    guarded = guarded_density_mass(raw)
    canonical = materialize(raw, "euler", 8)
    assert canonical == density_mass_to_time_grid(
        (1 - DENSITY_MIXTURE) * raw + DENSITY_MIXTURE / DENSITY_BINS,
        macro_steps=8,
        eps=0,
    )
    unguarded = density_mass_to_time_grid(raw, macro_steps=8, eps=0)
    assert len(canonical) == 9
    assert canonical[0] == 0 and canonical[-1] == 1
    assert not np.array_equal(canonical, unguarded)

    schedule = ScheduleBatch(
        density_mass=torch.tensor(guarded[None], dtype=torch.float64),
        reference_time_grid=torch.linspace(0, 1, 65, dtype=torch.float64),
        time_grid=torch.tensor([canonical], dtype=torch.float64),
        target_nfe=8,
        gico_density_mass=torch.tensor(raw[None], dtype=torch.float64),
    )
    assert torch.equal(schedule.time_grid, torch.tensor([canonical], dtype=torch.float64))

    with pytest.raises(ValueError, match="shared density decoder"):
        ScheduleBatch(
            density_mass=torch.tensor(guarded[None], dtype=torch.float64),
            reference_time_grid=torch.linspace(0, 1, 65, dtype=torch.float64),
            time_grid=torch.tensor([unguarded], dtype=torch.float64),
            target_nfe=8,
            gico_density_mass=torch.tensor(raw[None], dtype=torch.float64),
        )
