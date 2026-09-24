"""Staged utility-surrogate evidence and native-context masking contracts."""

import pytest
import torch

from genode.gico.networks import ModelConfig, UtilitySurrogate
from genode.gico.profiles import resolve_profile
from genode.gico.training import _pooled_density_groups


def test_staged_profile_requires_fixed_horizon_and_contextual_evidence():
    profile = resolve_profile("sana", utility_surrogate_profile="density_context_projection")
    assert profile.utility_surrogate_steps == 2000
    with pytest.raises(ValueError, match="2,000 steps"):
        resolve_profile("sana", utility_surrogate_profile="density_context_projection", utility_surrogate_steps=500)
    with pytest.raises(ValueError, match="native context"):
        resolve_profile(
            "sana", utility_surrogate_profile="density_context_projection", utility_surrogate_context_mode="global"
        )


def test_density_targets_pool_only_aligned_fitting_contexts():
    def row(name, schedule):
        return {"solver": "euler", "nfe": 8, "density_sha256": name, "schedule_key": schedule}

    rows = [[row("uniform", "uniform"), row("candidate", "late")]] * 2
    groups = [
        (None, None, torch.tensor([[0.0, 0.0], [1.0, 3.0]]), 1.0),
        (None, None, torch.tensor([[0.0, 0.0], [3.0, 5.0]]), 1.0),
    ]
    pooled = _pooled_density_groups(groups, rows)
    for group in pooled:
        torch.testing.assert_close(group[2], torch.tensor([[0.0, 0.0], [2.0, 4.0]]))
    with pytest.raises(ValueError, match="complete aligned"):
        _pooled_density_groups(groups, [rows[0], [row("uniform", "uniform"), row("other", "late")]])


def test_density_only_mask_ignores_native_context_but_keeps_solver_features():
    model = UtilitySurrogate(ModelConfig(5, 2)).eval()
    model.native_context_width = 3
    model.density_only = True
    mass = torch.full((1, 64), 1 / 64, dtype=torch.float64)
    first = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.5]])
    other_context = torch.tensor([[9.0, 8.0, 7.0, 0.0, 0.5]])
    other_solver = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.5]])
    with torch.no_grad():
        torch.testing.assert_close(model(first, mass), model(other_context, mass), atol=0, rtol=0)
        assert not torch.allclose(model(first, mass), model(other_solver, mass))
