from __future__ import annotations

import unittest

import numpy as np

from genode.gico.density_representation import (
    DENSITY_PROTOCOL,
    density_mass_to_time_grid,
    density_metadata,
    grid_to_density_mass,
    uniform_reference_grid,
)


class DensityPrimitiveTests(unittest.TestCase):
    def test_density_grid_roundtrip_uses_canonical_64_bins(self) -> None:
        reference = uniform_reference_grid(64)
        source_grid = (0.0, 0.25, 0.5, 1.0)

        mass = grid_to_density_mass(source_grid, reference_time_grid=reference)
        reconstructed = density_mass_to_time_grid(mass, reference_time_grid=reference, macro_steps=3)
        metadata = density_metadata(reference)

        self.assertEqual(len(mass), 64)
        self.assertAlmostEqual(float(np.sum(mass)), 1.0, places=6)
        self.assertEqual(len(reconstructed), 4)
        self.assertEqual(metadata["density_protocol"], DENSITY_PROTOCOL)
        self.assertEqual(metadata["reference_bin_count"], 64)
