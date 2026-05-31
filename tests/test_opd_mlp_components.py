from __future__ import annotations

import inspect
import math
import unittest

import torch
from torch import nn

from genode.conditional_opd.models import ScheduleStudentMLP, ScheduleTeacherMLP, count_parameters, setting_features
from genode.conditional_opd.objectives import crps_mase_reward


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


class ConditionalOPDMLPComponentsTest(unittest.TestCase):
    def test_teacher_is_one_three_hidden_layer_width_256_scalar_scorer(self) -> None:
        torch.manual_seed(0)
        teacher = ScheduleTeacherMLP(input_dim=18)

        linear_layers = _linear_layers(teacher)

        self.assertEqual(len(linear_layers), 4)
        self.assertEqual(linear_layers[0].in_features, 18)
        for layer in linear_layers[:3]:
            self.assertEqual(layer.out_features, 256)
        self.assertEqual(linear_layers[3].in_features, 256)
        self.assertEqual(linear_layers[3].out_features, 1)

        utility = teacher(torch.randn(5, 18))

        self.assertEqual(tuple(utility.shape), (5,))
        self.assertTrue(torch.isfinite(utility).all())

    def test_student_is_smaller_two_hidden_layer_width_128_interval_policy(self) -> None:
        torch.manual_seed(1)
        setting_dim = int(setting_features("euler", 8).numel())
        teacher = ScheduleTeacherMLP(input_dim=setting_dim + 12)
        student = ScheduleStudentMLP(setting_dim=setting_dim, max_macro_steps=12)

        linear_layers = _linear_layers(student)

        self.assertLess(count_parameters(student), count_parameters(teacher))
        self.assertEqual(len(linear_layers), 3)
        self.assertEqual(linear_layers[0].in_features, setting_dim)
        self.assertEqual(linear_layers[0].out_features, 128)
        self.assertEqual(linear_layers[1].in_features, 128)
        self.assertEqual(linear_layers[1].out_features, 128)
        self.assertEqual(linear_layers[2].in_features, 128)
        self.assertEqual(linear_layers[2].out_features, 12)

        features = torch.stack([setting_features("euler", 8), setting_features("heun", 8)], dim=0)
        logits = student.interval_logits(features)
        intervals = student.intervals(features, macro_steps=4)
        grid = student.time_grid(features, macro_steps=4)

        self.assertEqual(tuple(logits.shape), (2, 12))
        self.assertEqual(tuple(intervals.shape), (2, 4))
        self.assertEqual(tuple(grid.shape), (2, 5))
        self.assertTrue(torch.all(intervals > 0.0))
        self.assertTrue(torch.allclose(intervals.sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.allclose(grid[:, 0], torch.zeros(2)))
        self.assertTrue(torch.allclose(grid[:, -1], torch.ones(2)))
        self.assertTrue(torch.all(torch.diff(grid, dim=-1) > 0.0))
        self.assertTrue(torch.allclose(torch.diff(grid, dim=-1), intervals, atol=1e-6))

    def test_reward_uses_crps_and_mase_only_without_penalty_branches(self) -> None:
        reward = crps_mase_reward(2.0, 3.0, crps_center=4.0, mase_center=6.0)
        worse_reward = crps_mase_reward(8.0, 12.0, crps_center=4.0, mase_center=6.0)

        self.assertAlmostEqual(reward, math.log(2.0))
        self.assertLess(worse_reward, reward)
        source = inspect.getsource(crps_mase_reward)
        self.assertIn("crps", source)
        self.assertIn("mase", source)
        self.assertNotIn("soft_penalty", source)
        self.assertNotIn("uncertainty", source)


if __name__ == "__main__":
    unittest.main()
