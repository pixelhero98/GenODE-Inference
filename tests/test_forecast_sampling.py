from __future__ import annotations

import unittest

from genode.evaluation.otflow_evaluation_support import (
    TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
    choose_forecast_train_tuning_indices,
    train_tuning_target_example_count,
)


class ForecastSamplingTests(unittest.TestCase):
    def test_train_tuning_hash_sampling_is_deterministic_and_stratified(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 100

        first = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="sf")
        second = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="sf")
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 20)
        self.assertEqual(len({int(idx) // 5 for idx in first.tolist()}), 20)

    def test_train_tuning_target_count_matches_small_split_stratified_sampler(self) -> None:
        class FakeDataset:
            def __len__(self) -> int:
                return 10

        chosen = choose_forecast_train_tuning_indices(FakeDataset(), fraction=0.20, seed=7, strata=20, dataset="small")
        target = train_tuning_target_example_count(10, fraction=0.20, strata=20)

        self.assertEqual(len(chosen), 10)
        self.assertEqual(target, len(chosen))

    def test_validation_normalized_train_tuning_sampling_uses_holdout_scale(self) -> None:
        class FakeTrainDataset:
            def __len__(self) -> int:
                return 14_399_710

        first = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=0.20,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            sampling_mode=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
            reference_examples=862,
            train_split_fraction=0.70,
            val_split_fraction=0.10,
        )
        second = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=0.20,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            sampling_mode=TRAIN_TUNING_SAMPLING_MODE_VALIDATION_NORMALIZED,
            reference_examples=862,
            train_split_fraction=0.70,
            val_split_fraction=0.10,
        )
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 1207)
        self.assertEqual(len({int(idx) * 20 // 14_399_710 for idx in first.tolist()}), 20)

    def test_train_tuning_sampling_can_be_capped_before_large_candidate_materialization(self) -> None:
        class FakeTrainDataset:
            def __len__(self) -> int:
                return 1_000_000

        chosen = choose_forecast_train_tuning_indices(
            FakeTrainDataset(),
            fraction=1.0,
            seed=7,
            strata=20,
            dataset="traffic_hourly",
            max_examples=256,
        )

        self.assertEqual(len(chosen), 256)
        self.assertEqual(chosen.tolist(), sorted(chosen.tolist()))
        self.assertEqual(len(set(chosen.tolist())), 256)
