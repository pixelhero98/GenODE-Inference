from __future__ import annotations

import math
import unittest
from decimal import Decimal

from genode.canonical_experiment_layout import CANONICAL_SUPERVISION_SCHEDULE_KEYS
from genode.benchmarks.image.protocol import (
    IMAGE_SCHEDULE_KEYS,
    euler_image_workload,
    image_protocol_metadata,
)
from genode.gico.policy import grid_for_schedule, validate_gico_support_schedule_keys
from genode.gico.train_gico import build_argparser as build_gico_argparser
from genode.pipeline import full_pipeline
from genode.evaluation import diffusion_flow_time_reparameterization as schedule_runner
from genode.schedule_transfer.diffusion_flow_schedules import load_external_schedule_catalog
from genode.schedule_transfer.reference_clocks import (
    AYS_SD15_SIGMAS,
    AYS_SD15_TIMESTEPS,
    DEFAULT_REFERENCE_CLOCK_KEYS,
    GITS_CIFAR10_SIGMAS,
    OTS_VP_LINEAR_OFFICIAL_LAMBDAS,
    OTS_VP_LINEAR_OFFICIAL_TIMES,
    OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS,
    OTS_VP_LINEAR_TABLE_SHA256,
    SD15_BETA_END,
    SD15_BETA_SCHEDULE,
    SD15_BETA_START,
    SD15_NUM_TRAIN_TIMESTEPS,
    _sd15_sigma_ratio_t0,
    build_reference_clock_grid,
    canonical_late_p_key,
    ots_vp_linear_source_nodes,
    parse_extra_late_p_values,
    reference_clock_keys,
    reference_clock_provenance,
    reference_clock_registry,
    reverse_reference_clock_grid,
    validate_late_p_value,
)
from genode.schedules.fixed import (
    build_default_fixed_schedules,
    build_fixed_schedule,
    default_fixed_schedule_specifications,
    validate_fixed_schedule_keys,
)
from genode.schedules.specification import ScheduleSpecification


EXPECTED_DEFAULT_KEYS = (
    "uniform",
    "ays_sd15_native",
    "ays_sd15_log_sigma",
    "gits_cifar10_native",
    "gits_cifar10_log_sigma",
    "ots_vp_linear_native",
    "ots_vp_linear_log_sigma",
    "late_p_1p5",
    "late_p_2",
    "late_p_4",
    "late_p_8",
    "flowts_power_0p03",
    "ays_sd15_native_reversed",
    "ays_sd15_log_sigma_reversed",
    "gits_cifar10_native_reversed",
    "gits_cifar10_log_sigma_reversed",
    "ots_vp_linear_native_reversed",
    "ots_vp_linear_log_sigma_reversed",
    "late_p_1p5_reversed",
    "late_p_2_reversed",
    "late_p_4_reversed",
    "late_p_8_reversed",
    "flowts_power_0p03_reversed",
)


class ReferenceClockTests(unittest.TestCase):
    def test_default_key_set_and_order_is_exactly_canonical_23(self) -> None:
        self.assertEqual(DEFAULT_REFERENCE_CLOCK_KEYS, EXPECTED_DEFAULT_KEYS)
        self.assertEqual(CANONICAL_SUPERVISION_SCHEDULE_KEYS, EXPECTED_DEFAULT_KEYS)
        self.assertEqual(tuple(reference_clock_registry()), EXPECTED_DEFAULT_KEYS)
        self.assertEqual(IMAGE_SCHEDULE_KEYS, EXPECTED_DEFAULT_KEYS)
        self.assertEqual(len(DEFAULT_REFERENCE_CLOCK_KEYS), 23)
        self.assertEqual(len(set(DEFAULT_REFERENCE_CLOCK_KEYS)), 23)
        runner_defaults = tuple(
            schedule_runner.build_argparser()
            .parse_args([])
            .baseline_scheduler_names.split(",")
        )
        self.assertEqual(runner_defaults, EXPECTED_DEFAULT_KEYS)
        for removed in ("late_power_3", "ser_ptg_local_defect_eta005", "ays_avg_reversed"):
            self.assertNotIn(removed, DEFAULT_REFERENCE_CLOCK_KEYS)

    def test_image_protocol_uses_dynamic_canonical_clock_count_and_provenance(self) -> None:
        metadata = image_protocol_metadata()
        self.assertEqual(metadata["protocol_key"], "image_euler_248_v6")
        self.assertEqual(metadata["schedule_count"], 23)
        self.assertEqual(tuple(metadata["schedule_keys"]), EXPECTED_DEFAULT_KEYS)
        self.assertEqual(len(metadata["reference_clock_provenance"]), 23)
        self.assertEqual(euler_image_workload().evidence_images, 2_070_000)
        self.assertEqual(
            euler_image_workload().backbone_image_evaluations,
            9_660_000,
        )
        for dataset in metadata["datasets"].values():
            self.assertEqual(
                set(dataset),
                {"key", "resolution", "class_count", "conditioning"},
            )
        augmented = image_protocol_metadata(extra_late_p_values="3")
        self.assertEqual(augmented["schedule_count"], 25)
        self.assertIn("late_p_3", augmented["schedule_keys"])
        self.assertIn("late_p_3_reversed", augmented["schedule_keys"])

    def test_published_source_node_goldens(self) -> None:
        self.assertEqual(AYS_SD15_TIMESTEPS, (999, 850, 736, 645, 545, 455, 343, 233, 124, 24))
        self.assertEqual(
            AYS_SD15_SIGMAS,
            (14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.0),
        )
        self.assertEqual(GITS_CIFAR10_SIGMAS, (80.0, 10.9836, 3.8811, 1.584, 0.5666, 0.1698, 0.002))
        ots_times, ots_lambdas = ots_vp_linear_source_nodes(4)
        expected_times = (1.0, 0.62910004, 0.41664592, 0.12023062, 0.001)
        expected_lambdas = (-5.02497841, -1.99115979, -0.79098555, 0.88994725, 4.55771493)
        for observed, expected in zip(ots_times, expected_times):
            self.assertAlmostEqual(observed, expected, places=5)
        for observed, expected in zip(ots_lambdas, expected_lambdas):
            self.assertAlmostEqual(observed, expected, places=5)

        self.assertEqual(
            OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS,
            (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20),
        )
        self.assertEqual(
            tuple(OTS_VP_LINEAR_OFFICIAL_TIMES),
            OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS,
        )
        self.assertEqual(
            tuple(OTS_VP_LINEAR_OFFICIAL_LAMBDAS),
            OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS,
        )
        for step_count in OTS_VP_LINEAR_SUPPORTED_STEP_COUNTS:
            with self.subTest(step_count=step_count):
                times, lambdas = ots_vp_linear_source_nodes(step_count)
                self.assertIs(times, OTS_VP_LINEAR_OFFICIAL_TIMES[step_count])
                self.assertIs(lambdas, OTS_VP_LINEAR_OFFICIAL_LAMBDAS[step_count])
                self.assertEqual(len(times), step_count + 1)
                self.assertEqual(len(lambdas), step_count + 1)
                self.assertTrue(all(left > right for left, right in zip(times, times[1:])))
                self.assertTrue(all(left < right for left, right in zip(lambdas, lambdas[1:])))
        self.assertEqual(
            OTS_VP_LINEAR_TABLE_SHA256,
            "ots-vp-linear-official-tables-v2:"
            "7686201dc408a0c9d56b3f9abed9849967426dc352c5f9870488ceee57ad9271",
        )
        step7_times, step7_lambdas = ots_vp_linear_source_nodes(7)
        self.assertEqual(
            step7_times,
            (
                0.9999999999999998,
                0.7688483584352114,
                0.625333564004778,
                0.5319211467485778,
                0.3404677956461694,
                0.1726471614387146,
                0.06195925711813602,
                0.0010000000000000588,
            ),
        )
        self.assertEqual(
            step7_lambdas,
            (
                -5.024978406659204,
                -2.9780097879125127,
                -1.9670130505029986,
                -1.4049914624110396,
                -0.4117932430386521,
                0.49891679292980207,
                1.5461919793144823,
                4.557714932729866,
            ),
        )
        with self.assertRaisesRegex(ValueError, "supported source step counts"):
            ots_vp_linear_source_nodes(9)
        with self.assertRaisesRegex(ValueError, "supported source step counts"):
            build_reference_clock_grid("ots_vp_linear_native", 9)

    def test_ays_log_sigma_terminal_uses_pinned_sd15_scaled_linear_config(self) -> None:
        self.assertEqual(SD15_NUM_TRAIN_TIMESTEPS, 1000)
        self.assertEqual(SD15_BETA_START, 0.00085)
        self.assertEqual(SD15_BETA_END, 0.012)
        self.assertEqual(SD15_BETA_SCHEDULE, "scaled_linear")
        self.assertAlmostEqual(_sd15_sigma_ratio_t0(), 0.029167158151720066, places=15)

    def test_native_and_log_sigma_are_distinct_coordinate_views(self) -> None:
        for native, log_sigma in (
            ("ays_sd15_native", "ays_sd15_log_sigma"),
            ("gits_cifar10_native", "gits_cifar10_log_sigma"),
            ("ots_vp_linear_native", "ots_vp_linear_log_sigma"),
        ):
            with self.subTest(native=native):
                native_grid = build_reference_clock_grid(native, 4)
                log_grid = build_reference_clock_grid(log_sigma, 4)
                self.assertNotEqual(native_grid, log_grid)
                self.assertEqual((native_grid[0], native_grid[-1]), (0.0, 1.0))
                self.assertEqual((log_grid[0], log_grid[-1]), (0.0, 1.0))

    def test_late_p_augmentation_is_validated_canonical_deduped_and_sorted(self) -> None:
        self.assertEqual(canonical_late_p_key("1.500"), "late_p_1p5")
        self.assertEqual(canonical_late_p_key(Decimal("3.2500")), "late_p_3p25")
        self.assertEqual(parse_extra_late_p_values("4,2.25,3,2.250,1.5"), (
            Decimal("1.5"), Decimal("2.25"), Decimal("3"), Decimal("4")
        ))
        keys = reference_clock_keys("3,2.25,3.0")
        self.assertEqual(keys[7:14], (
            "late_p_1p5", "late_p_2", "late_p_2p25", "late_p_3", "late_p_4", "late_p_8", "flowts_power_0p03"
        ))
        self.assertEqual(len(keys), len(set(keys)))
        for invalid in ("nan", "inf", "-inf", "1.4999", "8.0001"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, r"finite and within \[1.5, 8\]"):
                    validate_late_p_value(invalid)

    def test_reverse_is_an_involution_and_all_default_grids_are_valid(self) -> None:
        for key in DEFAULT_REFERENCE_CLOCK_KEYS:
            with self.subTest(key=key):
                grid = build_reference_clock_grid(key, 4)
                self.assertEqual(len(grid), 5)
                self.assertEqual((grid[0], grid[-1]), (0.0, 1.0))
                self.assertTrue(all(math.isfinite(value) for value in grid))
                self.assertTrue(all(right > left for left, right in zip(grid, grid[1:])))
                restored = reverse_reference_clock_grid(reverse_reference_clock_grid(grid))
                for observed, expected in zip(restored, grid):
                    self.assertAlmostEqual(observed, expected, places=14)

    def test_image_fixed_schedules_delegate_to_canonical_reference_clocks(self) -> None:
        specifications = default_fixed_schedule_specifications()
        self.assertEqual(
            tuple(item.schedule_key for item in specifications),
            EXPECTED_DEFAULT_KEYS,
        )
        schedules = build_default_fixed_schedules(4)
        self.assertEqual(len(schedules), 23)
        for schedule in schedules:
            with self.subTest(key=schedule.specification.schedule_key):
                expected_grid = build_reference_clock_grid(
                    schedule.specification.schedule_key,
                    4,
                )
                self.assertEqual(
                    tuple(float(value) for value in schedule.time_grid.tolist()),
                    expected_grid,
                )
                self.assertEqual(
                    schedule.clock_provenance,
                    reference_clock_provenance(schedule.specification.schedule_key),
                )

    def test_image_fixed_schedule_pool_accepts_only_complete_canonical_augmentation(self) -> None:
        keys = reference_clock_keys("2.25,3")
        self.assertEqual(validate_fixed_schedule_keys(keys), keys)
        self.assertEqual(
            len(default_fixed_schedule_specifications(extra_late_p_values="2.25,3")),
            27,
        )
        with self.assertRaisesRegex(ValueError, "complete canonical pool"):
            validate_fixed_schedule_keys(keys[:-1])
        for removed_key in ("late_power", "flowts", "ays", "uniform_reversed"):
            with self.subTest(removed_key=removed_key):
                with self.assertRaisesRegex(ValueError, "Unsupported image reference clock"):
                    build_fixed_schedule(ScheduleSpecification(removed_key), 4)
        with self.assertRaisesRegex(ValueError, "encoded in the canonical schedule key"):
            build_fixed_schedule(
                ScheduleSpecification("late_p_3", {"p": 3.0}),
                4,
            )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_gico_support_schedule_keys(("uniform", "uniform"))

    def test_provenance_is_pinned_and_honest_about_transfer(self) -> None:
        expected = {
            "ays_sd15_native": ("50e7158093710f9c1b4ea9ff100137a91c9228f3", "Apache-2.0"),
            "gits_cifar10_native": ("68d5ce427f261962b89ce3b0ee8f6b29f0577328", "Apache-2.0"),
            "ots_vp_linear_native": ("95d4ac6b8a3d1d389ab63a197e1b05d8512b6a99", "MIT"),
            "flowts_power_0p03": ("1ec35fb1d3d89d91a1607a9f949a515347d54c8c", "FMTS MIT"),
        }
        catalog = load_external_schedule_catalog()
        expected_catalog = {
            key: spec.as_dict()
            for key, spec in reference_clock_registry().items()
            if not key.endswith("_reversed") and spec.application_behavior == "transferred_reference"
        }
        self.assertEqual(catalog, expected_catalog)
        for key, (commit, license_name) in expected.items():
            with self.subTest(key=key):
                provenance = reference_clock_provenance(key)
                self.assertEqual(provenance["source_commit"], commit)
                self.assertEqual(provenance["source_license"], license_name)
                self.assertEqual(provenance["application_behavior"], "transferred_reference")
                self.assertTrue(provenance["source_model"])
                self.assertTrue(provenance["source_solver"])
                self.assertTrue(provenance["source_coordinate"])
                self.assertEqual(catalog[key]["source_commit"], commit)
        self.assertIn("CIFAR-10", reference_clock_provenance("gits_cifar10_native")["display_name"])

    def test_unknown_clock_errors_are_explicit(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown reference clock"):
            build_reference_clock_grid("mystery", 4)
        with self.assertRaisesRegex(KeyError, "Unknown reference clock"):
            reference_clock_provenance("mystery")
        with self.assertRaisesRegex(ValueError, "n_steps must be positive"):
            build_reference_clock_grid("uniform", 0)
        for invalid_steps in (True, False, 4.9):
            with self.subTest(invalid_steps=invalid_steps):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    build_reference_clock_grid("uniform", invalid_steps)
        with self.assertRaisesRegex(KeyError, "Unknown reference clock"):
            build_reference_clock_grid("uniform_reversed", 4)

    def test_trainer_cli_and_density_grid_accept_explicit_late_p_augmentation(self) -> None:
        args = build_gico_argparser().parse_args(
            [
                "--rows_csv", "rows.csv",
                "--context_embeddings_npz", "contexts.npz",
                "--out_dir", "out",
                "--extra_late_p_values", "3,2.25,3.0",
            ]
        )
        self.assertEqual(args.extra_late_p_values, "3,2.25,3.0")
        keys = ("uniform", "late_p_2p25", "late_p_2p25_reversed", "late_p_3", "late_p_3_reversed")
        self.assertEqual(validate_gico_support_schedule_keys(keys), keys)
        grid = grid_for_schedule("late_p_3_reversed", "euler", 4)
        self.assertEqual(len(grid), 5)
        self.assertTrue(all(right > left for left, right in zip(grid, grid[1:])))

        pipeline_args = full_pipeline.build_argparser().parse_args(
            ["--schedule_keys", "uniform,late_p_4", "--extra_late_p_values", "3,2.25,3.0"]
        )
        self.assertEqual(
            full_pipeline._requested_schedule_keys(pipeline_args),
            ["uniform", "late_p_2p25", "late_p_3", "late_p_4", "late_p_2p25_reversed", "late_p_3_reversed"],
        )


if __name__ == "__main__":
    unittest.main()
