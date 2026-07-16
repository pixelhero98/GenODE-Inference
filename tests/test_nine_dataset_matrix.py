from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import torch

from genode.data.experiment_common import DATASET_PLANS, OTFLOW_PAPER_BACKBONE_PRESETS, OTFLOW_PAPER_DATASET_CHOICES
from genode.data.molecule_xyz import (
    MOLECULE_GROUP_DATASET_KEYS,
    build_balanced_molecule_stratum_groups,
    build_molecule_group_dataset_splits,
    discover_molecule_xyz_strata,
    prepare_molecule_xyz_group_datasets,
)
from genode.data.otflow_datasets import (
    LOBSTER_SYNTHETIC_DATASET_KEY,
    build_dataset_splits_from_lobster_synthetic,
    validate_lobster_synth_profile,
)
from genode.data.otflow_experiment_plan import (
    conditional_generation_dataset_keys,
    forecast_dataset_keys,
    experiment_plan_by_key,
)
from genode.data.otflow_medical_constants import LONG_TERM_ST_DATASET_KEY
from genode.data.otflow_monash_datasets import get_monash_dataset_spec, monash_paper_dataset_keys
from genode.evaluation import otflow_evaluation_support as eval_support
from genode.evaluation.fm_backbone_registry import CONDITIONAL_GENERATION_FAMILY
from genode.evaluation.otflow_evaluation_support import parse_conditional_generation_datasets, parse_forecast_datasets
from genode.evaluation.molecule_metrics import aggregate_molecule_group_evaluation
from genode.models.config import OTFlowConfig
from genode.training import train_molecule_backbone as train_molecule_module


def _profile(levels: int = 2) -> dict:
    return {
        "profiles": [
            {
                "rows": 1000,
                "tick_size": 0.01,
                "log_spread_mean": 0.2,
                "log_spread_std": 0.1,
                "spread_phi": 0.9,
                "imb_mean": 0.0,
                "imb_std": 0.1,
                "imb_phi": 0.9,
                "ret_scale_ticks": 0.1,
                "jump_prob_5ticks": 0.0,
                "jump_prob_2ticks": 0.0,
                "seasonality_abs_ret": [1.0, 1.0],
                "log_ask_gap_mean": [0.1] * max(1, levels - 1),
                "log_ask_gap_std": [0.01] * max(1, levels - 1),
                "log_bid_gap_mean": [0.1] * max(1, levels - 1),
                "log_bid_gap_std": [0.01] * max(1, levels - 1),
                "log_ask_vol_mean": [1.0] * levels,
                "log_ask_vol_std": [0.05] * levels,
                "log_bid_vol_mean": [1.0] * levels,
                "log_bid_vol_std": [0.05] * levels,
            }
        ]
    }


def _symbols(atom_count: int) -> list[str]:
    return ["C"] * max(1, atom_count - 2) + ["H"] * min(2, atom_count)


def _write_xyz_zip(path: Path, entries: dict[str, tuple[int, int]], *, frames: int = 7, root_level: bool = False) -> None:
    with ZipFile(path, "w") as zf:
        for category, (trajectory_count, atom_count) in entries.items():
            symbols = _symbols(atom_count)
            for idx in range(trajectory_count):
                rows = []
                for frame in range(frames):
                    rows.append(str(atom_count))
                    rows.append("")
                    for atom_idx, symbol in enumerate(symbols):
                        rows.append(f"{symbol} {0.1 * atom_idx + 0.01 * frame:.6f} {0.02 * idx:.6f} {0.03 * frame:.6f}")
                if root_level:
                    name = f"family_{category}_family_Iso{1000 + idx}.trj.xyz"
                else:
                    name = f"{category}/family_{category}_Iso{1000 + idx}.trj.xyz"
                zf.writestr(name, "\n".join(rows) + "\n")


class NineDatasetMatrixTests(unittest.TestCase):
    def test_exact_paper_dataset_lists_and_retired_parser_rejection(self) -> None:
        self.assertEqual(forecast_dataset_keys(), ("solar_energy_10m", "traffic_hourly", "weather_daily"))
        self.assertEqual(conditional_generation_dataset_keys(), ("cryptos", "lobster_synthetic", "long_term_st"))
        self.assertEqual(OTFLOW_PAPER_DATASET_CHOICES, ("cryptos", "lobster_synthetic", "long_term_st"))
        self.assertEqual(monash_paper_dataset_keys(), ("solar_energy_10m", "traffic_hourly", "weather_daily"))

        for retired in ("san_francisco_traffic", "wind_farms_wo_missing", "london_smart_meters_wo_missing", "electricity"):
            with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
                parse_forecast_datasets(retired)
        for retired in ("es_mbp_10", "sleep_edf"):
            with self.assertRaisesRegex(ValueError, "Unknown conditional-generation datasets"):
                parse_conditional_generation_datasets(retired)

    def test_paper_temporal_context_horizon_table_is_locked(self) -> None:
        expected = {
            "solar_energy_10m": (1008, 1008),
            "traffic_hourly": (336, 168),
            "weather_daily": (120, 30),
            "cryptos": (256, 128),
            "lobster_synthetic": (256, 128),
            "long_term_st": (12000, 3000),
        }
        plans = experiment_plan_by_key()
        for dataset_key, (history_len, future_block_len) in expected.items():
            spec = plans[dataset_key]
            self.assertEqual((spec.history_len, spec.future_block_len), (history_len, future_block_len))
            self.assertEqual(spec.experiment_horizon, future_block_len)

    def test_conditional_presets_and_plans_use_required_non_ar_horizons(self) -> None:
        for dataset_key in ("cryptos", LOBSTER_SYNTHETIC_DATASET_KEY):
            preset = OTFLOW_PAPER_BACKBONE_PRESETS[dataset_key]
            self.assertEqual(preset["rollout_mode"], "non_ar")
            self.assertEqual(preset["future_block_len"], 128)
            self.assertEqual(preset["history_len"], 256)
            self.assertEqual(DATASET_PLANS[dataset_key].horizons, (128, 128, 128))

    def test_strict_temporal_overrides_reject_nonpaper_lengths_and_rollout(self) -> None:
        args = SimpleNamespace(eval_horizon=167, future_block_len=0, rollout_mode="non_ar")
        with self.assertRaisesRegex(ValueError, "Non-paper --eval_horizon"):
            eval_support.resolved_eval_horizon(args, "traffic_hourly")

        args = SimpleNamespace(eval_horizon=0, future_block_len=127, rollout_mode="non_ar")
        with self.assertRaisesRegex(ValueError, "Non-paper --future_block_len"):
            eval_support.resolved_future_block_len(args, "cryptos")

        args = SimpleNamespace(eval_horizon=0, future_block_len=0, rollout_mode="autoregressive")
        with self.assertRaisesRegex(ValueError, "Non-paper --rollout_mode"):
            eval_support.resolved_rollout_mode(args, "weather_daily")

    def test_conditional_checkpoint_validation_rejects_ar_rollout_even_when_lengths_match(self) -> None:
        spec = experiment_plan_by_key()["cryptos"]
        metadata = {
            "dataset_key": "cryptos",
            "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
            "train_steps": 20000,
            "history_len": int(spec.history_len),
            "future_block_len": int(spec.future_block_len),
            "field_network_type": "transformer",
            "cond_dim": 0,
            "split_stats": {"cond_dim": 0, "history_len": int(spec.history_len)},
        }
        with self.assertRaisesRegex(RuntimeError, "rollout_mode mismatch"):
            eval_support._validate_conditional_generation_checkpoint_task(
                dataset="cryptos",
                ckpt_path=Path("model.pt"),
                metadata=metadata,
                checkpoint_model_cond_dim=0,
                checkpoint_train_steps=20000,
                checkpoint_history_len=int(spec.history_len),
                checkpoint_future_block_len=int(spec.future_block_len),
                checkpoint_rollout_mode="autoregressive",
                expected_train_steps=20000,
                expected_history_len=int(spec.history_len),
                expected_future_block_len=int(spec.future_block_len),
            )

    def test_molecule_3d_defaults_remain_variable_16_context_ar(self) -> None:
        parser = train_molecule_module.build_argparser()
        args = parser.parse_args(["--processed_dir", "data/molecule_xyz", "--stratum", "Dynamic_Test"])
        self.assertEqual(train_molecule_module.DEFAULT_HISTORY_LEN, 16)
        self.assertEqual(args.history_len, 16)
        self.assertEqual(args.future_horizon, 1)
        self.assertTrue(args.train_variable_context)
        self.assertEqual(args.train_context_max, 16)

    def test_monash_weather_and_traffic_specs_are_public_keys(self) -> None:
        traffic = get_monash_dataset_spec("traffic_hourly")
        weather = get_monash_dataset_spec("weather_daily")
        self.assertEqual(traffic.archive_name, "traffic_hourly_dataset.zip")
        self.assertEqual(traffic.official_horizon, 168)
        self.assertEqual(weather.archive_name, "weather_dataset.zip")
        self.assertEqual(weather.zenodo_record_id, 4654822)
        self.assertEqual(weather.source_frequency_label, "daily")
        self.assertEqual(weather.official_horizon, 30)

    def test_lobster_synthetic_profile_and_split_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "lobster_free_sample_profile_10.json"
            profile_path.write_text(json.dumps(_profile(levels=2)), encoding="utf-8")
            validate_lobster_synth_profile(json.loads(profile_path.read_text(encoding="utf-8")))
            cfg = OTFlowConfig(
                device=torch.device("cpu"),
                levels=2,
                token_dim=4,
                history_len=4,
                rollout_mode="non_ar",
                future_block_len=2,
                use_cond_features=False,
                use_time_features=False,
                use_time_gaps=False,
            )
            splits = build_dataset_splits_from_lobster_synthetic(
                str(profile_path),
                cfg,
                length=80,
                seed=7,
                stride_train=2,
                stride_eval=2,
                train_frac=0.6,
                val_frac=0.2,
            )
            hist, tgt, fut, meta = splits["train"][0]
            self.assertEqual(tuple(hist.shape), (4, 8))
            self.assertEqual(tuple(tgt.shape), (8,))
            self.assertEqual(tuple(fut.shape), (1, 8))
            self.assertEqual(splits["stats"]["dataset_kind"], LOBSTER_SYNTHETIC_DATASET_KEY)
            self.assertEqual(splits["train"].dataset_kind, LOBSTER_SYNTHETIC_DATASET_KEY)
            self.assertEqual(splits["train"].dataset_metadata["dataset_key"], LOBSTER_SYNTHETIC_DATASET_KEY)
            self.assertIn("t_global", meta)

    def test_long_term_st_is_active_conditional_generation(self) -> None:
        self.assertIn(LONG_TERM_ST_DATASET_KEY, conditional_generation_dataset_keys())
        self.assertEqual(parse_conditional_generation_datasets(LONG_TERM_ST_DATASET_KEY), [LONG_TERM_ST_DATASET_KEY])

    def test_molecule_group_discovery_balances_and_preserves_fixed_shape_strata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            root_zip = root / "trajectory.zip"
            strata_zip = root / "triangulene_3.zip"
            _write_xyz_zip(root_zip, {"Dynamic_RootFamily": (4, 5)}, root_level=True)
            _write_xyz_zip(
                strata_zip,
                {
                    "Dynamic_Alpha": (6, 6),
                    "Dynamic_Beta": (5, 7),
                    "Direct_Gamma": (3, 8),
                },
            )

            root_discovery = discover_molecule_xyz_strata(root_zip)
            self.assertEqual(tuple(root_discovery), ("Dynamic_RootFamily_family",))
            grouping = build_balanced_molecule_stratum_groups([root_zip, strata_zip])
            self.assertEqual(grouping["dataset_keys"], list(MOLECULE_GROUP_DATASET_KEYS))
            all_strata = [row["stratum"] for group in grouping["groups"] for row in group["strata"]]
            self.assertEqual(sorted(all_strata), ["Dynamic_Alpha", "Dynamic_Beta", "Dynamic_RootFamily_family"])
            self.assertNotIn("Direct_Gamma", all_strata)
            self.assertEqual(sorted(grouping["balance"]["group_trajectory_counts"]), [4, 5, 6])

            prepared = prepare_molecule_xyz_group_datasets([root_zip, strata_zip], root / "groups")
            self.assertEqual(set(prepared["manifests"]), set(MOLECULE_GROUP_DATASET_KEYS))
            first_group = build_molecule_group_dataset_splits(
                dataset_key=MOLECULE_GROUP_DATASET_KEYS[0],
                group_root=root / "groups",
                history_len=2,
                future_horizon=1,
            )
            for member in first_group["strata"].values():
                splits = member["splits"]
                split_ids = splits["data"].metadata["split_trajectory_ids"]
                self.assertFalse(set(split_ids["train"]) & set(split_ids["val"]))
                hist, tgt, _ = splits["train"][0]
                self.assertEqual(hist.shape[-1], splits["stats"]["context_feature_dim"])
                self.assertEqual(tgt.shape[-1], splits["stats"]["snapshot_dim"])
            group_summary = aggregate_molecule_group_evaluation(
                dataset_key=MOLECULE_GROUP_DATASET_KEYS[0],
                group_root=root / "groups",
                stratum_summaries=[
                    {
                        "stratum": str(member["member"]["stratum"]),
                        "examples": 2,
                        "metrics": {"all_first_horizon": {"molecule_kabsch_rmsd_3d": {"mean": 1.0}}},
                    }
                    for member in first_group["strata"].values()
                ],
            )
            self.assertEqual(group_summary["dataset_key"], MOLECULE_GROUP_DATASET_KEYS[0])
            self.assertEqual(group_summary["metrics"]["all_first_horizon"]["molecule_kabsch_rmsd_3d"]["mean"], 1.0)
            encoded = json.dumps(prepared)
            self.assertNotIn(str(root), encoded)


if __name__ == "__main__":
    unittest.main()
