from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from genode.data.experiment_common import OTFLOW_PAPER_DATASET_CHOICES
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
    canonical_conditional_generation_paper_dataset_keys,
    canonical_forecast_paper_dataset_keys,
)
from genode.data.otflow_medical_datasets import LONG_TERM_ST_DATASET_KEY
from genode.data.otflow_monash_datasets import get_monash_dataset_spec, monash_paper_dataset_keys
from genode.evaluation.otflow_evaluation_support import parse_conditional_generation_datasets, parse_forecast_datasets
from genode.evaluation.molecule_metrics import aggregate_molecule_group_evaluation
from genode.models.config import OTFlowConfig


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
    def test_exact_canonical_dataset_lists_and_retired_parser_rejection(self) -> None:
        self.assertEqual(canonical_forecast_paper_dataset_keys(), ("solar_energy_10m", "traffic_hourly", "weather_daily"))
        self.assertEqual(canonical_conditional_generation_paper_dataset_keys(), ("cryptos", "lobster_synthetic", "long_term_st"))
        self.assertEqual(OTFLOW_PAPER_DATASET_CHOICES, ("cryptos", "lobster_synthetic", "long_term_st"))
        self.assertEqual(monash_paper_dataset_keys(), ("solar_energy_10m", "traffic_hourly", "weather_daily"))

        for retired in ("san_francisco_traffic", "wind_farms_wo_missing", "london_smart_meters_wo_missing", "electricity"):
            with self.assertRaisesRegex(ValueError, "Unknown forecast datasets"):
                parse_forecast_datasets(retired)
        for retired in ("es_mbp_10", "sleep_edf"):
            with self.assertRaisesRegex(ValueError, "Unknown conditional-generation datasets"):
                parse_conditional_generation_datasets(retired)

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
        self.assertIn(LONG_TERM_ST_DATASET_KEY, canonical_conditional_generation_paper_dataset_keys())
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
                        "metrics": {"all_first_horizon": {"kabsch_rmsd": {"mean": 1.0}}},
                    }
                    for member in first_group["strata"].values()
                ],
            )
            self.assertEqual(group_summary["dataset_key"], MOLECULE_GROUP_DATASET_KEYS[0])
            self.assertEqual(group_summary["metrics"]["all_first_horizon"]["kabsch_rmsd"]["mean"], 1.0)
            encoded = json.dumps(prepared)
            self.assertNotIn(str(root), encoded)


if __name__ == "__main__":
    unittest.main()
