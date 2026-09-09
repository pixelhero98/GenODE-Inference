from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from genode.data.molecule_xyz import (
    MOLECULE_GROUP_DATASET_KEYS,
    build_balanced_molecule_stratum_groups,
    build_molecule_group_dataset_splits,
    discover_molecule_xyz_strata,
    prepare_molecule_xyz_group_datasets,
)
from genode.data.otflow_monash_datasets import get_monash_dataset_spec
from genode.evaluation.molecule_metrics import aggregate_molecule_group_evaluation
from genode.training import train_molecule_backbone as train_molecule_module


def _symbols(atom_count: int) -> list[str]:
    return ["C"] * max(1, atom_count - 2) + ["H"] * min(2, atom_count)


def _write_xyz_zip(
    path: Path, entries: dict[str, tuple[int, int]], *, frames: int = 7, root_level: bool = False
) -> None:
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


class DatasetMatrixTests(unittest.TestCase):
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

    def test_molecule_group_discovery_balances_and_preserves_fixed_shape_strata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            root_zip = root / "trajectory.zip"
            strata_zip = root / "triangulene_3.zip"
            _write_xyz_zip(root_zip, {"Dynamic_RootFamily": (4, 5)}, root_level=True)
            _write_xyz_zip(strata_zip, {"Dynamic_Alpha": (6, 6), "Dynamic_Beta": (5, 7), "Direct_Gamma": (3, 8)})
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
                dataset_key=MOLECULE_GROUP_DATASET_KEYS[0], group_root=root / "groups", history_len=2, future_horizon=1
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
