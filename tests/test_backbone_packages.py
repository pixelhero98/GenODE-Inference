from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from genode.backbone_packages import (
    PACKAGED_BACKBONE_MANIFEST,
    load_portable_backbone_manifest,
    package_backbone_family,
    validate_backbone_package,
)
from genode.data.otflow_experiment_plan import CONDITIONAL_GENERATION_FAMILY
from genode.canonical_experiment_layout import CANONICAL_CHECKPOINT_STEPS, SCENARIO_FAMILY_MOLECULE
from genode.data.otflow_experiment_plan import FORECAST_FAMILY

MOLECULE_FAMILY = SCENARIO_FAMILY_MOLECULE
TRAIN_BUDGET_STEPS = CANONICAL_CHECKPOINT_STEPS


def _write(path: Path, content: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)


class BackbonePackageTests(unittest.TestCase):
    def _source_tree(self, root: Path) -> None:
        scenarios = ("solar_energy_10m", "traffic_hourly", "weather_daily")
        artifacts = []
        for scenario in scenarios:
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                base = f"genode/outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}"
                ckpt_rel = f"{base}/model.pt"
                meta_rel = f"{base}/checkpoint_metadata.json"
                summary_rel = f"{base}/artifact_summary.json"
                _write(root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/model.pt")
                _write(
                    root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/checkpoint_metadata.json",
                    json.dumps(
                        {
                            "checkpoint_path": (
                                "/" + "scratch" + f"/example/genode/outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/model.pt"
                            )
                        }
                    ),
                )
                _write(
                    root / f"outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/artifact_summary.json",
                    json.dumps(
                        {
                            "summary_path": (
                                "/" + "projects" + f"/example/genode/outputs/backbone_matrix/otflow/temporal_extrapolation/{label}/{scenario}/artifact_summary.json"
                            )
                        }
                    ),
                )
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": FORECAST_FAMILY,
                        "dataset_key": scenario,
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"{scenario}_{label}",
                        "checkpoint_path": ckpt_rel,
                        "summary_path": summary_rel,
                        "metadata_path": meta_rel,
                        "status": "ready",
                        "seed": 0,
                    }
                )
            _write(root / f"paper_datasets/monash/{scenario}/manifest.json", "{}")
        for scenario in ("cryptos", "lobster_synthetic", "long_term_st"):
            for train_steps in TRAIN_BUDGET_STEPS:
                label = f"{int(train_steps) // 1000}k"
                artifacts.append(
                    {
                        "backbone_name": "otflow",
                        "benchmark_family": CONDITIONAL_GENERATION_FAMILY,
                        "dataset_key": scenario,
                        "train_steps": int(train_steps),
                        "train_budget_label": label,
                        "checkpoint_id": f"{scenario}_{label}",
                        "checkpoint_path": f"genode/outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/model.pt",
                        "summary_path": f"genode/outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/artifact_summary.json",
                        "metadata_path": f"genode/outputs/backbone_matrix/otflow/temporal_conditional_generation_transformer/{label}/{scenario}/checkpoint_metadata.json",
                        "status": "ready",
                        "seed": 0,
                    }
                )
        for scenario in ("molecule_3d_set1", "molecule_3d_set2", "molecule_3d_set3"):
            for member_idx in range(6):
                for train_steps in TRAIN_BUDGET_STEPS:
                    artifacts.append(
                        {
                            "backbone_name": "otflow_molecule_3d",
                            "benchmark_family": MOLECULE_FAMILY,
                            "dataset_key": scenario,
                            "member_key": f"member_{member_idx}",
                            "stratum": f"stratum_{member_idx}",
                            "variant": "ar_h1",
                            "train_steps": int(train_steps),
                            "train_budget_label": f"{int(train_steps) // 1000}k",
                            "checkpoint_id": f"{scenario}_member_{member_idx}_{int(train_steps)}",
                            "checkpoint_path": f"outputs/molecule_3d_backbones/{scenario}/member_{member_idx}/ar_h1/{int(train_steps)}_steps/model.pt",
                            "summary_path": f"outputs/molecule_3d_backbones/{scenario}/member_{member_idx}/ar_h1/{int(train_steps)}_steps/artifact_summary.json",
                            "metadata_path": f"outputs/molecule_3d_backbones/{scenario}/member_{member_idx}/ar_h1/{int(train_steps)}_steps/checkpoint_metadata.json",
                            "status": "ready",
                            "seed": 0,
                        }
                    )
        manifest = {
            "version": "fm_backbone_manifest",
            "artifact_count": len(artifacts),
            "ready_count": len(artifacts),
            "missing_count": 0,
            "artifacts": artifacts,
        }
        _write(root / "outputs/backbone_matrix/backbone_manifest.json", json.dumps(manifest))

    def test_package_family_writes_relative_clean_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)

            summary = package_backbone_family(
                family="temporal-extrapolation",
                source_root=source_root,
                output_dir=output_dir,
                overwrite=True,
                make_zip=False,
            )

            package_root = Path(summary["package_root"])
            validation = validate_backbone_package(package_root, expected_family="temporal-extrapolation")
            self.assertEqual(validation["status"], "complete", validation.get("errors"))
            raw_manifest = json.loads((package_root / PACKAGED_BACKBONE_MANIFEST).read_text(encoding="utf-8"))
            artifact = raw_manifest["artifacts"][0]
            self.assertEqual(artifact["checkpoint_path"], "outputs/backbone_matrix/otflow/temporal_extrapolation/4k/solar_energy_10m/model.pt")
            self.assertEqual(raw_manifest["path_base"], "../..")
            loaded = load_portable_backbone_manifest(package_root / PACKAGED_BACKBONE_MANIFEST)
            self.assertTrue(Path(loaded["artifacts"][0]["checkpoint_path"]).exists())

    def test_pipeline_rejects_backbone_training_with_provided_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "packages"
            self._source_tree(source_root)
            package_root = Path(
                package_backbone_family(
                    family="temporal-extrapolation",
                    source_root=source_root,
                    output_dir=output_dir,
                    overwrite=True,
                    make_zip=False,
                )["package_root"]
            )
            try:
                from genode.pipeline.full_pipeline import build_argparser, run_full_pipeline
            except Exception as exc:  # pragma: no cover - exercised only in minimal dependency environments.
                self.skipTest(f"full pipeline dependencies are unavailable: {exc}")
            args = build_argparser().parse_args(
                [
                    "--scenario_key",
                    "solar_energy_10m",
                    "--run_root",
                    str(Path(tmpdir) / "run"),
                    "--backbone_package_root",
                    str(package_root),
                    "--dry_run",
                ]
            )
            with self.assertRaisesRegex(ValueError, "cannot include backbone_training"):
                run_full_pipeline(args)


if __name__ == "__main__":
    unittest.main()
