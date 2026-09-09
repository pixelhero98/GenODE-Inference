from __future__ import annotations

import subprocess
import tomllib
import unittest
from pathlib import Path
from typing import Any, get_type_hints

from genode.gico.density_representation import density_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]


class PackagingHygieneTests(unittest.TestCase):
    def test_retired_task_modules_and_dependencies_are_absent(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        self.assertNotIn("medical", extras)
        dependencies = pyproject["project"]["dependencies"] + [dep for group in extras.values() for dep in group]
        self.assertFalse(any(str(dep).startswith(("wfdb", "pyedflib")) for dep in dependencies))
        for module in ("otflow_medical_constants", "otflow_medical_datasets", "experiment_common"):
            self.assertFalse((REPO_ROOT / "src" / "genode" / "data" / f"{module}.py").exists())

    def test_density_metadata_type_hints_resolve(self) -> None:
        hints = get_type_hints(density_metadata)

        self.assertEqual(hints["return"], dict[str, Any])

    def test_local_sensitive_and_generated_artifacts_are_ignored(self) -> None:
        candidates = (
            ".env",
            ".env.local",
            "credentials-private.json",
            "secrets-local.json",
            "private.key",
            "private.pem",
            "id_rsa",
            "id_dsa",
            "id_ecdsa",
            "id_ed25519",
            "model.pt",
            "model.pth",
            "model.ckpt",
            "array.npy",
            "array.npz",
            "results/run.json",
            "figures/plot.png",
            "wandb/run.json",
            "mlruns/experiment.json",
            "tensorboard/events.out.tfevents",
            "torchinductor_user/cache.bin",
            "tmpabcdefgh",
        )
        completed = subprocess.run(
            ["git", "check-ignore", "-z", "--stdin"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            input="\0".join(candidates) + "\0",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual({value for value in completed.stdout.split("\0") if value}, set(candidates))

        public_example = subprocess.run(
            ["git", "check-ignore", ".env.example"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(public_example.returncode, 1, public_example.stderr)
        self.assertEqual(public_example.stdout, "")

    def test_external_image_asset_terms_are_explicit(self) -> None:
        notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        for required_notice in (
            "CC-BY-NC-SA-4.0",
            "No separate license notice was found",
            "external, user-supplied dependencies",
            "weights-inception-2015-12-05-6726825d.pth",
            "It is not included in this",
            "GenODE neither downloads nor bundles it automatically",
        ):
            with self.subTest(required_notice=required_notice):
                self.assertIn(required_notice, notices)


if __name__ == "__main__":
    unittest.main()
