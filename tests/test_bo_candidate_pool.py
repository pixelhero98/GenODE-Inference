from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from genode.conditional_opd.bo_candidate_pool import build_argparser as build_bo_argparser
from genode.conditional_opd.bo_candidate_pool import build_and_write_bo_candidate_pool, build_bo_candidate_pool, require_botorch
from genode.conditional_opd.clock_lowrank import (
    DEFAULT_DENSITY_GRID_SIZE,
    LOWRANK_THETA_DIM,
    schedule_grid_from_theta,
    sobol_thetas,
)
from genode.conditional_opd.models import solver_macro_steps, validate_time_grid
from genode.schedule_transfer.diffusion_flow_schedules import BASELINE_SCHEDULE_KEYS


SOLVERS = ("euler", "heun", "midpoint_rk2", "dpmpp2m")
NFES = (4, 8, 12)


def _uniform_grid(n_steps: int) -> list[float]:
    return [float(idx) / float(n_steps) for idx in range(int(n_steps) + 1)]


def _write_ser_summary(path: Path) -> None:
    predictions = []
    for solver in SOLVERS:
        for nfe in NFES:
            macro_steps = solver_macro_steps(solver, nfe)
            predictions.append(
                {
                    "solver_key": solver,
                    "target_nfe": int(nfe),
                    "macro_steps": int(macro_steps),
                    "time_grid": _uniform_grid(macro_steps),
                }
            )
    path.write_text(
        json.dumps(
            {
                "status": "ready",
                "dataset": "san_francisco_traffic",
                "schedules": [
                    {
                        "scheduler_key": "ser_ptg_local_defect_eta005",
                        "schedule_name": "SER-PTG test",
                        "predictions": predictions,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class BOCandidatePoolTests(unittest.TestCase):
    def test_botorch_import_preflight(self) -> None:
        versions = require_botorch()
        self.assertIn("botorch", versions)
        self.assertIn("gpytorch", versions)

    def test_lowrank_grid_is_deterministic_and_uses_128_bins(self) -> None:
        theta = sobol_thetas(1, seed=13)[0]
        self.assertEqual(len(theta), LOWRANK_THETA_DIM)
        first = schedule_grid_from_theta(theta, macro_steps=8, base_grid=_uniform_grid(8), grid_size=DEFAULT_DENSITY_GRID_SIZE)
        second = schedule_grid_from_theta(theta, macro_steps=8, base_grid=_uniform_grid(8), grid_size=DEFAULT_DENSITY_GRID_SIZE)
        self.assertEqual(first, second)
        self.assertEqual(validate_time_grid(first, macro_steps=8), first)

    def test_bo_candidate_pool_emits_complete_12_cell_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ser_path = Path(tmpdir) / "ser.json"
            _write_ser_summary(ser_path)
            pool = build_bo_candidate_pool(
                reference_schedule_summary=ser_path,
                active_round=0,
                seed=0,
                candidate_count=3,
                density_grid_size=128,
            )
        self.assertEqual(pool["candidate_source"], "bayesian_optimization")
        self.assertEqual(pool["density_grid_size"], 128)
        self.assertEqual(pool["schedule_count"], 3)
        for schedule in pool["schedules"]:
            self.assertEqual(schedule["candidate_source"], "bayesian_optimization")
            self.assertTrue(schedule["scheduler_key"].startswith("train20_v43_bo_r0_cand"))
            self.assertEqual(schedule["comparison_role"], "v43_pooled_bo_candidate")
            self.assertEqual(len(schedule["predictions"]), 12)
            self.assertTrue(all(item["density_grid_size"] == 128 for item in schedule["predictions"]))
            self.assertTrue(all(item["bo_theta_json"] for item in schedule["predictions"]))

    def test_bo_candidate_pool_no_longer_exposes_direct_selection_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_bo_argparser().parse_args(["--mode", "select"])

    def test_bo_candidate_pool_round1_uses_pooled_observation_rows(self) -> None:
        def write_rows(path: Path, rows: list[dict]) -> None:
            fields = ["dataset", "split_phase", "solver_key", "target_nfe", "scheduler_key", "seed", "crps", "mase"]
            import csv

            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ser_path = root / "ser.json"
            prior_path = root / "prior_bo.json"
            out_path = root / "round1_pool.json"
            _write_ser_summary(ser_path)
            prior_pool = build_bo_candidate_pool(
                reference_schedule_summary=ser_path,
                active_round=0,
                seed=0,
                candidate_count=1,
                density_grid_size=128,
            )
            prior_path.write_text(json.dumps(prior_pool), encoding="utf-8")
            candidate_key = prior_pool["schedules"][0]["scheduler_key"]

            fixed_rows: list[dict] = []
            candidate_rows: list[dict] = []
            for solver in SOLVERS:
                for nfe in NFES:
                    for idx, key in enumerate(BASELINE_SCHEDULE_KEYS):
                        base = {
                            "dataset": "san_francisco_traffic",
                            "solver_key": solver,
                            "target_nfe": str(nfe),
                            "scheduler_key": key,
                            "seed": "0",
                            "crps": str(1.0 + 0.01 * idx),
                            "mase": str(1.2 + 0.01 * idx),
                        }
                        fixed_rows.append({**base, "split_phase": "pooled_calibration"})
                    candidate_rows.append(
                        {
                            "dataset": "san_francisco_traffic",
                            "split_phase": "pooled_calibration",
                            "solver_key": solver,
                            "target_nfe": str(nfe),
                            "scheduler_key": candidate_key,
                            "seed": "0",
                            "crps": "0.95",
                            "mase": "1.1",
                        }
                    )

            fixed_csv = root / "fixed.csv"
            candidate_csv = root / "candidate.csv"
            write_rows(fixed_csv, fixed_rows)
            write_rows(candidate_csv, candidate_rows)

            args = build_bo_argparser().parse_args(
                [
                    "--reference_schedule_summary",
                    str(ser_path),
                    "--out_path",
                    str(out_path),
                    "--active_round",
                    "1",
                    "--candidate_count",
                    "2",
                    "--observed_schedule_summaries",
                    str(prior_path),
                    "--observed_rows_csv",
                    str(candidate_csv),
                    "--fixed_reference_rows_csv",
                    str(fixed_csv),
                ]
            )
            summary = build_and_write_bo_candidate_pool(args)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(payload["active_round"], 1)
        self.assertEqual(payload["schedule_count"], 2)
        self.assertTrue(all(meta["observation_count"] == 1 for meta in payload["bo_acquisition"].values()))


if __name__ == "__main__":
    unittest.main()
