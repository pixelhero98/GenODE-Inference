from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from genode.canonical_experiment_layout import SCENARIO_FAMILY_FORECAST
from genode.gipo.preflight import build_argparser, preflight_gipo_rows


SUPPORT_SCHEDULES = ("uniform", "late_power_3")
ROW_FIELDS = (
    "benchmark_family",
    "dataset",
    "split_phase",
    "seed",
    "solver_key",
    "target_nfe",
    "scheduler_key",
    "context_id",
    "series_id",
    "target_t",
    "u_comp_uniform",
)


def _row(context_id: str, scheduler_key: str, *, series_id: str | None = None, target_t: int | None = None) -> dict[str, object]:
    suffix = context_id.rsplit("_", 1)[-1]
    context_idx = int(suffix) if suffix.isdigit() else 0
    return {
        "benchmark_family": SCENARIO_FAMILY_FORECAST,
        "dataset": "private_forecast_dataset",
        "split_phase": "train_tuning",
        "seed": 0,
        "solver_key": "euler",
        "target_nfe": 4,
        "scheduler_key": scheduler_key,
        "context_id": context_id,
        "series_id": series_id if series_id is not None else f"series_{context_idx}",
        "target_t": target_t if target_t is not None else 100 + context_idx,
        "u_comp_uniform": 1.0,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_preflight(rows_csv: Path, *extra: str):
    args = build_argparser().parse_args(
        [
            "--rows_csv",
            str(rows_csv),
            "--support_schedule_keys",
            ",".join(SUPPORT_SCHEDULES),
            "--teacher_metric_target_keys",
            "u_comp_uniform",
            *extra,
        ]
    )
    return preflight_gipo_rows(args)


class GipoPreflightTests(unittest.TestCase):
    def test_incomplete_cells_report_missing_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_csv = Path(tmpdir) / "rows.csv"
            _write_rows(
                rows_csv,
                [
                    _row("ctx_0", "uniform"),
                    _row("ctx_1", "uniform"),
                    _row("ctx_1", "late_power_3"),
                ],
            )

            report = _run_preflight(rows_csv)

        self.assertEqual(report["status"], "issues_found")
        self.assertEqual(report["support"]["missing_support_cell_count"], 1)
        self.assertEqual(report["support"]["missing_support_cells"][0]["missing_schedule_keys"], ["late_power_3"])
        self.assertEqual(report["support"]["complete_context_identity_clean_support_cell_count"], 1)

    def test_duplicate_cells_report_duplicate_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_csv = Path(tmpdir) / "rows.csv"
            _write_rows(
                rows_csv,
                [
                    _row("ctx_0", "uniform"),
                    _row("ctx_0", "uniform"),
                    _row("ctx_0", "late_power_3"),
                ],
            )

            report = _run_preflight(rows_csv)

        self.assertEqual(report["support"]["duplicate_support_cell_count"], 1)
        duplicate = report["support"]["duplicate_support_cells"][0]
        self.assertEqual(duplicate["scheduler_key"], "uniform")
        self.assertEqual(duplicate["count"], 2)
        self.assertEqual(report["support"]["complete_context_identity_clean_support_cell_count"], 0)

    def test_context_identity_conflict_blocks_complete_clean_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rows_csv = Path(tmpdir) / "rows.csv"
            _write_rows(
                rows_csv,
                [
                    _row("ctx", "uniform", series_id="series_a", target_t=10),
                    _row("ctx", "late_power_3", series_id="series_b", target_t=11),
                ],
            )

            report = _run_preflight(rows_csv)

        self.assertEqual(report["support"]["complete_support_cell_count"], 1)
        self.assertEqual(report["support"]["complete_context_identity_clean_support_cell_count"], 0)
        self.assertEqual(report["context_identity_conflict_count"], 1)
        self.assertEqual(report["context_identity_conflicts"][0]["type"], "context_id_multiple_identities")

    def test_complete_rows_csv_preserves_header_and_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows_csv = root / "rows.csv"
            complete_rows_csv = root / "complete.csv"
            _write_rows(
                rows_csv,
                [
                    _row("ctx_0", "uniform"),
                    _row("ctx_1", "late_power_3"),
                    _row("ctx_1", "uniform"),
                    _row("ctx_1", "ays"),
                ],
            )

            report = _run_preflight(rows_csv, "--complete_rows_csv", str(complete_rows_csv))

            with complete_rows_csv.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                exported_rows = list(reader)
                exported_header = reader.fieldnames

        self.assertEqual(report["support"]["missing_support_cell_count"], 1)
        self.assertEqual(report["support"]["extra_support_cell_count"], 1)
        self.assertEqual(report["complete_rows_row_count"], 2)
        self.assertEqual(exported_header, list(ROW_FIELDS))
        self.assertEqual([row["scheduler_key"] for row in exported_rows], ["late_power_3", "uniform"])
        self.assertEqual({row["context_id"] for row in exported_rows}, {"ctx_1"})


if __name__ == "__main__":
    unittest.main()
