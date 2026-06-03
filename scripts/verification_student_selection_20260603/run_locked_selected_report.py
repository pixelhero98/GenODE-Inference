from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

from genode.gipo.report_locked_test import report_gipo_locked_test


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_locked_report(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    matrix = _read_json(root / "summary" / "student_selection_matrix.json")
    selected_run_id = str(matrix.get("selected_run_id") or "")
    if not selected_run_id:
        raise SystemExit("student_selection_matrix.json has no selected_run_id")
    manifest = _read_json(root / "selection_inputs" / "student_selection_inputs_manifest.json")
    run = next((dict(item) for item in manifest["runs"] if str(item["run_id"]) == selected_run_id), None)
    if run is None:
        raise SystemExit(f"selected run {selected_run_id!r} not found in input manifest")
    source_root = Path(run["source_root"])
    cal_root = source_root / "calibration_inputs"
    out_dir = root / "locked_selected_report" / selected_run_id
    summary_path = out_dir / "locked_test_gipo_policy_summary.json"
    if summary_path.exists():
        return {
            "selected_run_id": selected_run_id,
            "locked_report": str(summary_path),
            "status": "skipped_existing",
            "locked_test_used_for_selection": False,
        }
    report_args = Namespace(
        gipo_student_checkpoint=str(run["student_checkpoint"]),
        training_summary=str(run["training_summary"]),
        context_rows=str(cal_root / "fixed_locked_test" / "fixed_locked_context_rows.csv")
        + ","
        + str(cal_root / "ser_locked_test" / "ser_locked_context_rows.csv"),
        context_embeddings_npz=str(cal_root / "locked_context_embeddings.npz"),
        split_phase="locked_test",
        selection_mode="reporting",
        report_label="",
        out_dir=str(out_dir),
        baseline_rows=str(cal_root / "fixed_locked_test" / "fixed_locked_rows.csv"),
        comparator_rows=str(cal_root / "ser_locked_test" / "ser_locked_rows.csv"),
        dataset=str(args.dataset),
        dataset_root=str(args.dataset_root),
        shared_backbone_root=str(args.shared_backbone_root),
        backbone_manifest=str(args.backbone_manifest),
        output_root=str(root),
        otflow_train_steps=int(args.otflow_train_steps),
        device=str(args.device),
        seeds=str(args.seeds),
        solver_names=str(args.solver_names),
        target_nfe_values=str(args.target_nfe_values),
        num_eval_samples=int(args.num_eval_samples),
        forecast_eval_batch_size=int(args.forecast_eval_batch_size),
    )
    report_gipo_locked_test(report_args)
    status = {
        "selected_run_id": selected_run_id,
        "source_root": str(source_root),
        "locked_report": str(summary_path),
        "locked_test_used_for_selection": False,
        "status": "reported",
    }
    (root / "summary" / "locked_selected_report_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return status


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run locked-test report for the calibration-selected student policy.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="solar_energy_10m")
    parser.add_argument("--dataset_root", default="/projects/b35z/genode/paper_datasets")
    parser.add_argument("--shared_backbone_root", default="/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix")
    parser.add_argument("--backbone_manifest", default="/scratch/b35z/pixelhero.b35z/genode/outputs/backbone_matrix/backbone_manifest.json")
    parser.add_argument("--otflow_train_steps", type=int, default=20000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--solver_names", default="euler,heun,midpoint_rk2,dpmpp2m")
    parser.add_argument("--target_nfe_values", default="4,8,12")
    parser.add_argument("--num_eval_samples", type=int, default=5)
    parser.add_argument("--forecast_eval_batch_size", type=int, default=64)
    return parser


def main() -> None:
    print(json.dumps(run_locked_report(build_argparser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
